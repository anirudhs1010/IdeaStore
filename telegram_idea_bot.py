import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def append_idea_to_md(message: str):
    lines = message.strip().split('\n', 1)
    if len(lines) == 2:
        title, description = lines
    else:
        title, description = lines[0], ''
    with open("ideas.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n# {title.strip()}\n{description.strip()}\n")

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me an idea in the format:\n# ideaname\ndescription')

def add_idea(update: Update, context: CallbackContext) -> None:
    append_idea_to_md(update.message.text)
    update.message.reply_text('Idea added to ideas.md!')

def main() -> None:
    updater = Updater("7554544125:AAGNKK9TQKVIgPHrOMQu7zZlFdw6FfPqw4U")
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, add_idea))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main() 