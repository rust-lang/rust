#![feature(custom_inner_attributes)]
#![rustfmt::skip]
#![warn(clippy::doc_paragraphs_missing_punctuation)]
//@no-rustfix

enum EmojiTrailers {
    /// Sometimes the doc comment ends with an emoji! 😅
    ExistingPunctuationBeforeEmoji,
    /// But it may still be missing punctuation 😢
    //~^ doc_paragraphs_missing_punctuation
    MissingPunctuationBeforeEmoji,
}
