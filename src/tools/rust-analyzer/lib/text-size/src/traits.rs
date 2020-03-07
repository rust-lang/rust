use {
    crate::{TextRange, TextSize},
    std::convert::TryInto,
};

/// Text-like structures that have a text size.
pub trait TextSized {
    /// The size of this text-alike.
    fn text_size(&self) -> TextSize;
}

impl TextSized for str {
    fn text_size(&self) -> TextSize {
        let len = self.len();
        TextSize::new(len)
    }
}

impl TextSized for char {
    fn text_size(&self) -> TextSize {
        self.len_utf8().try_into().unwrap()
    }
}

impl TextSized for TextRange {
    fn text_size(&self) -> TextSize {
        self.len()
    }
}
