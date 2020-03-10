use {crate::TextSize, std::convert::TryInto};

/// Text-like structures that have a text size.
pub trait TextSized: Copy {
    /// The size of this text-alike.
    fn text_size(self) -> TextSize;
}

impl TextSized for &'_ str {
    fn text_size(self) -> TextSize {
        let len = self.len();
        if let Ok(size) = len.try_into() {
            size
        } else if cfg!(debug_assertions) {
            panic!("overflow when converting to TextSize");
        } else {
            TextSize(len as u32)
        }
    }
}

impl TextSized for char {
    fn text_size(self) -> TextSize {
        TextSize(self.len_utf8() as u32)
    }
}
