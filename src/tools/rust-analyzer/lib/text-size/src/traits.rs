use {
    crate::TextSize,
    std::{convert::TryInto, ops::Deref},
};

/// Text-like structures that have a text size.
pub trait LenTextSize: Copy {
    /// The size of this text-alike.
    fn len_text_size(self) -> TextSize;
}

impl LenTextSize for &'_ str {
    #[inline]
    fn len_text_size(self) -> TextSize {
        self.len()
            .try_into()
            .unwrap_or_else(|_| panic!("string too large ({}) for TextSize", self.len()))
    }
}

impl<D> LenTextSize for &'_ D
where
    D: Deref<Target = str>,
{
    #[inline]
    fn len_text_size(self) -> TextSize {
        self.deref().len_text_size()
    }
}

impl LenTextSize for char {
    #[inline]
    fn len_text_size(self) -> TextSize {
        (self.len_utf8() as u32).into()
    }
}
