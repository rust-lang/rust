use {
    crate::TextSize,
    std::{borrow::Cow, convert::TryInto, rc::Rc, sync::Arc},
};

/// Text-like structures that have a text size.
pub trait TextLen: Copy {
    /// The size of this text-alike.
    fn text_len(self) -> TextSize;
}

impl TextLen for &'_ str {
    #[inline]
    fn text_len(self) -> TextSize {
        self.len().try_into().unwrap()
    }
}

impl TextLen for char {
    #[inline]
    fn text_len(self) -> TextSize {
        (self.len_utf8() as u32).into()
    }
}

impl<D> TextLen for &'_ D
where
    D: TextLen + Copy,
{
    fn text_len(self) -> TextSize {
        D::text_len(*self)
    }
}

// Because we could not find a smart blanket impl to do this automatically and
// cleanly (rust-analyzer/text-size#36), just provide a bunch of manual impls.
// If a standard type fits in this macro and you need it to impl TextLen, just
// open a PR and we are likely to accept it. Or convince Rust to deref to &str.
macro_rules! impl_textlen_for_string {
    ($($ty:ty),+ $(,)?) => {$(
        impl TextLen for $ty {
            #[inline]
            fn text_len(self) -> TextSize {
                <&str>::text_len(self)
            }
        }
    )+};
}

impl_textlen_for_string! {
    &Box<str>,
    &String,
    &Cow<'_, str>,
    &Cow<'_, String>,
    &Arc<str>,
    &Arc<String>,
    &Rc<str>,
    &Rc<String>,
}
