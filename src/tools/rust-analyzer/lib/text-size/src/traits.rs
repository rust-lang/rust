use {
    crate::TextSize,
    std::{borrow::Cow, convert::TryInto, rc::Rc, sync::Arc},
};

/// Text-like structures that have a text size.
pub trait LenTextSize: Copy {
    /// The size of this text-alike.
    fn len_text_size(self) -> TextSize;
}

impl LenTextSize for &'_ str {
    #[inline]
    fn len_text_size(self) -> TextSize {
        self.len().try_into().unwrap()
    }
}

impl LenTextSize for char {
    #[inline]
    fn len_text_size(self) -> TextSize {
        (self.len_utf8() as u32).into()
    }
}

impl<D> LenTextSize for &'_ D
where
    D: LenTextSize + Copy,
{
    fn len_text_size(self) -> TextSize {
        D::len_text_size(*self)
    }
}

// Because we could not find a smart blanket impl to do this automatically and
// cleanly (rust-analyzer/text-size#36), just provide a bunch of manual impls.
// If a type fits in this macro and you need it to impl LenTextSize, just open
// a PR and we are likely to accept it. Or use `TextSize::of::<&str>` for now.
macro_rules! impl_lentextsize_for_string {
    ($($ty:ty),+ $(,)?) => {$(
        impl LenTextSize for $ty {
            #[inline]
            fn len_text_size(self) -> TextSize {
                <&str>::len_text_size(self)
            }
        }
    )+};
}

impl_lentextsize_for_string! {
    &Box<str>,
    &'_ String,
    &Cow<'_, str>,
    &Cow<'_, String>,
    &Arc<str>,
    &Arc<String>,
    &Rc<str>,
    &Rc<String>,
}
