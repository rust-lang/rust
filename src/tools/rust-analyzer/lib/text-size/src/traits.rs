use {
    crate::TextSize,
    std::{borrow::Cow, convert::TryInto, rc::Rc, sync::Arc},
};

use priv_in_pub::Sealed;
mod priv_in_pub {
    pub trait Sealed {}
}

/// Primitives with a textual length that can be passed to [`TextSize::of`].
pub trait TextLen: Copy + Sealed {
    /// The textual length of this primitive.
    fn text_len(self) -> TextSize;
}

impl Sealed for &'_ str {}
impl TextLen for &'_ str {
    #[inline]
    fn text_len(self) -> TextSize {
        self.len().try_into().unwrap()
    }
}

impl Sealed for char {}
impl TextLen for char {
    #[inline]
    fn text_len(self) -> TextSize {
        (self.len_utf8() as u32).into()
    }
}

impl<D> Sealed for &'_ D where D: TextLen + Copy {}
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
        impl Sealed for $ty {}
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
