use {
    crate::TextSize,
    std::{convert::TryInto, ops::Deref},
};

/// Text-like structures that have a text size.
pub trait TextSized: Copy {
    /// The size of this text-alike.
    fn text_size(self) -> TextSize;
}

impl TextSized for &'_ str {
    #[inline]
    fn text_size(self) -> TextSize {
        self.len()
            .try_into()
            .unwrap_or_else(|_| panic!("string too large ({}) for TextSize", self.len()))
    }
}

impl<D> TextSized for &'_ D
where
    D: Deref<Target = str>,
{
    #[inline]
    fn text_size(self) -> TextSize {
        self.deref().text_size()
    }
}

impl TextSized for char {
    #[inline]
    fn text_size(self) -> TextSize {
        (self.len_utf8() as u32).into()
    }
}

// assertion shape from static_assertions::assert_impl_all!
const _: fn() = || {
    use std::borrow::Cow;

    fn assert_impl<T: TextSized>() {}

    assert_impl::<&String>();
    assert_impl::<&Cow<str>>();

    struct StringLike {}
    impl Deref for StringLike {
        type Target = str;
        fn deref(&self) -> &str {
            unreachable!()
        }
    }

    assert_impl::<&StringLike>();
};
