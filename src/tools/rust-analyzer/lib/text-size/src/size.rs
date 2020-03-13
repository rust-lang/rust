use {
    crate::TextSized,
    std::{
        convert::TryFrom,
        fmt, iter,
        num::TryFromIntError,
        ops::{Add, AddAssign, Sub, SubAssign},
        u32,
    },
};

/// A measure of text length. Also, equivalently, an index into text.
///
/// This is a utf8-bytes-offset stored as `u32`, but
/// most clients should treat it as an opaque measure.
///
/// # Translation from `text_unit`
///
/// - `TextUnit::of_char(c)`        ⟹ `TextSize::of(c)`
/// - `TextUnit::of_str(s)`         ⟹ `TextSize:of(s)`
/// - `TextUnit::from_usize(size)`  ⟹ `TextSize::try_from(size).unwrap_or_else(|| panic!(_))`
/// - `unit.to_usize()`             ⟹ `usize::try_from(size).unwrap_or_else(|| panic!(_))`
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextSize {
    pub(crate) raw: u32,
}

#[allow(non_snake_case)]
pub(crate) const fn TextSize(raw: u32) -> TextSize {
    TextSize { raw }
}

impl fmt::Debug for TextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

impl TextSize {
    /// The text size of some text-like object.
    pub fn of(text: impl TextSized) -> TextSize {
        text.text_size()
    }

    /// A size of zero.
    ///
    /// This is equivalent to `TextSize::default()` or [`TextSize::MIN`],
    /// but is more explicit on intent.
    pub const fn zero() -> TextSize {
        TextSize(0)
    }
}

/// Methods to act like a primitive integer type, where reasonably applicable.
//  Last updated for parity with Rust 1.42.0.
impl TextSize {
    /// The smallest representable text size. (`u32::MIN`)
    pub const MIN: TextSize = TextSize(u32::MIN);
    /// The largest representable text size. (`u32::MAX`)
    pub const MAX: TextSize = TextSize(u32::MAX);

    #[allow(missing_docs)]
    pub fn checked_add(self, rhs: TextSize) -> Option<TextSize> {
        self.raw.checked_add(rhs.raw).map(TextSize)
    }

    #[allow(missing_docs)]
    pub fn checked_sub(self, rhs: TextSize) -> Option<TextSize> {
        self.raw.checked_sub(rhs.raw).map(TextSize)
    }
}

impl From<u32> for TextSize {
    fn from(raw: u32) -> Self {
        TextSize { raw }
    }
}

impl From<TextSize> for u32 {
    fn from(value: TextSize) -> Self {
        value.raw
    }
}

impl TryFrom<usize> for TextSize {
    type Error = TryFromIntError;
    fn try_from(value: usize) -> Result<Self, TryFromIntError> {
        Ok(u32::try_from(value)?.into())
    }
}

impl From<TextSize> for usize {
    fn from(value: TextSize) -> Self {
        assert_lossless_conversion();
        return value.raw as usize;

        const fn assert_lossless_conversion() {
            [()][(std::mem::size_of::<usize>() < std::mem::size_of::<u32>()) as usize]
        }
    }
}

macro_rules! ops {
    (impl $Op:ident for TextSize by fn $f:ident = $op:tt) => {
        impl $Op<TextSize> for TextSize {
            type Output = TextSize;
            fn $f(self, other: TextSize) -> TextSize {
                TextSize(self.raw $op other.raw)
            }
        }
        impl $Op<&TextSize> for TextSize {
            type Output = TextSize;
            fn $f(self, other: &TextSize) -> TextSize {
                self $op *other
            }
        }
        impl<T> $Op<T> for &TextSize
        where
            TextSize: $Op<T, Output=TextSize>,
        {
            type Output = TextSize;
            fn $f(self, other: T) -> TextSize {
                *self $op other
            }
        }
    };
}

ops!(impl Add for TextSize by fn add = +);
ops!(impl Sub for TextSize by fn sub = -);

impl<A> AddAssign<A> for TextSize
where
    TextSize: Add<A, Output = TextSize>,
{
    fn add_assign(&mut self, rhs: A) {
        *self = *self + rhs
    }
}

impl<S> SubAssign<S> for TextSize
where
    TextSize: Sub<S, Output = TextSize>,
{
    fn sub_assign(&mut self, rhs: S) {
        *self = *self - rhs
    }
}

impl<A> iter::Sum<A> for TextSize
where
    TextSize: Add<A, Output = TextSize>,
{
    fn sum<I: Iterator<Item = A>>(iter: I) -> TextSize {
        iter.fold(TextSize::zero(), Add::add)
    }
}
