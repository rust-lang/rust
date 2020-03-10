use {
    crate::TextSized,
    std::{
        convert::{TryFrom, TryInto},
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
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.raw, f)
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

macro_rules! conversions {
    (From<TextSize> for $gte:ident) => {
        impl From<TextSize> for $gte {
            fn from(value: TextSize) -> $gte {
                value.raw.into()
            }
        }
    };
    (From<$lte:ident> for TextSize) => {
        impl From<$lte> for TextSize {
            fn from(value: $lte) -> TextSize {
                TextSize(value.into())
            }
        }
    };
    (TryFrom<TextSize> for $lt:ident) => {
        impl TryFrom<TextSize> for $lt {
            type Error = TryFromIntError;
            fn try_from(value: TextSize) -> Result<$lt, Self::Error> {
                value.raw.try_into()
            }
        }
    };
    (TryFrom<$gt:ident> for TextSize) => {
        impl TryFrom<$gt> for TextSize {
            type Error = <$gt as TryInto<u32>>::Error;
            fn try_from(value: $gt) -> Result<TextSize, Self::Error> {
                value.try_into().map(TextSize)
            }
        }
    };
    {
        lt u32  [$($lt:ident)*]
        eq u32  [$($eq:ident)*]
        gt u32  [$($gt:ident)*]
        varries [$($var:ident)*]
    } => {
        $(
            conversions!(From<$lt> for TextSize);
            conversions!(TryFrom<TextSize> for $lt);
        )*

        $(
            conversions!(From<$eq> for TextSize);
            conversions!(From<TextSize> for $eq);
        )*

        $(
            conversions!(TryFrom<$gt> for TextSize);
            conversions!(From<TextSize> for $gt);
        )*

        $(
            conversions!(TryFrom<$var> for TextSize);
            conversions!(TryFrom<TextSize> for $var);
        )*
    };
}

conversions! {
    lt u32  [u8 u16]
    eq u32  [u32]
    gt u32  [u64]
    varries [usize]
}

macro_rules! arith {
    ($Op:ident $op:ident, $OpAssign:ident $op_assign:ident) => {
        impl $Op<TextSize> for TextSize {
            type Output = TextSize;
            fn $op(self, rhs: TextSize) -> TextSize {
                TextSize($Op::$op(self.raw, rhs.raw))
            }
        }
        impl $Op<TextSize> for &'_ TextSize {
            type Output = TextSize;
            fn $op(self, rhs: TextSize) -> TextSize {
                TextSize($Op::$op(self.raw, rhs.raw))
            }
        }
        impl $Op<&'_ TextSize> for TextSize {
            type Output = TextSize;
            fn $op(self, rhs: &TextSize) -> TextSize {
                TextSize($Op::$op(self.raw, rhs.raw))
            }
        }
        impl $Op<&'_ TextSize> for &'_ TextSize {
            type Output = TextSize;
            fn $op(self, rhs: &TextSize) -> TextSize {
                TextSize($Op::$op(self.raw, rhs.raw))
            }
        }

        impl<A> $OpAssign<A> for TextSize
        where
            TextSize: $Op<A, Output = TextSize>,
        {
            fn $op_assign(&mut self, rhs: A) {
                *self = $Op::$op(*self, rhs)
            }
        }
    };
}

arith!(Add add, AddAssign add_assign);
arith!(Sub sub, SubAssign sub_assign);

impl<A> iter::Sum<A> for TextSize
where
    TextSize: Add<A, Output = TextSize>,
{
    fn sum<I: Iterator<Item = A>>(iter: I) -> TextSize {
        iter.fold(TextSize::zero(), Add::add)
    }
}
