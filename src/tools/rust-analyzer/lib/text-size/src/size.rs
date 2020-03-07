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
/// - `TextUnit::of_char(c)` ⟹ `TextSize::of(c)`
/// - `TextUnit::of_str(s)` ⟹ `TextSize:of(s)`
/// - `TextUnit::from_usize(size)` ⟹ `TextSize::new(size)`
/// - `unit.to_usize()` ⟹ `size.ix()`
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
    pub fn of(text: &impl TextSized) -> TextSize {
        text.text_size()
    }

    /// A text size for some `usize`.
    ///
    /// # Panics
    ///
    /// Panics if the size is greater than `u32::MAX` and debug assertions are
    /// enabled. If debug assertions are not enabled, wraps into `u32` space.
    pub fn new(size: usize) -> TextSize {
        if let Ok(size) = size.try_into() {
            size
        } else if cfg!(debug_assertions) {
            panic!("overflow when converting to TextSize");
        } else {
            TextSize(size as u32)
        }
    }

    /// Convert this text size into the standard indexing type.
    ///
    /// # Panics
    ///
    /// Panics if the size is greater than `usize::MAX`. This can only
    /// occur on targets where `size_of::<usize>() < size_of::<u32>()`.
    pub fn ix(self) -> usize {
        if let Ok(ix) = self.try_into() {
            ix
        } else {
            panic!("overflow when converting TextSize to usize index")
        }
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
    pub fn checked_add(self, rhs: impl TryInto<TextSize>) -> Option<TextSize> {
        let rhs = rhs.try_into().ok()?;
        self.raw.checked_add(rhs.raw).map(TextSize)
    }

    #[allow(missing_docs)]
    pub fn checked_sub(self, rhs: impl TryInto<TextSize>) -> Option<TextSize> {
        let rhs = rhs.try_into().ok()?;
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
            // Not `From` yet because of integer type fallback. We want e.g.
            // `TextSize::from(0)` and `size + 1` to work, and more `From`
            // impls means that this will try (and fail) to use i32 rather
            // than one of the unsigned integer types that actually work.
            conversions!(TryFrom<$lt> for TextSize);
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
    varries [usize i32] // i32 so that `checked_add($lit)` (`try_from($lit)`) can work
    // this will unfortunately have to hang around even if integer literal type fallback improves
}

impl Into<TextSize> for &'_ TextSize {
    fn into(self) -> TextSize {
        *self
    }
}

impl Into<TextSize> for &'_ mut TextSize {
    fn into(self) -> TextSize {
        *self
    }
}

macro_rules! op {
    (impl $Op:ident for TextSize by fn $f:ident = $op:tt) => {
        impl<IntoSize: Into<TextSize>> $Op<IntoSize> for TextSize {
            type Output = TextSize;
            fn $f(self, rhs: IntoSize) -> TextSize {
                TextSize(self.raw $op rhs.into().raw)
            }
        }
        impl<IntoSize> $Op<IntoSize> for &'_ TextSize
        where
            TextSize: $Op<IntoSize, Output = TextSize>,
        {
            type Output = TextSize;
            fn $f(self, rhs: IntoSize) -> TextSize {
                *self $op rhs
            }
        }
        impl<IntoSize> $Op<IntoSize> for &'_ mut TextSize
        where
            TextSize: $Op<IntoSize, Output = TextSize>,
        {
            type Output = TextSize;
            fn $f(self, rhs: IntoSize) -> TextSize {
                *self $op rhs
            }
        }
    };
}

op!(impl Add for TextSize by fn add = +);
op!(impl Sub for TextSize by fn sub = -);

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

impl iter::Sum for TextSize {
    fn sum<I: Iterator<Item = TextSize>>(iter: I) -> TextSize {
        iter.fold(TextSize::default(), Add::add)
    }
}

impl<'a> iter::Sum<&'a Self> for TextSize {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(TextSize::default(), Add::add)
    }
}
