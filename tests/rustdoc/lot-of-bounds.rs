// This is a regression test for <https://github.com/rust-lang/rust/issues/107283>.

#![crate_name = "foo"]

use std::convert::TryFrom;
use std::iter::{Product, Sum};
use std::num::TryFromIntError;
use std::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};
use std::fmt::{Binary, Debug, Display, LowerExp, LowerHex, Octal, UpperExp, UpperHex};
use std::hash::Hash;

// @has 'foo/trait.Fixed.html'
// @snapshot bounds - '//*[@class="item-decl"]//code'
pub trait Fixed
where
    Self: Default + Hash + Ord,
    Self: Debug + Display + LowerExp + UpperExp,
    Self: Binary + Octal + LowerHex + UpperHex,
    Self: Add<Output = Self> + AddAssign,
    Self: Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign,
    Self: Div<Output = Self> + DivAssign,
    Self: Rem<Output = Self> + RemAssign,
    Self: Mul<<Self as Fixed>::Bits, Output = Self> + MulAssign<<Self as Fixed>::Bits>,
    Self: Div<<Self as Fixed>::Bits, Output = Self> + DivAssign<<Self as Fixed>::Bits>,
    Self: Rem<<Self as Fixed>::Bits, Output = Self> + RemAssign<<Self as Fixed>::Bits>,
    Self: Rem<<Self as Fixed>::NonZeroBits, Output = Self>,
    Self: RemAssign<<Self as Fixed>::NonZeroBits>,
    Self: Not<Output = Self>,
    Self: BitAnd<Output = Self> + BitAndAssign,
    Self: BitOr<Output = Self> + BitOrAssign,
    Self: BitXor<Output = Self> + BitXorAssign,
    Self: Shl<u32, Output = Self> + ShlAssign<u32>,
    Self: Shr<u32, Output = Self> + ShrAssign<u32>,
    Self: Sum + Product,
    Self: PartialOrd<i8> + PartialOrd<i16> + PartialOrd<i32>,
    Self: PartialOrd<i64> + PartialOrd<i128> + PartialOrd<isize>,
    Self: PartialOrd<u8> + PartialOrd<u16> + PartialOrd<u32>,
    Self: PartialOrd<u64> + PartialOrd<u128> + PartialOrd<usize>,
    Self: PartialOrd<f32> + PartialOrd<f64>,
{
    type Bits: From<Self::NonZeroBits>;
    type NonZeroBits: TryFrom<Self::Bits, Error = TryFromIntError>;
}
