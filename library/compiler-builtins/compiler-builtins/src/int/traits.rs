use core::ops;

/// Minimal integer implementations needed on all integer types, including wide integers.
#[allow(dead_code)]
pub trait MinInt:
    Copy
    + core::fmt::Debug
    + ops::BitOr<Output = Self>
    + ops::Not<Output = Self>
    + ops::Shl<u32, Output = Self>
{
    /// Type with the same width but other signedness
    type OtherSign: MinInt;
    /// Unsigned version of Self
    type UnsignedInt: MinInt;

    /// If `Self` is a signed integer
    const SIGNED: bool;

    /// The bitwidth of the int type
    const BITS: u32;

    const ZERO: Self;
    const ONE: Self;
    const MIN: Self;
    const MAX: Self;
}

/// Trait for some basic operations on integers
#[allow(dead_code)]
pub trait Int:
    MinInt
    + PartialEq
    + PartialOrd
    + ops::AddAssign
    + ops::SubAssign
    + ops::BitAndAssign
    + ops::BitOrAssign
    + ops::BitXorAssign
    + ops::ShlAssign<i32>
    + ops::ShrAssign<u32>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::Shr<u32, Output = Self>
    + ops::BitXor<Output = Self>
    + ops::BitAnd<Output = Self>
{
    /// LUT used for maximizing the space covered and minimizing the computational cost of fuzzing
    /// in `builtins-test`. For example, Self = u128 produces [0,1,2,7,8,15,16,31,32,63,64,95,96,
    /// 111,112,119,120,125,126,127].
    const FUZZ_LENGTHS: [u8; 20] = make_fuzz_lengths(<Self as MinInt>::BITS);

    /// The number of entries of `FUZZ_LENGTHS` actually used. The maximum is 20 for u128.
    const FUZZ_NUM: usize = {
        let log2 = (<Self as MinInt>::BITS - 1).count_ones() as usize;
        if log2 == 3 {
            // case for u8
            6
        } else {
            // 3 entries on each extreme, 2 in the middle, and 4 for each scale of intermediate
            // boundaries.
            8 + (4 * (log2 - 4))
        }
    };

    fn unsigned(self) -> Self::UnsignedInt;
    fn from_unsigned(unsigned: Self::UnsignedInt) -> Self;
    fn unsigned_abs(self) -> Self::UnsignedInt;

    fn from_bool(b: bool) -> Self;

    /// Prevents the need for excessive conversions between signed and unsigned
    fn logical_shr(self, other: u32) -> Self;

    /// Absolute difference between two integers.
    fn abs_diff(self, other: Self) -> Self::UnsignedInt;

    // copied from primitive integers, but put in a trait
    fn is_zero(self) -> bool;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_shl(self, other: u32) -> Self;
    fn wrapping_shr(self, other: u32) -> Self;
    fn rotate_left(self, other: u32) -> Self;
    fn overflowing_add(self, other: Self) -> (Self, bool);
    fn leading_zeros(self) -> u32;
    fn ilog2(self) -> u32;
}

pub(crate) const fn make_fuzz_lengths(bits: u32) -> [u8; 20] {
    let mut v = [0u8; 20];
    v[0] = 0;
    v[1] = 1;
    v[2] = 2; // important for parity and the iX::MIN case when reversed
    let mut i = 3;

    // No need for any more until the byte boundary, because there should be no algorithms
    // that are sensitive to anything not next to byte boundaries after 2. We also scale
    // in powers of two, which is important to prevent u128 corner tests from getting too
    // big.
    let mut l = 8;
    loop {
        if l >= ((bits / 2) as u8) {
            break;
        }
        // get both sides of the byte boundary
        v[i] = l - 1;
        i += 1;
        v[i] = l;
        i += 1;
        l *= 2;
    }

    if bits != 8 {
        // add the lower side of the middle boundary
        v[i] = ((bits / 2) - 1) as u8;
        i += 1;
    }

    // We do not want to jump directly from the Self::BITS/2 boundary to the Self::BITS
    // boundary because of algorithms that split the high part up. We reverse the scaling
    // as we go to Self::BITS.
    let mid = i;
    let mut j = 1;
    loop {
        v[i] = (bits as u8) - (v[mid - j]) - 1;
        if j == mid {
            break;
        }
        i += 1;
        j += 1;
    }
    v
}

macro_rules! int_impl_common {
    ($ty:ty) => {
        fn from_bool(b: bool) -> Self {
            b as $ty
        }

        fn logical_shr(self, other: u32) -> Self {
            Self::from_unsigned(self.unsigned().wrapping_shr(other))
        }

        fn is_zero(self) -> bool {
            self == Self::ZERO
        }

        fn wrapping_neg(self) -> Self {
            <Self>::wrapping_neg(self)
        }

        fn wrapping_add(self, other: Self) -> Self {
            <Self>::wrapping_add(self, other)
        }

        fn wrapping_mul(self, other: Self) -> Self {
            <Self>::wrapping_mul(self, other)
        }
        fn wrapping_sub(self, other: Self) -> Self {
            <Self>::wrapping_sub(self, other)
        }

        fn wrapping_shl(self, other: u32) -> Self {
            <Self>::wrapping_shl(self, other)
        }

        fn wrapping_shr(self, other: u32) -> Self {
            <Self>::wrapping_shr(self, other)
        }

        fn rotate_left(self, other: u32) -> Self {
            <Self>::rotate_left(self, other)
        }

        fn overflowing_add(self, other: Self) -> (Self, bool) {
            <Self>::overflowing_add(self, other)
        }

        fn leading_zeros(self) -> u32 {
            <Self>::leading_zeros(self)
        }

        fn ilog2(self) -> u32 {
            <Self>::ilog2(self)
        }
    };
}

macro_rules! int_impl {
    ($ity:ty, $uty:ty) => {
        impl MinInt for $uty {
            type OtherSign = $ity;
            type UnsignedInt = $uty;

            const BITS: u32 = <Self as MinInt>::ZERO.count_zeros();
            const SIGNED: bool = Self::MIN != Self::ZERO;

            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MIN: Self = <Self>::MIN;
            const MAX: Self = <Self>::MAX;
        }

        impl Int for $uty {
            fn unsigned(self) -> $uty {
                self
            }

            // It makes writing macros easier if this is implemented for both signed and unsigned
            #[allow(clippy::wrong_self_convention)]
            fn from_unsigned(me: $uty) -> Self {
                me
            }

            fn unsigned_abs(self) -> Self {
                self
            }

            fn abs_diff(self, other: Self) -> Self {
                self.abs_diff(other)
            }

            int_impl_common!($uty);
        }

        impl MinInt for $ity {
            type OtherSign = $uty;
            type UnsignedInt = $uty;

            const BITS: u32 = <Self as MinInt>::ZERO.count_zeros();
            const SIGNED: bool = Self::MIN != Self::ZERO;

            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MIN: Self = <Self>::MIN;
            const MAX: Self = <Self>::MAX;
        }

        impl Int for $ity {
            fn unsigned(self) -> $uty {
                self as $uty
            }

            fn from_unsigned(me: $uty) -> Self {
                me as $ity
            }

            fn unsigned_abs(self) -> Self::UnsignedInt {
                self.unsigned_abs()
            }

            fn abs_diff(self, other: Self) -> $uty {
                self.abs_diff(other)
            }

            int_impl_common!($ity);
        }
    };
}

int_impl!(isize, usize);
int_impl!(i8, u8);
int_impl!(i16, u16);
int_impl!(i32, u32);
int_impl!(i64, u64);
int_impl!(i128, u128);

/// Trait for integers twice the bit width of another integer. This is implemented for all
/// primitives except for `u8`, because there is not a smaller primitive.
pub trait DInt: MinInt {
    /// Integer that is half the bit width of the integer this trait is implemented for
    type H: HInt<D = Self>;

    /// Returns the low half of `self`
    fn lo(self) -> Self::H;
    /// Returns the high half of `self`
    fn hi(self) -> Self::H;
    /// Returns the low and high halves of `self` as a tuple
    fn lo_hi(self) -> (Self::H, Self::H) {
        (self.lo(), self.hi())
    }
    /// Constructs an integer using lower and higher half parts
    fn from_lo_hi(lo: Self::H, hi: Self::H) -> Self {
        lo.zero_widen() | hi.widen_hi()
    }
}

/// Trait for integers half the bit width of another integer. This is implemented for all
/// primitives except for `u128`, because it there is not a larger primitive.
pub trait HInt: Int {
    /// Integer that is double the bit width of the integer this trait is implemented for
    type D: DInt<H = Self> + MinInt;

    // NB: some of the below methods could have default implementations (e.g. `widen_hi`), but for
    // unknown reasons this can cause infinite recursion when optimizations are disabled. See
    // <https://github.com/rust-lang/compiler-builtins/pull/707> for context.

    /// Widens (using default extension) the integer to have double bit width
    fn widen(self) -> Self::D;
    /// Widens (zero extension only) the integer to have double bit width. This is needed to get
    /// around problems with associated type bounds (such as `Int<Othersign: DInt>`) being unstable
    fn zero_widen(self) -> Self::D;
    /// Widens the integer to have double bit width and shifts the integer into the higher bits
    fn widen_hi(self) -> Self::D;
    /// Widening multiplication with zero widening. This cannot overflow.
    fn zero_widen_mul(self, rhs: Self) -> Self::D;
    /// Widening multiplication. This cannot overflow.
    fn widen_mul(self, rhs: Self) -> Self::D;
}

macro_rules! impl_d_int {
    ($($X:ident $D:ident),*) => {
        $(
            impl DInt for $D {
                type H = $X;

                fn lo(self) -> Self::H {
                    self as $X
                }
                fn hi(self) -> Self::H {
                    (self >> <$X as MinInt>::BITS) as $X
                }
            }
        )*
    };
}

macro_rules! impl_h_int {
    ($($H:ident $uH:ident $X:ident),*) => {
        $(
            impl HInt for $H {
                type D = $X;

                fn widen(self) -> Self::D {
                    self as $X
                }
                fn zero_widen(self) -> Self::D {
                    (self as $uH) as $X
                }
                fn zero_widen_mul(self, rhs: Self) -> Self::D {
                    self.zero_widen().wrapping_mul(rhs.zero_widen())
                }
                fn widen_mul(self, rhs: Self) -> Self::D {
                    self.widen().wrapping_mul(rhs.widen())
                }
                fn widen_hi(self) -> Self::D {
                    (self as $X) << <Self as MinInt>::BITS
                }
            }
        )*
    };
}

impl_d_int!(u8 u16, u16 u32, u32 u64, u64 u128, i8 i16, i16 i32, i32 i64, i64 i128);
impl_h_int!(
    u8 u8 u16,
    u16 u16 u32,
    u32 u32 u64,
    u64 u64 u128,
    i8 u8 i16,
    i16 u16 i32,
    i32 u32 i64,
    i64 u64 i128
);

/// Trait to express (possibly lossy) casting of integers
pub trait CastInto<T: Copy>: Copy {
    fn cast(self) -> T;
}

pub trait CastFrom<T: Copy>: Copy {
    fn cast_from(value: T) -> Self;
}

impl<T: Copy, U: CastInto<T> + Copy> CastFrom<U> for T {
    fn cast_from(value: U) -> Self {
        value.cast()
    }
}

macro_rules! cast_into {
    ($ty:ty) => {
        cast_into!($ty; usize, isize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);
    };
    ($ty:ty; $($into:ty),*) => {$(
        impl CastInto<$into> for $ty {
            fn cast(self) -> $into {
                self as $into
            }
        }
    )*};
}

cast_into!(usize);
cast_into!(isize);
cast_into!(u8);
cast_into!(i8);
cast_into!(u16);
cast_into!(i16);
cast_into!(u32);
cast_into!(i32);
cast_into!(u64);
cast_into!(i64);
cast_into!(u128);
cast_into!(i128);
