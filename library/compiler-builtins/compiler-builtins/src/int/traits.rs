pub use crate::support::{CastFrom, CastInto, Int, MinInt};

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
