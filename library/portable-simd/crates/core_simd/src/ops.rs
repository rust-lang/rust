use crate::simd::{cmp::SimdPartialEq, LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::ops::{Add, Mul};
use core::ops::{BitAnd, BitOr, BitXor};
use core::ops::{Div, Rem, Sub};
use core::ops::{Shl, Shr};

mod assign;
mod deref;
mod shift_scalar;
mod unary;

impl<I, T, const N: usize> core::ops::Index<I> for Simd<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = I::Output;
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.as_array()[index]
    }
}

impl<I, T, const N: usize> core::ops::IndexMut<I> for Simd<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

macro_rules! unsafe_base {
    ($lhs:ident, $rhs:ident, {$simd_call:ident}, $($_:tt)*) => {
        // Safety: $lhs and $rhs are vectors
        unsafe { core::intrinsics::simd::$simd_call($lhs, $rhs) }
    };
}

/// SAFETY: This macro should not be used for anything except Shl or Shr, and passed the appropriate shift intrinsic.
/// It handles performing a bitand in addition to calling the shift operator, so that the result
/// is well-defined: LLVM can return a poison value if you shl, lshr, or ashr if `rhs >= <Int>::BITS`
/// At worst, this will maybe add another instruction and cycle,
/// at best, it may open up more optimization opportunities,
/// or simply be elided entirely, especially for SIMD ISAs which default to this.
///
// FIXME: Consider implementing this in cg_llvm instead?
// cg_clif defaults to this, and scalar MIR shifts also default to wrapping
macro_rules! wrap_bitshift {
    ($lhs:ident, $rhs:ident, {$simd_call:ident}, $int:ident) => {
        #[allow(clippy::suspicious_arithmetic_impl)]
        // Safety: $lhs and the bitand result are vectors
        unsafe {
            core::intrinsics::simd::$simd_call(
                $lhs,
                $rhs.bitand(Simd::splat(<$int>::BITS as $int - 1)),
            )
        }
    };
}

/// SAFETY: This macro must only be used to impl Div or Rem and given the matching intrinsic.
/// It guards against LLVM's UB conditions for integer div or rem using masks and selects,
/// thus guaranteeing a Rust value returns instead.
///
/// |                  | LLVM | Rust
/// | :--------------: | :--- | :----------
/// | N {/,%} 0        | UB   | panic!()
/// | <$int>::MIN / -1 | UB   | <$int>::MIN
/// | <$int>::MIN % -1 | UB   | 0
///
macro_rules! int_divrem_guard {
    (   $lhs:ident,
        $rhs:ident,
        {   const PANIC_ZERO: &'static str = $zero:literal;
            $simd_call:ident, $op:tt
        },
        $int:ident ) => {
        if $rhs.simd_eq(Simd::splat(0 as _)).any() {
            panic!($zero);
        } else {
            // Prevent otherwise-UB overflow on the MIN / -1 case.
            let rhs = if <$int>::MIN != 0 {
                // This should, at worst, optimize to a few branchless logical ops
                // Ideally, this entire conditional should evaporate
                // Fire LLVM and implement those manually if it doesn't get the hint
                ($lhs.simd_eq(Simd::splat(<$int>::MIN))
                // type inference can break here, so cut an SInt to size
                & $rhs.simd_eq(Simd::splat(-1i64 as _)))
                .select(Simd::splat(1 as _), $rhs)
            } else {
                // Nice base case to make it easy to const-fold away the other branch.
                $rhs
            };

            // aarch64 div fails for arbitrary `v % 0`, mod fails when rhs is MIN, for non-powers-of-two
            // these operations aren't vectorized on aarch64 anyway
            #[cfg(target_arch = "aarch64")]
            {
                let mut out = Simd::splat(0 as _);
                for i in 0..Self::LEN {
                    out[i] = $lhs[i] $op rhs[i];
                }
                out
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                // Safety: $lhs and rhs are vectors
                unsafe { core::intrinsics::simd::$simd_call($lhs, rhs) }
            }
        }
    };
}

macro_rules! for_base_types {
    (   T = ($($scalar:ident),*);
        type Lhs = Simd<T, N>;
        type Rhs = Simd<T, N>;
        type Output = $out:ty;

        impl $op:ident::$call:ident {
            $macro_impl:ident $inner:tt
        }) => {
            $(
                impl<const N: usize> $op<Self> for Simd<$scalar, N>
                where
                    $scalar: SimdElement,
                    LaneCount<N>: SupportedLaneCount,
                {
                    type Output = $out;

                    #[inline]
                    // TODO: only useful for int Div::div, but we hope that this
                    // will essentially always get inlined anyway.
                    #[track_caller]
                    fn $call(self, rhs: Self) -> Self::Output {
                        $macro_impl!(self, rhs, $inner, $scalar)
                    }
                }
            )*
    }
}

// A "TokenTree muncher": takes a set of scalar types `T = {};`
// type parameters for the ops it implements, `Op::fn` names,
// and a macro that expands into an expr, substituting in an intrinsic.
// It passes that to for_base_types, which expands an impl for the types,
// using the expanded expr in the function, and recurses with itself.
//
// tl;dr impls a set of ops::{Traits} for a set of types
macro_rules! for_base_ops {
    (
        T = $types:tt;
        type Lhs = Simd<T, N>;
        type Rhs = Simd<T, N>;
        type Output = $out:ident;
        impl $op:ident::$call:ident
            $inner:tt
        $($rest:tt)*
    ) => {
        for_base_types! {
            T = $types;
            type Lhs = Simd<T, N>;
            type Rhs = Simd<T, N>;
            type Output = $out;
            impl $op::$call
                $inner
        }
        for_base_ops! {
            T = $types;
            type Lhs = Simd<T, N>;
            type Rhs = Simd<T, N>;
            type Output = $out;
            $($rest)*
        }
    };
    ($($done:tt)*) => {
        // Done.
    }
}

// Integers can always accept add, mul, sub, bitand, bitor, and bitxor.
// For all of these operations, simd_* intrinsics apply wrapping logic.
for_base_ops! {
    T = (i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
    type Lhs = Simd<T, N>;
    type Rhs = Simd<T, N>;
    type Output = Self;

    impl Add::add {
        unsafe_base { simd_add }
    }

    impl Mul::mul {
        unsafe_base { simd_mul }
    }

    impl Sub::sub {
        unsafe_base { simd_sub }
    }

    impl BitAnd::bitand {
        unsafe_base { simd_and }
    }

    impl BitOr::bitor {
        unsafe_base { simd_or }
    }

    impl BitXor::bitxor {
        unsafe_base { simd_xor }
    }

    impl Div::div {
        int_divrem_guard {
            const PANIC_ZERO: &'static str = "attempt to divide by zero";
            simd_div, /
        }
    }

    impl Rem::rem {
        int_divrem_guard {
            const PANIC_ZERO: &'static str = "attempt to calculate the remainder with a divisor of zero";
            simd_rem, %
        }
    }

    // The only question is how to handle shifts >= <Int>::BITS?
    // Our current solution uses wrapping logic.
    impl Shl::shl {
        wrap_bitshift { simd_shl }
    }

    impl Shr::shr {
        wrap_bitshift {
            // This automatically monomorphizes to lshr or ashr, depending,
            // so it's fine to use it for both UInts and SInts.
            simd_shr
        }
    }
}

// We don't need any special precautions here:
// Floats always accept arithmetic ops, but may become NaN.
for_base_ops! {
    T = (f32, f64);
    type Lhs = Simd<T, N>;
    type Rhs = Simd<T, N>;
    type Output = Self;

    impl Add::add {
        unsafe_base { simd_add }
    }

    impl Mul::mul {
        unsafe_base { simd_mul }
    }

    impl Sub::sub {
        unsafe_base { simd_sub }
    }

    impl Div::div {
        unsafe_base { simd_div }
    }

    impl Rem::rem {
        unsafe_base { simd_rem }
    }
}
