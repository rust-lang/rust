use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::ops::{Add, Mul};
use core::ops::{BitAnd, BitOr, BitXor};
use core::ops::{Div, Rem, Sub};
use core::ops::{Shl, Shr};

mod assign;
mod deref;
mod unary;

impl<I, T, const LANES: usize> core::ops::Index<I> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.as_array()[index]
    }
}

impl<I, T, const LANES: usize> core::ops::IndexMut<I> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

macro_rules! unsafe_base {
    ($lhs:ident, $rhs:ident, {$simd_call:ident}, $($_:tt)*) => {
        unsafe { $crate::simd::intrinsics::$simd_call($lhs, $rhs) }
    };
}

/// SAFETY: This macro should not be used for anything except Shl or Shr, and passed the appropriate shift intrinsic.
/// It handles performing a bitand in addition to calling the shift operator, so that the result
/// is well-defined: LLVM can return a poison value if you shl, lshr, or ashr if rhs >= <Int>::BITS
/// At worst, this will maybe add another instruction and cycle,
/// at best, it may open up more optimization opportunities,
/// or simply be elided entirely, especially for SIMD ISAs which default to this.
///
// FIXME: Consider implementing this in cg_llvm instead?
// cg_clif defaults to this, and scalar MIR shifts also default to wrapping
macro_rules! wrap_bitshift {
    ($lhs:ident, $rhs:ident, {$simd_call:ident}, $int:ident) => {
        unsafe {
            $crate::simd::intrinsics::$simd_call(
                $lhs,
                $rhs.bitand(Simd::splat(<$int>::BITS as $int - 1)),
            )
        }
    };
}

// Division by zero is poison, according to LLVM.
// So is dividing the MIN value of a signed integer by -1,
// since that would return MAX + 1.
// FIXME: Rust allows <SInt>::MIN / -1,
// so we should probably figure out how to make that safe.
macro_rules! int_divrem_guard {
    (   $lhs:ident,
        $rhs:ident,
        {   const PANIC_ZERO: &'static str = $zero:literal;
            const PANIC_OVERFLOW: &'static str = $overflow:literal;
            $simd_call:ident
        },
        $int:ident ) => {
        if $rhs.lanes_eq(Simd::splat(0)).any() {
            panic!($zero);
        } else if <$int>::MIN != 0
            && ($lhs.lanes_eq(Simd::splat(<$int>::MIN))
                // type inference can break here, so cut an SInt to size
                & $rhs.lanes_eq(Simd::splat(-1i64 as _))).any()
        {
            panic!($overflow);
        } else {
            unsafe { $crate::simd::intrinsics::$simd_call($lhs, $rhs) }
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
                    #[must_use = "operator returns a new vector without mutating the inputs"]
                    fn $call(self, rhs: Self) -> Self::Output {
                        $macro_impl!(self, rhs, $inner, $scalar)
                    }
                })*
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
            const PANIC_OVERFLOW: &'static str = "attempt to divide with overflow";
            simd_div
        }
    }

    impl Rem::rem {
        int_divrem_guard {
            const PANIC_ZERO: &'static str = "attempt to calculate the remainder with a divisor of zero";
            const PANIC_OVERFLOW: &'static str = "attempt to calculate the remainder with overflow";
            simd_rem
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
