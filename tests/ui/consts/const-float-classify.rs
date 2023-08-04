// compile-flags: -Zmir-opt-level=0
// known-bug: #110395
// FIXME run-pass

#![feature(const_float_bits_conv)]
#![feature(const_float_classify)]
#![feature(const_trait_impl)]

// Don't promote
const fn nop<T>(x: T) -> T { x }

// FIXME(const-hack): replace with PartialEq
#[const_trait]
trait MyEq<T> {
    fn eq(self, b: T) -> bool;
}

impl const MyEq<bool> for bool {
    fn eq(self, b: bool) -> bool {
        self == b
    }
}

impl const MyEq<NonDet> for bool {
    fn eq(self, _: NonDet) -> bool {
        true
    }
}

const fn eq<A: ~const MyEq<B>, B>(x: A, y: B) -> bool {
    x.eq(y)
}

macro_rules! const_assert {
    ($a:expr, $b:expr) => {
        {
            const _: () = assert!(eq($a, $b));
            assert!(eq(nop($a), nop($b)));
        }
    };
}

macro_rules! suite {
    ( $( $tt:tt )* ) => {
        fn f32() {
            suite_inner!(f32 $($tt)*);
        }

        fn f64() {
            suite_inner!(f64 $($tt)*);
        }
    }

}

macro_rules! suite_inner {
    (
        $ty:ident [$( $fn:ident ),*]
        $val:expr => [$($out:ident),*]

        $( $tail:tt )*
    ) => {
        $( const_assert!($ty::$fn($val), $out); )*
        suite_inner!($ty [$($fn),*] $($tail)*)
    };

    ( $ty:ident [$( $fn:ident ),*]) => {};
}

#[derive(Debug)]
struct NonDet;

// The result of the `is_sign` methods are not checked for correctness, since LLVM does not
// guarantee anything about the signedness of NaNs. See
// https://github.com/rust-lang/rust/issues/55131.

suite! {
                   [is_nan, is_infinite, is_finite, is_normal, is_sign_positive, is_sign_negative]
     -0.0 / 0.0 => [  true,       false,     false,     false,           NonDet,           NonDet]
      0.0 / 0.0 => [  true,       false,     false,     false,           NonDet,           NonDet]
            1.0 => [ false,       false,      true,      true,             true,            false]
           -1.0 => [ false,       false,      true,      true,            false,             true]
            0.0 => [ false,       false,      true,     false,             true,            false]
           -0.0 => [ false,       false,      true,     false,            false,             true]
      1.0 / 0.0 => [ false,        true,     false,     false,             true,            false]
     -1.0 / 0.0 => [ false,        true,     false,     false,            false,             true]
}

fn main() {
    f32();
    f64();
}
