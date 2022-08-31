// check-pass

// compile-flags: --emit=mir,link -O

// At one point the MIR inlining, when guarding against infinitely (or even just
// excessive) recursion, was using `ty::Instance` as the basis for its history
// check. The problem is that when you have polymorphic recursion, you can have
// distinct instances of the same code (because you're inlining the same code
// with differing substitutions), causing the amount of inlining to blow up
// exponentially.
//
// This test illustrates an example of that filed in issue rust#100476.

#![allow(unconditional_recursion)]
#![feature(decl_macro)]

macro emit($($m:ident)*) {$(
    // Randomize `def_path_hash` by defining them under a module with
    // different names
    pub mod $m {
    pub trait Tr {
        type Next: Tr;
    }

    pub fn hoge<const N: usize, T: Tr>() {
        inner::<N, T>();
    }

    #[inline(always)]
    fn inner<const N: usize, T: Tr>() {
        inner::<N, T::Next>();
    }
    }
)*}

// Increase the chance of triggering the bug
emit!(
    m00 m01 m02 m03 m04 m05 m06 m07 m08 m09
    m10 m11 m12 m13 m14 m15 m16 m17 m18 m19
);

fn main() { }
