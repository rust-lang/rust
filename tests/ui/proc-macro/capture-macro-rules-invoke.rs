//@ proc-macro: test-macros.rs
//@ check-pass
//@ compile-flags: -Z span-debug

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate test_macros;
use test_macros::{print_bang, print_bang_consume};

macro_rules! test_matchers {
    ($expr:expr, $block:block, $stmt:stmt, $ty:ty, $ident:ident, $lifetime:lifetime,
     $meta:meta, $path:path, $vis:vis, $tt:tt, $lit:literal) => {
        print_bang_consume!($expr, $block, $stmt, $ty, $ident,
                            $lifetime, $meta, $path, $vis, $tt, $lit)
    }
}

macro_rules! use_expr {
    ($expr:expr) => {
        print_bang!($expr)
    }
}

macro_rules! use_pat {
    ($pat:pat) => {
        print_bang!($pat)
    }
}

#[allow(dead_code)]
struct Foo;
impl Foo {
    #[allow(dead_code)]
    fn use_self(self) {
        drop(use_expr!(self));
        test_matchers!(
            1 + 1,
            { "a" },
            let a = 1,
            String,
            my_name,
            'a,
            my_val = 30,
            std::option::Option,
            pub(in some::path),
            [ a b c ],
            -30
        );
    }

    fn with_pat(use_pat!((a, b)): (u32, u32)) {
        let _ = (a, b);
    }
}

fn main() {}
