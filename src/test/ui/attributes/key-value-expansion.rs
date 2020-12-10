// Regression tests for issue #55414, expansion happens in the value of a key-value attribute,
// and the expanded expression is more complex than simply a macro call.

// aux-build:key-value-expansion.rs

#![feature(rustc_attrs)]

extern crate key_value_expansion;

// Minimized test case.

macro_rules! bug {
    ($expr:expr) => {
        #[rustc_dummy = $expr] // Any key-value attribute, not necessarily `doc`
        struct S;
    };
}

// Any expressions containing macro call `X` that's more complex than `X` itself.
// Parentheses will work.
bug!((column!())); //~ ERROR unexpected token: `(7u32)`

// Original test case.

macro_rules! bug {
    () => {
        bug!("bug" + stringify!(found)); //~ ERROR unexpected token: `"bug" + "found"`
    };
    ($test:expr) => {
        #[doc = $test]
        struct Test {}
    };
}

bug!();

// Test case from #66804.

macro_rules! doc_comment {
    ($x:expr) => {
        #[doc = $x]
        extern {}
    };
}

macro_rules! some_macro {
    ($t1: ty) => {
        doc_comment! {format!("{coor}", coor = stringify!($t1)).as_str()}
        //~^ ERROR unexpected token: `{
    };
}

some_macro!(u8);

fn main() {}
