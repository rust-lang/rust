// run-pass
#![allow(unused_mut)]
// The logic for parsing Kleene operators in macros has a special case to disambiguate `?`.
// Specifically, `$(pat)?` is the ZeroOrOne operator whereas `$(pat)?+` or `$(pat)?*` are the
// ZeroOrMore and OneOrMore operators using `?` as a separator. These tests are intended to
// exercise that logic in the macro parser.
//
// Moreover, we also throw in some tests for using a separator with `?`, which is meaningless but
// included for consistency with `+` and `*`.
//
// This test focuses on non-error cases and making sure the correct number of repetitions happen.

// edition:2018

macro_rules! foo {
    ($($a:ident)? ; $num:expr) => { {
        let mut x = 0;

        $(
            x += $a;
         )?

        assert_eq!(x, $num);
    } }
}

pub fn main() {
    let a = 1;

    // accept 0 or 1 repetitions
    foo!( ; 0);
    foo!(a ; 1);
}
