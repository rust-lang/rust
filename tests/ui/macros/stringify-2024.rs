//@ run-pass
//@ edition:2024
//@ compile-flags: --test

#![deny(unused_macros)]

macro_rules! expr { ($expr:expr) => { stringify!($expr) }; }

macro_rules! c1 {
    ($frag:ident, [$($tt:tt)*], $s:literal) => {
        assert_eq!($frag!($($tt)*), $s);
        assert_eq!(stringify!($($tt)*), $s);
    };
}

#[test]
fn test_expr() {
    // ExprKind::Let (chains)
    c1!(expr, [ if let _ = true && false {} ], "if let _ = true && false {}");
    c1!(expr, [ if let _ = (true && false) {} ], "if let _ = (true && false) {}");
}
