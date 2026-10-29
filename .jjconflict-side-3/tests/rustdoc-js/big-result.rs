#![allow(nonstandard_style)]
/// Generate 250 items that all match the query, starting with the longest.
/// Those long items should be dropped from the result set, and the short ones
/// should be shown instead.
macro_rules! generate {
    ([$($x:ident),+], $y:tt, $z:tt) => {
        $(
            generate!(@ $x, $y, $z);
        )+
    };
    (@ $x:ident , [$($y:ident),+], $z:tt) => {
        pub struct $x;
        $(
            generate!(@@ $x, $y, $z);
        )+
    };
    (@@ $x:ident , $y:ident, [$($z:ident: $zt:ident),+]) => {
        impl $y {
            pub fn $x($($z: $zt,)+) {}
        }
    }
}

pub struct First;
pub struct Second;
pub struct Third;
pub struct Fourth;
pub struct Fifth;

generate!(
    [a, b, c, d, e],
    [a, b, c, d, e, f, g, h, i, j],
    [a: First, b: Second, c: Third, d: Fourth, e: Fifth]
);

generate!(
    [f, g, h, i, j],
    [a, b, c, d, e, f, g, h, i, j],
    [a: First, b: Second, c: Third, d: Fourth]
);

generate!(
    [k, l, m, n, o],
    [a, b, c, d, e, f, g, h, i, j],
    [a: First, b: Second, c: Third]
);

generate!(
    // reverse it, just to make sure they're alphabetized
    // in the result set when all else is equal
    [t, s, r, q, p],
    [a, b, c, d, e, f, g, h, i, j],
    [a: First, b: Second]
);

generate!(
    [u, v, w, x, y],
    [a, b, c, d, e, f, g, h, i, j],
    [a: First]
);
