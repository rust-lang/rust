//@aux-build:proc_macros.rs
#![feature(stmt_expr_attributes)]
#![deny(clippy::unneeded_wildcard_pattern)]
#![allow(clippy::needless_ifs)]

#[macro_use]
extern crate proc_macros;

fn main() {
    let t = (0, 1, 2, 3);

    if let (0, .., _) = t {};
    //~^ unneeded_wildcard_pattern
    if let (0, _, ..) = t {};
    //~^ unneeded_wildcard_pattern
    if let (_, .., 0) = t {};
    //~^ unneeded_wildcard_pattern
    if let (.., _, 0) = t {};
    //~^ unneeded_wildcard_pattern
    if let (0, _, _, ..) = t {};
    //~^ unneeded_wildcard_pattern
    if let (0, .., _, _) = t {};
    //~^ unneeded_wildcard_pattern
    if let (_, 0, ..) = t {};
    if let (.., 0, _) = t {};
    if let (0, _, _, _) = t {};
    if let (0, ..) = t {};
    if let (.., 0) = t {};

    #[rustfmt::skip]
    {
        if let (0, .., _, _,) = t {};
        //~^ unneeded_wildcard_pattern
    }

    struct S(usize, usize, usize, usize);

    let s = S(0, 1, 2, 3);

    if let S(0, .., _) = s {};
    //~^ unneeded_wildcard_pattern
    if let S(0, _, ..) = s {};
    //~^ unneeded_wildcard_pattern
    if let S(_, .., 0) = s {};
    //~^ unneeded_wildcard_pattern
    if let S(.., _, 0) = s {};
    //~^ unneeded_wildcard_pattern
    if let S(0, _, _, ..) = s {};
    //~^ unneeded_wildcard_pattern
    if let S(0, .., _, _) = s {};
    //~^ unneeded_wildcard_pattern
    if let S(_, 0, ..) = s {};
    if let S(.., 0, _) = s {};
    if let S(0, _, _, _) = s {};
    if let S(0, ..) = s {};
    if let S(.., 0) = s {};

    #[rustfmt::skip]
    {
        if let S(0, .., _, _,) = s {};
        //~^ unneeded_wildcard_pattern
    }
    external! {
        let t = (0, 1, 2, 3);
        if let (0, _, ..) = t {};
    }

    struct Struct4 {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    }

    let fourval = Struct4 {
        a: 5,
        b: 10,
        c: 15,
        d: 20,
    };

    // unlike the tuple forms, the struct form can only have the `..` at the end of the list
    let Struct4 { mut a, mut b, c: _, .. } = fourval;
    //~^ unneeded_wildcard_pattern
    let Struct4 { mut b, c: _, d: _, .. } = fourval;
    //~^ unneeded_wildcard_pattern
    let Struct4 { mut a, b, c, d: _ } = fourval;
    let Struct4 { mut c, d, .. } = fourval;
    let Struct4 { b: _, c: _, .. } = fourval;
    //~^ unneeded_wildcard_pattern
    let Struct4 { c: _, .. } = fourval;
    //~^ unneeded_wildcard_pattern
}
