//@aux-build:proc_macros.rs
#![feature(stmt_expr_attributes)]
#![deny(clippy::unneeded_wildcard_pattern)]
#![allow(clippy::needless_if)]

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
}
