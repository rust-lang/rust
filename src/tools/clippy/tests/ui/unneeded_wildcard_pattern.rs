//@run-rustfix
#![feature(stmt_expr_attributes)]
#![deny(clippy::unneeded_wildcard_pattern)]

fn main() {
    let t = (0, 1, 2, 3);

    if let (0, .., _) = t {};
    if let (0, _, ..) = t {};
    if let (_, .., 0) = t {};
    if let (.., _, 0) = t {};
    if let (0, _, _, ..) = t {};
    if let (0, .., _, _) = t {};
    if let (_, 0, ..) = t {};
    if let (.., 0, _) = t {};
    if let (0, _, _, _) = t {};
    if let (0, ..) = t {};
    if let (.., 0) = t {};

    #[rustfmt::skip]
    {
        if let (0, .., _, _,) = t {};
    }

    struct S(usize, usize, usize, usize);

    let s = S(0, 1, 2, 3);

    if let S(0, .., _) = s {};
    if let S(0, _, ..) = s {};
    if let S(_, .., 0) = s {};
    if let S(.., _, 0) = s {};
    if let S(0, _, _, ..) = s {};
    if let S(0, .., _, _) = s {};
    if let S(_, 0, ..) = s {};
    if let S(.., 0, _) = s {};
    if let S(0, _, _, _) = s {};
    if let S(0, ..) = s {};
    if let S(.., 0) = s {};

    #[rustfmt::skip]
    {
        if let S(0, .., _, _,) = s {};
    }
}
