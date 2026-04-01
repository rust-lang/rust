#![allow(unused_macros)]

// Issue #21370

macro_rules! test {
    ($wrong:t_ty) => () //~ ERROR invalid fragment specifier `t_ty`
}

fn main() { }
