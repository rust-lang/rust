#![allow(unused_macros)]

macro_rules! test { ($wrong:t_ty ..) => () }
                  //~^ ERROR: invalid fragment specifier `t_ty`

fn main() {}
