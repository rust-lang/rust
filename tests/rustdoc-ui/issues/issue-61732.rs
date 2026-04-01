// This previously triggered an ICE.

pub(in crate::r#mod) fn main() {}
//~^ ERROR cannot find module or crate `r#mod` in `crate`
