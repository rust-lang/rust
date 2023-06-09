// This previously triggered an ICE.

pub(in crate::r#mod) fn main() {}
//~^ ERROR failed to resolve: maybe a missing crate `r#mod`
