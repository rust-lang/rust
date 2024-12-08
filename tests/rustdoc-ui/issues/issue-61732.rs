// This previously triggered an ICE.

pub(in crate::r#mod) fn main() {}
//~^ ERROR failed to resolve: you might be missing crate `r#mod`
