// This previously triggered an ICE.

pub(in crate::r#mod) fn main() {}
//~^ ERROR failed to resolve: use of unresolved module or unlinked crate `r#mod`
