// Obsolete attributes fall back to unstable custom attributes.

#[ab_isize = "stdcall"] extern "C" {}
//~^ ERROR cannot find attribute `ab_isize` in this scope

#[fixed_stack_segment] fn f() {}
//~^ ERROR cannot find attribute `fixed_stack_segment` in this scope

fn main() {}
