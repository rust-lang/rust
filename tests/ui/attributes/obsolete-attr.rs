// Obsolete attributes fall back to unstable custom attributes.

#[ab_isize = "stdcall"] extern "C" {}
//~^ ERROR cannot find attribute `ab_isize`

#[fixed_stack_segment] fn f() {}
//~^ ERROR cannot find attribute `fixed_stack_segment`

fn main() {}
