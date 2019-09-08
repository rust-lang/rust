// Obsolete attributes fall back to unstable custom attributes.

#[ab_isize="stdcall"] extern {}
//~^ ERROR cannot find attribute macro `ab_isize` in this scope

#[fixed_stack_segment] fn f() {}
//~^ ERROR cannot find attribute macro `fixed_stack_segment` in this scope

fn main() {}
