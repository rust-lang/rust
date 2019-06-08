// Obsolete attributes fall back to feature gated custom attributes.

#[ab_isize="stdcall"] extern {} //~ ERROR attribute `ab_isize` is currently unknown

#[fixed_stack_segment] fn f() {} //~ ERROR attribute `fixed_stack_segment` is currently unknown

fn main() {}
