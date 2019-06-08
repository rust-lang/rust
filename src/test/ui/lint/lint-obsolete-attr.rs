// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.

#[ab_isize="stdcall"] extern {} //~ ERROR attribute `ab_isize` is currently unknown

#[fixed_stack_segment] fn f() {} //~ ERROR attribute `fixed_stack_segment` is currently unknown

fn main() {}
