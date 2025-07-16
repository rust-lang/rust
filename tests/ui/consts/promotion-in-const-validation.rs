fn foo() {}

const _: &usize = unsafe { &std::mem::transmute(foo as fn()) };
//~^ ERROR: constructing invalid value at .<deref>: encountered a pointer, but expected an integer

const _: usize = unsafe { std::mem::transmute(foo as fn()) };
//~^ ERROR: unable to turn pointer into integer

fn main() {}
