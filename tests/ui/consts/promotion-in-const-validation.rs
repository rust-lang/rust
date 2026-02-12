fn foo() {}

const _: &usize = unsafe { &std::mem::transmute(foo as fn()) };
//~^ ERROR: encountered a pointer, but expected an integer

const _: usize = unsafe { std::mem::transmute(foo as fn()) };
//~^ ERROR: encountered a pointer, but expected an integer

fn main() {}
