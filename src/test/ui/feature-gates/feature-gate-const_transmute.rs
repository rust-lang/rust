use std::mem;

#[repr(transparent)]
struct Foo(u32);

const TRANSMUTED_U32: u32 = unsafe { mem::transmute(Foo(3)) };
//~^ ERROR The use of std::mem::transmute() is gated in constants

fn main() {}
