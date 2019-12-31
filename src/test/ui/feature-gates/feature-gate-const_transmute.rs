use std::mem;

#[repr(transparent)]
struct Foo(u32);

const TRANSMUTED_U32: u32 = unsafe { mem::transmute(Foo(3)) };
//~^ ERROR `std::intrinsics::transmute` is not yet stable as a const fn

fn main() {}
