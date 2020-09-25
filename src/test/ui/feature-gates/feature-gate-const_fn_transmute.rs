use std::mem;

#[repr(transparent)]
struct Foo(u32);

const TRANSMUTED_U32: u32 = unsafe { mem::transmute(Foo(3)) };

const fn transmute_fn() -> u32 { unsafe { mem::transmute(Foo(3)) } }
//~^ ERROR `transmute`

const fn transmute_fn_intrinsic() -> u32 { unsafe { std::intrinsics::transmute(Foo(3)) } }
//~^ ERROR `transmute`

const fn transmute_fn_core_intrinsic() -> u32 { unsafe { core::intrinsics::transmute(Foo(3)) } }
//~^ ERROR `transmute`

const unsafe fn unsafe_transmute_fn() -> u32 { mem::transmute(Foo(3)) }
//~^ ERROR `transmute`

const unsafe fn unsafe_transmute_fn_intrinsic() -> u32 { std::intrinsics::transmute(Foo(3)) }
//~^ ERROR `transmute`

const unsafe fn unsafe_transmute_fn_core_intrinsic() -> u32 { core::intrinsics::transmute(Foo(3)) }
//~^ ERROR `transmute`

const fn safe_transmute_fn() -> u32 { mem::transmute(Foo(3)) }
//~^ ERROR `transmute`
//~| ERROR call to unsafe function is unsafe and requires unsafe function or block

const fn safe_transmute_fn_intrinsic() -> u32 { std::intrinsics::transmute(Foo(3)) }
//~^ ERROR `transmute`
//~| ERROR call to unsafe function is unsafe and requires unsafe function or block

const fn safe_transmute_fn_core_intrinsic() -> u32 { core::intrinsics::transmute(Foo(3)) }
//~^ ERROR `transmute`
//~| ERROR call to unsafe function is unsafe and requires unsafe function or block

fn main() {}
