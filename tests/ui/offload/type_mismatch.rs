//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat

#![feature(core_intrinsics)]

fn main() {
    // kernel_ty is not a function item
    let not_fn = 42;
    core::intrinsics::offload::<_, _, ()>(not_fn, [1, 1, 1], [1, 1, 1], 0, ());
    //~^ ERROR expected a function item for the offload kernel, found `i32`

    // argument count mismatch
    core::intrinsics::offload::<_, _, ()>(kernel_1, [1, 1, 1], [1, 1, 1], 0, ());
    //~^ ERROR offload kernel expects 1 arguments, but 0 arguments were provided

    // argument type mismatch
    core::intrinsics::offload::<_, _, ()>(kernel_1, [1, 1, 1], [1, 1, 1], 0, (42.0f64,));
    //~^ ERROR type mismatch in offload kernel argument 0: expected `f32`, found `f64`

    // return type mismatch
    let _: f64 = core::intrinsics::offload::<_, _, f64>(kernel_0, [1, 1, 1], [1, 1, 1], 0, ());
    //~^ ERROR offload kernel return type mismatch: kernel returns `()`, but offload call expects `f64`

    // multiple argument type mismatch
    core::intrinsics::offload::<_, _, ()>(kernel_2, [1, 1, 1], [1, 1, 1], 0, (42.0f64, 42.0f64));
    //~^ ERROR type mismatch in offload kernel argument 0: expected `f32`, found `f64`
    //~| ERROR type mismatch in offload kernel argument 1: expected `f32`, found `f64`
}

fn kernel_0() {}
fn kernel_1(_x: f32) {}
fn kernel_2(_x: f32, _y: f32) {}
