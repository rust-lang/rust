//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat

#![feature(core_intrinsics)]

fn main() {
    // args_ty is not a tuple
    core::intrinsics::offload::<_, _, ()>(kernel_0, [1, 1, 1], [1, 1, 1], 0, 42);
    //~^ ERROR `{integer}` is not a tuple
}

fn kernel_0() {}
