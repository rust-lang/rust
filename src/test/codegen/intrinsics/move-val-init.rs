// compile-flags: -C no-prepopulate-passes

#![feature(core_intrinsics)]
#![crate_type = "lib"]

// test that `move_val_init` actually avoids big allocas

use std::intrinsics::move_val_init;

pub struct Big {
    pub data: [u8; 65536]
}

// CHECK-LABEL: @test_mvi
#[no_mangle]
pub unsafe fn test_mvi(target: *mut Big, make_big: fn() -> Big) {
    // CHECK: call void %make_big(%Big*{{[^%]*}} %target)
    move_val_init(target, make_big());
}
