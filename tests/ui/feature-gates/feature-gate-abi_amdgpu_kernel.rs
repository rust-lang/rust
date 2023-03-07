// compile-flags: --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

extern "amdgpu-kernel" fn fu() {} //~ ERROR amdgpu-kernel ABI is experimental
//~^ ERROR is not a supported ABI

trait T {
    extern "amdgpu-kernel" fn mu(); //~ ERROR amdgpu-kernel ABI is experimental
    extern "amdgpu-kernel" fn dmu() {} //~ ERROR amdgpu-kernel ABI is experimental
    //~^ ERROR is not a supported ABI
}

struct S;
impl T for S {
    extern "amdgpu-kernel" fn mu() {} //~ ERROR amdgpu-kernel ABI is experimental
    //~^ ERROR is not a supported ABI
}

impl S {
    extern "amdgpu-kernel" fn imu() {} //~ ERROR amdgpu-kernel ABI is experimental
    //~^ ERROR is not a supported ABI
}

type TAU = extern "amdgpu-kernel" fn(); //~ ERROR amdgpu-kernel ABI is experimental

extern "amdgpu-kernel" {} //~ ERROR amdgpu-kernel ABI is experimental
//~^ ERROR is not a supported ABI
