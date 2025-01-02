//@ compile-flags: --crate-type=rlib

#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }

#[lang="tuple_trait"]
trait Tuple { }

// Functions
extern "amdgpu-kernel" fn f1(_: ()) {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
//~^ ERROR is not a supported ABI

// Methods in trait definition
trait Tr {
    extern "amdgpu-kernel" fn m1(_: ()); //~ ERROR amdgpu-kernel ABI is experimental and subject to change

    extern "amdgpu-kernel" fn dm1(_: ()) {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
    //~^ ERROR is not a supported ABI
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "amdgpu-kernel" fn m1(_: ()) {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
    //~^ ERROR is not a supported ABI
}

// Methods in inherent impl
impl S {
    extern "amdgpu-kernel" fn im1(_: ()) {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
    //~^ ERROR is not a supported ABI
}

// Function pointer types
type A1 = extern "amdgpu-kernel" fn(_: ()); //~ ERROR amdgpu-kernel ABI is experimental and subject to change
//~^ WARN the calling convention "amdgpu-kernel" is not supported on this target
//~^^ WARN this was previously accepted by the compiler but is being phased out

// Foreign modules
extern "amdgpu-kernel" {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
//~^ ERROR is not a supported ABI
