//@ compile-flags: --crate-type=rlib

#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }

#[lang="tuple_trait"]
trait Tuple { }

// Functions
extern "gpu-kernel" fn f1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
//~^ ERROR is not a supported ABI

// Methods in trait definition
trait Tr {
    extern "gpu-kernel" fn m1(_: ()); //~ ERROR "gpu-kernel" ABI is experimental and subject to change

    extern "gpu-kernel" fn dm1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
    //~^ ERROR is not a supported ABI
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "gpu-kernel" fn m1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
    //~^ ERROR is not a supported ABI
}

// Methods in inherent impl
impl S {
    extern "gpu-kernel" fn im1(_: ()) {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
    //~^ ERROR is not a supported ABI
}

// Function pointer types
type A1 = extern "gpu-kernel" fn(_: ()); //~ ERROR "gpu-kernel" ABI is experimental and subject to change
//~^ WARN the calling convention "gpu-kernel" is not supported on this target
//~^^ WARN this was previously accepted by the compiler but is being phased out

// Foreign modules
extern "gpu-kernel" {} //~ ERROR "gpu-kernel" ABI is experimental and subject to change
//~^ ERROR is not a supported ABI
