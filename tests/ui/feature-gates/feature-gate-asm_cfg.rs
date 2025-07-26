//@ only-x86_64
#![crate_type = "lib"]

use std::arch::{asm, global_asm, naked_asm};

global_asm!(
    "nop",
    #[cfg(false)]
    //~^ ERROR the `#[cfg(/* ... */)]` and `#[cfg_attr(/* ... */)]` attributes on assembly are unstable
    "nop"
);

#[unsafe(naked)]
#[no_mangle]
extern "C" fn naked() {
    naked_asm!(
        "mov rax, 5",
        #[cfg(false)]
        //~^ ERROR the `#[cfg(/* ... */)]` and `#[cfg_attr(/* ... */)]` attributes on assembly are unstable
        "mov rax, {a}",
        "ret",
        #[cfg(false)]
        //~^ ERROR the `#[cfg(/* ... */)]` and `#[cfg_attr(/* ... */)]` attributes on assembly are unstable
        a = const 10,
    )
}

fn asm() {
    unsafe {
        asm!(
            "nop",
            #[cfg(false)]
            //~^ ERROR the `#[cfg(/* ... */)]` and `#[cfg_attr(/* ... */)]` attributes on assembly are unstable
            clobber_abi("C"),
            clobber_abi("C"), //~ ERROR `C` ABI specified multiple times
        );
    }
}

fn bad_attribute() {
    unsafe {
        asm!(
            #[inline]
            //~^ ERROR this attribute is not supported on assembly
            "nop"
        )
    };
}
