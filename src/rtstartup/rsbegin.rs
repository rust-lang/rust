// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rsbegin.o and rsend.o are the so called "compiler runtime startup objects".
// They contain code needed to correctly initialize the compiler runtime.
//
// When an executable or dylib image is linked, all user code and libraries are
// "sandwiched" between these two object files, so code or data from rsbegin.o
// become first in the respective sections of the image, whereas code and data
// from rsend.o become the last ones.  This effect can be used to place symbols
// at the beginning or at the end of a section, as well as to insert any required
// headers or footers.
//
// Note that the actual module entry point is located in the C runtime startup
// object (usually called `crtX.o), which then invokes initialization callbacks
// of other runtime components (registered via yet another special image section).

#![feature(no_std)]

#![crate_type="rlib"]
#![no_std]
#![allow(non_camel_case_types)]

#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
pub mod eh_frames
{
    #[no_mangle]
    #[link_section = ".eh_frame"]
    // Marks beginning of the stack frame unwind info section
    pub static __EH_FRAME_BEGIN__: [u8; 0] = [];

    // Scratch space for unwinder's internal book-keeping.
    // This is defined as `struct object` in $GCC/libgcc/unwind-dw2-fde.h.
    static mut obj: [isize; 6] = [0; 6];

    // Unwind info registration/deregistration routines.
    // See the docs of `unwind` module in libstd.
    extern {
        fn rust_eh_register_frames(eh_frame_begin: *const u8, object: *mut u8);
        fn rust_eh_unregister_frames(eh_frame_begin: *const u8, object: *mut u8);
    }

    unsafe fn init() {
        // register unwind info on module startup
        rust_eh_register_frames(&__EH_FRAME_BEGIN__ as *const u8,
                                &mut obj as *mut _ as *mut u8);
    }

    unsafe fn uninit() {
        // unregister on shutdown
        rust_eh_unregister_frames(&__EH_FRAME_BEGIN__ as *const u8,
                                  &mut obj as *mut _ as *mut u8);
    }

    // MSVC-specific init/uninit routine registration
    pub mod ms_init
    {
        // .CRT$X?? sections are roughly analogous to ELF's .init_array and .fini_array,
        // except that they exploit the fact that linker will sort them alphabitically,
        // so e.g. sections with names between .CRT$XIA and .CRT$XIZ are guaranteed to be
        // placed between those two, without requiring any ordering of objects on the linker
        // command line.
        // Note that ordering of same-named sections from different objects is not guaranteed.
        // Since .CRT$XIA contains init array's header symbol, which must always come first,
        // we place our initialization callback into .CRT$XIB.

        #[link_section = ".CRT$XIB"] // .CRT$XI? : C initialization callbacks
        pub static P_INIT: unsafe fn() = super::init;

        #[link_section = ".CRT$XTY"] // .CRT$XT? : C termination callbacks
        pub static P_UNINIT: unsafe fn() = super::uninit;
    }
}
