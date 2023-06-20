//@compile-flags: -Zmiri-disable-isolation
//@ignore-target-windows: No libc on Windows

#![feature(rustc_private)]
#![feature(strict_provenance)]

use std::ptr;

fn main() {
    // Linux specifies that it is not an error if the specified range does not contain any pages.
    // But we simply do not support such calls. This test checks that we report this as
    // unsupported, not Undefined Behavior.
    let res = unsafe {
        libc::munmap(
            //~^ ERROR: unsupported operation
            // Some high address we surely have not allocated anything at
            ptr::invalid_mut(1 << 30),
            4096,
        )
    };
    assert_eq!(res, 0);
}
