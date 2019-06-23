// check-pass
// ignore-android
// ignore-arm
// ignore-aarch64
// ignore-s390x
// ignore-emscripten
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64
// ignore-mips
// ignore-mips64

#![feature(asm)]

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
fn main() {
    // assignment not dead
    let mut x: isize = 0;
    unsafe {
        // extra colon
        asm!("mov $1, $0" : "=r"(x) : "r"(5_usize), "0"(x) : : "cc");
        //~^ WARNING unrecognized option
    }
    assert_eq!(x, 5);

    unsafe {
        // comma in place of a colon
        asm!("add $2, $1; mov $1, $0" : "=r"(x) : "r"(x), "r"(8_usize) : "cc", "volatile");
        //~^ WARNING expected a clobber, found an option
    }
    assert_eq!(x, 13);
}
