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

#![feature(asm, rustc_attrs)]

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
#[rustc_error]
pub fn main() {
    unsafe {
        // clobber formatted as register input/output
        asm!("xor %eax, %eax" : : : "{eax}");
        //~^ ERROR clobber should not be surrounded by braces
    }
}
