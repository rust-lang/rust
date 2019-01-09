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

fn foo(x: isize) { println!("{}", x); }

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64",
          target_arch = "arm",
          target_arch = "aarch64"))]
pub fn main() {
    let x: isize;
    let y: isize;
    unsafe {
        asm!("mov $1, $0" : "=r"(x) : "=r"(5)); //~ ERROR operand constraint contains '='
        asm!("mov $1, $0" : "=r"(y) : "+r"(5)); //~ ERROR operand constraint contains '+'
    }
    foo(x);
    foo(y);
}

#[cfg(not(any(target_arch = "x86",
              target_arch = "x86_64",
              target_arch = "arm",
              target_arch = "aarch64")))]
pub fn main() {}
