// ignore-s390x
// ignore-emscripten
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64

#![feature(asm)]

#[cfg(any(target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64",
            target_arch = "mips",
            target_arch = "mips64"))]
mod test_cases {
    fn is_move() {
        let y: &mut isize;
        let x = &mut 0isize;
        unsafe {
            asm!("nop" : : "r"(x));
        }
        let z = x;  //~ ERROR use of moved value: `x`
    }

    fn in_is_read() {
        let mut x = 3;
        let y = &mut x;
        unsafe {
            asm!("nop" : : "r"(x)); //~ ERROR cannot use
        }
        let z = y;
    }

    fn out_is_assign() {
        let x = 3;
        unsafe {
            asm!("nop" : "=r"(x));  //~ ERROR cannot assign twice
        }
        let mut a = &mut 3;
        let b = &*a;
        unsafe {
            asm!("nop" : "=r"(a));  // OK, Shallow write to `a`
        }
        let c = b;
        let d = *a;
    }

    fn rw_is_assign() {
        let x = 3;
        unsafe {
            asm!("nop" : "+r"(x));  //~ ERROR cannot assign twice
        }
    }

    fn indirect_is_not_init() {
        let x: i32;
        unsafe {
            asm!("nop" : "=*r"(x)); //~ ERROR use of possibly uninitialized variable
        }
    }

    fn rw_is_read() {
        let mut x = &mut 3;
        let y = &*x;
        unsafe {
            asm!("nop" : "+r"(x));  //~ ERROR cannot assign to `x` because it is borrowed
        }
        let z = y;
    }

    fn two_moves() {
        let x = &mut 2;
        unsafe {
            asm!("nop" : : "r"(x), "r"(x) );    //~ ERROR use of moved value
        }
    }
}

fn main() {}
