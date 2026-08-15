// NOTE: this is essentially a smoke-test, with more comprehensive tests living in the rustc
// repository at tests/ui/consts/const-eval/ub-enum.rs
#![feature(never_type)]

#[repr(C)]
#[allow(dead_code)]
enum E {
    V1,    // discriminant: 0
    V2(!), // 1
}

fn main() {
    unsafe {
        std::mem::transmute::<u32, E>(1);
        //~^ ERROR: encountered an uninhabited enum variant
    }
}
