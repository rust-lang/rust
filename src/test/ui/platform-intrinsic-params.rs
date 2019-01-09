#![feature(platform_intrinsics)]
extern "platform-intrinsic" {
    fn x86_mm_movemask_ps() -> i32; //~ERROR found 0, expected 1
}

fn main() { }
