//@ run-pass
#![feature(unboxed_closures)]
#![feature(fn_traits)]

struct Test;

impl FnOnce<(u32, u32)> for Test {
    type Output = u32;

    extern "rust-call" fn call_once(self, (a, b): (u32, u32)) -> u32 {
        a + b
    }
}

fn main() {
    assert_eq!(Test(1u32, 2u32), 3u32);
}
