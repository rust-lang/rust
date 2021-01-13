// Tests that linking to C++ code with global destructors works.

// For linking libstdc++ on MinGW
#![cfg_attr(all(windows, target_env = "gnu"), feature(static_nobundle))]

extern "C" {
    fn get() -> u32;
}

fn main() {
    let i = unsafe { get() };
    assert_eq!(i, 1234);
}
