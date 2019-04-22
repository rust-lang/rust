#![feature(asm)]
#![allow(dead_code)]

use std::cell::Cell;

#[repr(C)]
struct NoisyDrop<'a>(&'a Cell<&'static str>);
impl<'a> Drop for NoisyDrop<'a> {
    fn drop(&mut self) {
        self.0.set("destroyed");
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    let status = Cell::new("alive");
    {
        let _y: Box<NoisyDrop>;
        let x = Box::new(NoisyDrop(&status));
        unsafe {
            asm!("mov $1, $0" : "=r"(_y) : "r"(x));
        }
        assert_eq!(status.get(), "alive");
    }
    assert_eq!(status.get(), "destroyed");
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
