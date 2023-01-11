// run-pass
#![no_std]

extern crate std;

std::thread_local!(static A: usize = 30);

fn main() {
}
