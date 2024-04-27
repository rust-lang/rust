//@ run-pass
#![allow(dead_code)]
#![allow(improper_ctypes)]

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn issue_97463_leak_uninit_data(a: u32, b: u32, c: u32) -> u16;
}

fn main() {
    const C1: usize = 0x327b23c6;
    const C2: usize = C1 & 0xFFFF;

    let r1: usize = 0x0;
    let r2: usize = C1;
    let r3: usize = 0x0;
    let value: u16 = unsafe { issue_97463_leak_uninit_data(r1 as u32, r2 as u32, r3 as u32) };

    // NOTE: as an example of the sensitivity of this test to optimization choices,
    // uncommenting this block of code makes the bug go away on pnkfelix's machine.
    // (But observing via `dbg!` doesn't hide the bug. At least sometimes.)
    /*
    println!("{}", value);
    println!("{}", value as usize);
    println!("{}", usize::from(value));
    println!("{}", (value as usize) & 0xFFFF);
     */

    let d1 = value;
    let d2 = value as usize;
    let d3 = usize::from(value);
    let d4 = (value as usize) & 0xFFFF;

    let d = (&d1, &d2, &d3, &d4);
    let d_ = (d1, d2, d3, d4);

    assert_eq!(((&(C2 as u16), &C2, &C2, &C2), (C2 as u16, C2, C2, C2)), (d, d_));
}
