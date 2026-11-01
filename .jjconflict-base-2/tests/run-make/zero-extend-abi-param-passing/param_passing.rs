// NOTE: Exposing the bug encoded in this test is sensitive to
// LLVM optimization choices. See additional note below for an
// example.

#[link(name = "bad", kind = "static")]
extern "C" {
    pub fn c_read_value(a: u32, b: u32, c: u32) -> u16;
}

fn main() {
    const C1: usize = 0x327b23c6;
    const C2: usize = C1 & 0xFFFF;

    let r1: usize = 0x0;
    let r2: usize = C1;
    let r3: usize = 0x0;
    let value: u16 = unsafe { c_read_value(r1 as u32, r2 as u32, r3 as u32) };

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
