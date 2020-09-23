// check-pass
// compile-flags: -Z parse-only

#![feature(inline_const)]
const MMIO_BIT1: u8 = 4;
const MMIO_BIT2: u8 = 5;

fn main() {
    match read_mmio() {
        0 => {}
        const { 1 << MMIO_BIT1 } => println!("FOO"),
        const { 1 << MMIO_BIT2 } => println!("BAR"),

        _ => unreachable!(),
    }
}

fn read_mmio() -> u8 {
    1 << 5
}
