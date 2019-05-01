#![deny(overflowing_literals)]

fn main() {
    const XYZ: char = 0x1F888 as char;
    //~^ ERROR only `u8` can be cast into `char`
    const XY: char = 129160 as char;
    //~^ ERROR only `u8` can be cast into `char`
    const ZYX: char = '\u{01F888}';
    println!("{}", XYZ);
}
