//@no-rustfix
// Format width has no effect for certain traits (issue #15039)

#![warn(clippy::unused_format_specs)]
#![allow(clippy::zero_ptr, clippy::manual_dangling_ptr)]

fn main() {
    // Integer formats with # (alternate): 0x/0o/0b prefix makes min width 4
    println!("{:#02X}", 1u8); //~ unused_format_specs
    println!("{:#2X}", 1u8); //~ unused_format_specs
    println!("{:#02x}", 1u8); //~ unused_format_specs
    println!("{:#02o}", 1u8); //~ unused_format_specs
    println!("{:#02b}", 1u8); //~ unused_format_specs

    // Exponent formats: min width 4 (e.g. 1e0)
    println!("{:02e}", 1u8); //~ unused_format_specs
    println!("{:02E}", 1u8); //~ unused_format_specs
    println!("{:2e}", 1.0); //~ unused_format_specs
    println!("{:2E}", 1.0); //~ unused_format_specs
    println!("{:2e}", 0.1); //~ unused_format_specs
    println!("{:2E}", 0.1); //~ unused_format_specs

    // Pointer: min width 4 (0x1)
    println!("{:2p}", 0 as *const usize); //~ unused_format_specs
    println!("{:02p}", 1 as *const usize); //~ unused_format_specs

    // Width 2 still too small for exponent; precision+width
    println!("{:2.2e}", 1.0); //~ unused_format_specs
    println!("{:2.2E}", 1.0); //~ unused_format_specs
    println!("{:2.2e}", 0.1); //~ unused_format_specs
    println!("{:2.2E}", 0.1); //~ unused_format_specs

    // Width 3 is exactly the minimum for alternate hex, still warn
    println!("{:#03X}", 1u8); //~ unused_format_specs

    // Not linted: width more than 3, or no # for x/o/b
    println!("{:#04X}", 1u8);
    println!("{:2X}", 1u8); // no #, so no prefix
    println!("{:2o}", 1u8);
    println!("{}", 1);
}
