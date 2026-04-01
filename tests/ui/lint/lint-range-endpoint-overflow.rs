#![deny(overflowing_literals)]

fn main() {
    let range_a = 0..256; //~ ERROR range endpoint is out of range for `u8`
    let range_b = 0..=255; // ok
    let range_c = 0..=256; //~ ERROR literal out of range for `u8`
    let range_d = 256..5; //~ ERROR literal out of range for `u8`
    let range_e = 0..257; //~ ERROR literal out of range for `u8`
    let _range_f = 0..256u8;  //~ ERROR range endpoint is out of range for `u8`
    let _range_g = 0..128i8;  //~ ERROR range endpoint is out of range for `i8`

    range_a.collect::<Vec<u8>>();
    range_b.collect::<Vec<u8>>();
    range_c.collect::<Vec<u8>>();
    range_d.collect::<Vec<u8>>();
    range_e.collect::<Vec<u8>>();
}
