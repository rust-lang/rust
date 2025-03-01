#![allow(unused)]
#![warn(clippy::byte_char_slices)]

fn main() {
    let bad = &[b'a', b'b', b'c'];
    //~^ byte_char_slices
    let quotes = &[b'"', b'H', b'i'];
    //~^ byte_char_slices
    let quotes = &[b'\'', b'S', b'u', b'p'];
    //~^ byte_char_slices
    let escapes = &[b'\x42', b'E', b's', b'c'];
    //~^ byte_char_slices

    let good = &[b'a', 0x42];
    let good = vec![b'a', b'a'];
    //~^ useless_vec
    let good: u8 = [b'a', b'c'].into_iter().sum();
}
