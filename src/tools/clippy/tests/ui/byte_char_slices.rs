#![allow(unused)]
#![warn(clippy::byte_char_slices)]

fn main() {
    let bad = &[b'a', b'b', b'c'];
    let quotes = &[b'"', b'H', b'i'];
    let quotes = &[b'\'', b'S', b'u', b'p'];
    let escapes = &[b'\x42', b'E', b's', b'c'];

    let good = &[b'a', 0x42];
    let good = vec![b'a', b'a'];
    let good: u8 = [b'a', b'c'].into_iter().sum();
}
