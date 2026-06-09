#![warn(clippy::byte_char_slices)]
#![allow(clippy::useless_vec)]

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
}

fn takes_array_ref(_: &[u8; 2]) {}

fn takes_array_ref_ref(_: &&[u8; 2]) {}

fn issue16759(bytes: [u32; 3]) {
    const START: u32 = u32::from_le_bytes([b'W', b'O', b'R', b'K']);
    //~^ byte_char_slices

    let auto_deref_to_slice: u8 = [b'a', b'c'].iter().copied().sum();
    //~^ byte_char_slices

    let with_comment = [
        // 1     2     3
        b'a', b'b', b'c', //   x
        b'd', b'e', b'f', //  2x
        b'g', b'h', b'i', //  3x
    ];
    let with_cfg = [
        b'a',
        b'b',
        b'c',
        #[cfg(feature = "foo")]
        b'd',
    ];

    let with_escape: u8 = [b'\'', b'"', b'\x00', b'\n', b'\\'].iter().copied().sum();
    //~^ byte_char_slices

    takes_array_ref(&[b'a', b'b']);
    //~^ byte_char_slices

    takes_array_ref_ref(&&[b'a', b'b']);
    //~^ byte_char_slices
}
