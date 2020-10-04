// run-pass
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

trait FromBytesLittleEndian {
    const N: usize;
    fn from_bytes_le(other: &[u8; Self::N]) -> Self;
}

impl FromBytesLittleEndian for u32 {
    const N: usize = 4;
    fn from_bytes_le(other: &[u8; 4]) -> Self {
        Self::from_le_bytes(*other)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Header(u32);

impl FromBytesLittleEndian for Header {
    const N: usize = 4;
    fn from_bytes_le(r: &[u8; 4]) -> Self {
        Self(FromBytesLittleEndian::from_bytes_le(r))
        // This previously caused an ICE in the above line.
    }
}

fn main() {
    let data = [1, 2, 3, 4];
    let h = Header::from_bytes_le(&data);
    assert_eq!(h, Header(0x04030201));
}
