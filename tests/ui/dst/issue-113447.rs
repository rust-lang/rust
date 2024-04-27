pub struct Bytes;

impl Bytes {
    pub fn as_slice(&self) -> &[u8] {
        todo!()
    }
}

impl PartialEq<[u8]> for Bytes {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<Bytes> for &[u8] {
    fn eq(&self, other: &Bytes) -> bool {
        *other == **self
    }
}

fn main() {
    let _ = &[0u8] == [0xAA]; //~ ERROR can't compare `&[u8; 1]` with `[{integer}; 1]`
}
