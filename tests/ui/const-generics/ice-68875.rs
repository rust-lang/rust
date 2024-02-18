//@ check-fail

struct DataWrapper<'a> {
    data: &'a [u8; Self::SIZE], //~ ERROR generic `Self` types are currently not permitted in anonymous constants
}

impl DataWrapper<'_> {
    const SIZE: usize = 14;
}

pub fn main() {}
