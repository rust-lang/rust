struct ArpIPv4<'a> {
    s: &'a u8
}

impl<'a> ArpIPv4<'a> {
    const LENGTH: usize = 20;

    pub fn to_buffer() -> [u8; Self::LENGTH] {
        //~^ ERROR: generic `Self` types are currently not permitted in anonymous constants
        unimplemented!()
    }
}

fn main() {}
