pub struct Thing {
    octets: [u8; 8],
}

impl Thing {
    pub contst fn octets(&self) -> [u8; 8] {
    //~^ ERROR visibility `pub` is not followed by an item
        self.octets
    }
}
