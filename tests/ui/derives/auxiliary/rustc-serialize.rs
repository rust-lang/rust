#![crate_type = "lib"]

pub trait Decoder {
    type Error;

    fn read_enum<T, F>(&mut self, name: &str, f: F) -> Result<T, Self::Error>
        where F: FnOnce(&mut Self) -> Result<T, Self::Error>;
    fn read_enum_variant<T, F>(&mut self, names: &[&str], f: F)
                               -> Result<T, Self::Error>
        where F: FnMut(&mut Self, usize) -> Result<T, Self::Error>;

}

pub trait Decodable: Sized {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error>;
}
