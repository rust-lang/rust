use std::fmt;

impl fmt::Display for DecoderError { //~ ERROR cannot find type `DecoderError` in this scope
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Missing data: {}", self.0)
    }
}
fn main() {
}
