// run-pass

use std::fmt;

union U {
    a: u8
}

impl fmt::Display for U {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe { write!(f, "Oh hai {}", self.a) }
    }
}

fn main() {
    assert_eq!(U { a: 2 }.to_string(), "Oh hai 2");
}
