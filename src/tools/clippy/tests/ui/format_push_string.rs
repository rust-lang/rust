#![warn(clippy::format_push_string)]

fn main() {
    let mut string = String::new();
    string += &format!("{:?}", 1234);
    //~^ ERROR: `format!(..)` appended to existing `String`
    string.push_str(&format!("{:?}", 5678));
    //~^ ERROR: `format!(..)` appended to existing `String`
}

mod issue9493 {
    pub fn u8vec_to_hex(vector: &Vec<u8>, upper: bool) -> String {
        let mut hex = String::with_capacity(vector.len() * 2);
        for byte in vector {
            hex += &(if upper {
                //~^ ERROR: `format!(..)` appended to existing `String`
                format!("{byte:02X}")
            } else {
                format!("{byte:02x}")
            });
        }
        hex
    }

    pub fn other_cases() {
        let mut s = String::new();
        // if let
        s += &(if let Some(_a) = Some(1234) {
            //~^ ERROR: `format!(..)` appended to existing `String`
            format!("{}", 1234)
        } else {
            format!("{}", 1234)
        });
        // match
        s += &(match Some(1234) {
            //~^ ERROR: `format!(..)` appended to existing `String`
            Some(_) => format!("{}", 1234),
            None => format!("{}", 1234),
        });
    }
}
