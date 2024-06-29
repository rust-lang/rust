static FOO: &'static [u8] = b"\c";  //~ ERROR unknown byte escape

pub fn main() {
    b"\c";  //~ ERROR unknown byte escape
    b"\x0Z";  //~ ERROR invalid character in numeric character escape: `Z`
    b"é";  //~ ERROR non-ASCII character in byte string literal
    br##"é"##;  //~ ERROR non-ASCII character in raw byte string literal
    b"a  //~ ERROR unterminated double quote byte string
}
