static FOO: &'static [u8] = b"\f";  //~ ERROR unknown byte escape

pub fn main() {
    b"\f";  //~ ERROR unknown byte escape
    b"\x0Z";  //~ ERROR invalid character in numeric character escape: `Z`
    b"é";  //~ ERROR non-ASCII character in byte constant
    br##"é"##;  //~ ERROR raw byte string must be ASCII
    b"a  //~ ERROR unterminated double quote byte string
}
