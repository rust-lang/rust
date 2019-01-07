// compile-flags: -Z continue-parse-after-error


// ignore-tidy-tab

static FOO: u8 = b'\f';  //~ ERROR unknown byte escape

pub fn main() {
    b'\f';  //~ ERROR unknown byte escape
    b'\x0Z';  //~ ERROR invalid character in numeric character escape: Z
    b'	';  //~ ERROR byte constant must be escaped
    b''';  //~ ERROR byte constant must be escaped
    b'é';  //~ ERROR byte constant must be ASCII
    b'a  //~ ERROR unterminated byte constant
}
