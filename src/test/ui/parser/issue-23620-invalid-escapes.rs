// compile-flags: -Z continue-parse-after-error

fn main() {
    let _ = b"\u{a66e}";
    //~^ ERROR unicode escape sequences cannot be used as a byte or in a byte string

    let _ = b'\u{a66e}';
    //~^ ERROR unicode escape sequences cannot be used as a byte or in a byte string

    let _ = b'\u';
    //~^ ERROR incorrect unicode escape sequence
    //~^^ ERROR unicode escape sequences cannot be used as a byte or in a byte string

    let _ = b'\x5';
    //~^ ERROR numeric character escape is too short

    let _ = b'\xxy';
    //~^ ERROR invalid character in numeric character escape: x
    //~^^ ERROR invalid character in numeric character escape: y

    let _ = '\x5';
    //~^ ERROR numeric character escape is too short

    let _ = '\xxy';
    //~^ ERROR invalid character in numeric character escape: x
    //~^^ ERROR invalid character in numeric character escape: y

    let _ = b"\u{a4a4} \xf \u";
    //~^ ERROR unicode escape sequences cannot be used as a byte or in a byte string
    //~^^ ERROR invalid character in numeric character escape:
    //~^^^ ERROR incorrect unicode escape sequence
    //~^^^^ ERROR unicode escape sequences cannot be used as a byte or in a byte string

    let _ = "\xf \u";
    //~^ ERROR invalid character in numeric character escape:
    //~^^ ERROR form of character escape may only be used with characters in the range [\x00-\x7f]
    //~^^^ ERROR incorrect unicode escape sequence

    let _ = "\u8f";
    //~^ ERROR incorrect unicode escape sequence
}
