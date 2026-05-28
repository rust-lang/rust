fn main() {
    let _ = b"\u{a66e}";
    //~^ ERROR unicode escape in byte string

    let _ = b'\u{a66e}';
    //~^ ERROR unicode escape in byte string

    let _ = b'\u';
    //~^ ERROR incorrect unicode escape sequence

    let _ = b'\x5';
    //~^ ERROR numeric character escape is too short

    let _ = b'\xxy';
    //~^ ERROR invalid character in numeric character escape: `x`

    let _ = '\x5';
    //~^ ERROR numeric character escape is too short

    let _ = '\xxy';
    //~^ ERROR invalid character in numeric character escape: `x`

    let _ = b"\u{a4a4} \xf \u";
    //~^ ERROR unicode escape in byte string
    //~^^ ERROR invalid character in numeric character escape: ` `
    //~^^^ ERROR incorrect unicode escape sequence

    let _ = "\xf \u";
    //~^ ERROR invalid character in numeric character escape: ` `
    //~^^ ERROR incorrect unicode escape sequence

    let _ = "\u8f";
    //~^ ERROR incorrect unicode escape sequence
}
