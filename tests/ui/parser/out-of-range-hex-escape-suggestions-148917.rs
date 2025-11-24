fn main() {
    let _c = '\xFF'; //~ ERROR out of range hex escape
    let _s = "\xFF"; //~ ERROR out of range hex escape

    let _c2 = '\xff'; //~ ERROR out of range hex escape
    let _s2 = "\xff"; //~ ERROR out of range hex escape

    let _c3 = '\x80'; //~ ERROR out of range hex escape
    let _s3 = "\x80"; //~ ERROR out of range hex escape

    // Byte literals should not get suggestions (they're already valid)
    let _b = b'\xFF'; // OK
    let _bs = b"\xFF"; // OK

    dbg!('\xFF'); //~ ERROR out of range hex escape

    // do not suggest for out of range escapes that are too long
    dbg!("\xFFFFF"); //~ ERROR out of range hex escape

    dbg!("this is some kind of string \xa7"); //~ ERROR out of range hex escape
}
