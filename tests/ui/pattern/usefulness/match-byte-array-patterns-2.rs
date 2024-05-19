fn main() {
    let buf = &[0, 1, 2, 3];

    match buf { //~ ERROR non-exhaustive
        b"AAAA" => {}
    }

    let buf: &[u8] = buf;

    match buf { //~ ERROR non-exhaustive
        b"AAAA" => {}
    }
}
