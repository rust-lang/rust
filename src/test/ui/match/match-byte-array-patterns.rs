#![feature(slice_patterns)]
#![deny(unreachable_patterns)]

fn main() {
    let buf = &[0, 1, 2, 3];

    match buf {
        b"AAAA" => {},
        &[0x41, 0x41, 0x41, 0x41] => {} //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, 0x41, 0x41, 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[_, 0x41, 0x41, 0x41] => {},
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, .., 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    let buf: &[u8] = buf;

    match buf {
        b"AAAA" => {},
        &[0x41, 0x41, 0x41, 0x41] => {} //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, 0x41, 0x41, 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[_, 0x41, 0x41, 0x41] => {},
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }

    match buf {
        &[0x41, .., 0x41] => {}
        b"AAAA" => {}, //~ ERROR unreachable pattern
        _ => {}
    }
}
