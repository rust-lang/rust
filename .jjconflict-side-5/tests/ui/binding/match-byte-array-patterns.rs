//@ run-pass

fn main() {
    let buf = &[0u8; 4];
    match buf {
        &[0, 1, 0, 0] => unimplemented!(),
        b"true" => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, 1, 0, 0] => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, x, 0, 0] => assert_eq!(x, 0),
        _ => unimplemented!(),
    }

    let buf: &[u8] = buf;

    match buf {
        &[0, 1, 0, 0] => unimplemented!(),
        &[_] => unimplemented!(),
        &[_, _, _, _, _, ..] => unimplemented!(),
        b"true" => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, 1, 0, 0] => unimplemented!(),
        _ => {}
    }

    match buf {
        b"true" => unimplemented!(),
        &[0, x, 0, 0] => assert_eq!(x, 0),
        _ => unimplemented!(),
    }
}
