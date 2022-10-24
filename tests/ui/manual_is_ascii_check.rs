// run-rustfix

#![feature(custom_inner_attributes)]
#![allow(unused, dead_code)]
#![warn(clippy::manual_is_ascii_check)]

fn main() {
    assert!(matches!('x', 'a'..='z'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!(b'x', b'a'..=b'z'));
    assert!(matches!(b'X', b'A'..=b'Z'));

    let num = '2';
    assert!(matches!(num, '0'..='9'));
    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));

    assert!(matches!('x', 'A'..='Z' | 'a'..='z' | '_'));
}

fn msrv_1_23() {
    #![clippy::msrv = "1.23"]

    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
}

fn msrv_1_24() {
    #![clippy::msrv = "1.24"]

    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
}

fn msrv_1_46() {
    #![clippy::msrv = "1.46"]
    const FOO: bool = matches!('x', '0'..='9');
}

fn msrv_1_47() {
    #![clippy::msrv = "1.47"]
    const FOO: bool = matches!('x', '0'..='9');
}
