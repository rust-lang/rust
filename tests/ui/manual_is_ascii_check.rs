// run-rustfix

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

    assert!((b'0'..=b'9').contains(&b'0'));
    assert!((b'a'..=b'z').contains(&b'a'));
    assert!((b'A'..=b'Z').contains(&b'A'));

    assert!(('0'..='9').contains(&'0'));
    assert!(('a'..='z').contains(&'a'));
    assert!(('A'..='Z').contains(&'A'));
}

#[clippy::msrv = "1.23"]
fn msrv_1_23() {
    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
}

#[clippy::msrv = "1.24"]
fn msrv_1_24() {
    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
}

#[clippy::msrv = "1.46"]
fn msrv_1_46() {
    const FOO: bool = matches!('x', '0'..='9');
}

#[clippy::msrv = "1.47"]
fn msrv_1_47() {
    const FOO: bool = matches!('x', '0'..='9');
}
