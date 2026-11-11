//@ run-pass

#![feature(postfix_match)]

struct Bar {
    foo: u8,
    baz: u8,
}

pub fn main() {
    let thing = Some("thing");

    thing.match {
        Some("nothing") => {},
        Some(text) if text.eq_ignore_ascii_case("tapir")  => {},
        Some("true") | Some("false") => {},
        Some("thing") => {},
        Some(_) => {},
        None => {}
    };

    let num = 2u8;

    num.match {
        0 => {},
        1..=5 => {},
        _ => {},
    };

    let slic = &[1, 2, 3, 4][..];

    slic.match {
        [1] => {},
        [2, _tail @ ..] => {},
        [1, _] => {},
        _ => {},
    };

    slic[0].match {
        1 => 0,
        i => i,
    };

    let out = (1, 2).match {
        (1, 3) => 0,
        (_, 1) => 0,
        (1, i) => i,
        _ => 3,
    };
    assert!(out == 2);

    let strct = Bar {
        foo: 3,
        baz: 4
    };

    strct.match {
        Bar { foo: 1, .. } => {},
        Bar { baz: 2, .. } => {},
        _ => (),
    };
}
