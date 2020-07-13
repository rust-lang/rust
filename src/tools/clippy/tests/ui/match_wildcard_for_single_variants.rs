// run-rustfix

#![warn(clippy::match_wildcard_for_single_variants)]
#![allow(dead_code)]

enum Foo {
    A,
    B,
    C,
}

enum Color {
    Red,
    Green,
    Blue,
    Rgb(u8, u8, u8),
}

fn main() {
    let f = Foo::A;
    match f {
        Foo::A => {},
        Foo::B => {},
        _ => {},
    }

    let color = Color::Red;

    // check exhaustive bindings
    match color {
        Color::Red => {},
        Color::Green => {},
        Color::Rgb(_r, _g, _b) => {},
        _ => {},
    }

    // check exhaustive wild
    match color {
        Color::Red => {},
        Color::Green => {},
        Color::Rgb(..) => {},
        _ => {},
    }
    match color {
        Color::Red => {},
        Color::Green => {},
        Color::Rgb(_, _, _) => {},
        _ => {},
    }

    // shouldn't lint as there is one missing variant
    // and one that isn't exhaustively covered
    match color {
        Color::Red => {},
        Color::Green => {},
        Color::Rgb(255, _, _) => {},
        _ => {},
    }
}
