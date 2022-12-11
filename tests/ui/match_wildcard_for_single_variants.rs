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
impl Color {
    fn f(self) {
        match self {
            Self::Red => (),
            Self::Green => (),
            Self::Blue => (),
            _ => (),
        };
    }
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

    // References shouldn't change anything
    match &color {
        &Color::Red => (),
        Color::Green => (),
        &Color::Rgb(..) => (),
        &_ => (),
    }

    use self::Color as C;

    match color {
        C::Red => (),
        C::Green => (),
        C::Rgb(..) => (),
        _ => (),
    }

    match color {
        C::Red => (),
        Color::Green => (),
        Color::Rgb(..) => (),
        _ => (),
    }

    match Some(0) {
        Some(0) => 0,
        Some(_) => 1,
        _ => 2,
    };

    #[non_exhaustive]
    enum Bar {
        A,
        B,
        C,
    }
    match Bar::A {
        Bar::A => (),
        Bar::B => (),
        _ => (),
    };

    //#6984
    {
        #![allow(clippy::manual_non_exhaustive)]
        pub enum Enum {
            A,
            B,
            C,
            #[doc(hidden)]
            __Private,
        }
        match Enum::A {
            Enum::A => (),
            Enum::B => (),
            Enum::C => (),
            _ => (),
        }
        match Enum::A {
            Enum::A => (),
            Enum::B => (),
            _ => (),
        }
    }
}

mod issue9993 {
    enum Foo {
        A(bool),
        B,
    }

    fn test() {
        let _ = match Foo::A(true) {
            _ if false => 0,
            Foo::A(true) => 1,
            Foo::A(false) => 2,
            Foo::B => 3,
        };

        let _ = match Foo::B {
            _ if false => 0,
            Foo::A(_) => 1,
            _ => 2,
        };
    }
}
