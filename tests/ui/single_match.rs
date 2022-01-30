#![warn(clippy::single_match)]

fn dummy() {}

fn single_match() {
    let x = Some(1u8);

    match x {
        Some(y) => {
            println!("{:?}", y);
        },
        _ => (),
    };

    let x = Some(1u8);
    match x {
        // Note the missing block braces.
        // We suggest `if let Some(y) = x { .. }` because the macro
        // is expanded before we can do anything.
        Some(y) => println!("{:?}", y),
        _ => (),
    }

    let z = (1u8, 1u8);
    match z {
        (2..=3, 7..=9) => dummy(),
        _ => {},
    };

    // Not linted (pattern guards used)
    match x {
        Some(y) if y == 0 => println!("{:?}", y),
        _ => (),
    }

    // Not linted (no block with statements in the single arm)
    match z {
        (2..=3, 7..=9) => println!("{:?}", z),
        _ => println!("nope"),
    }
}

enum Foo {
    Bar,
    Baz(u8),
}
use std::borrow::Cow;
use Foo::*;

fn single_match_know_enum() {
    let x = Some(1u8);
    let y: Result<_, i8> = Ok(1i8);

    match x {
        Some(y) => dummy(),
        None => (),
    };

    match y {
        Ok(y) => dummy(),
        Err(..) => (),
    };

    let c = Cow::Borrowed("");

    match c {
        Cow::Borrowed(..) => dummy(),
        Cow::Owned(..) => (),
    };

    let z = Foo::Bar;
    // no warning
    match z {
        Bar => println!("42"),
        Baz(_) => (),
    }

    match z {
        Baz(_) => println!("42"),
        Bar => (),
    }
}

// issue #173
fn if_suggestion() {
    let x = "test";
    match x {
        "test" => println!(),
        _ => (),
    }

    #[derive(PartialEq, Eq)]
    enum Foo {
        A,
        B,
        C(u32),
    }

    let x = Foo::A;
    match x {
        Foo::A => println!(),
        _ => (),
    }

    const FOO_C: Foo = Foo::C(0);
    match x {
        FOO_C => println!(),
        _ => (),
    }

    match &&x {
        Foo::A => println!(),
        _ => (),
    }

    let x = &x;
    match &x {
        Foo::A => println!(),
        _ => (),
    }

    enum Bar {
        A,
        B,
    }
    impl PartialEq for Bar {
        fn eq(&self, rhs: &Self) -> bool {
            matches!((self, rhs), (Self::A, Self::A) | (Self::B, Self::B))
        }
    }
    impl Eq for Bar {}

    let x = Bar::A;
    match x {
        Bar::A => println!(),
        _ => (),
    }

    // issue #7038
    struct X;
    let x = Some(X);
    match x {
        None => println!(),
        _ => (),
    };
}

// See: issue #8282
fn ranges() {
    enum E {
        V,
    }
    let x = (Some(E::V), Some(42));

    // Don't lint, because the `E` enum can be extended with additional fields later. Thus, the
    // proposed replacement to `if let Some(E::V)` may hide non-exhaustive warnings that appeared
    // because of `match` construction.
    match x {
        (Some(E::V), _) => {},
        (None, _) => {},
    }

    // lint
    match x {
        (Some(_), _) => {},
        (None, _) => {},
    }

    // lint
    match x {
        (Some(E::V), _) => todo!(),
        (_, _) => {},
    }

    // lint
    match (Some(42), Some(E::V), Some(42)) {
        (.., Some(E::V), _) => {},
        (..) => {},
    }

    // Don't lint, see above.
    match (Some(E::V), Some(E::V), Some(E::V)) {
        (.., Some(E::V), _) => {},
        (.., None, _) => {},
    }

    // Don't lint, see above.
    match (Some(E::V), Some(E::V), Some(E::V)) {
        (Some(E::V), ..) => {},
        (None, ..) => {},
    }

    // Don't lint, see above.
    match (Some(E::V), Some(E::V), Some(E::V)) {
        (_, Some(E::V), ..) => {},
        (_, None, ..) => {},
    }
}

fn skip_type_aliases() {
    enum OptionEx {
        Some(i32),
        None,
    }
    enum ResultEx {
        Err(i32),
        Ok(i32),
    }

    use OptionEx::{None, Some};
    use ResultEx::{Err, Ok};

    // don't lint
    match Err(42) {
        Ok(_) => dummy(),
        Err(_) => (),
    };

    // don't lint
    match Some(1i32) {
        Some(_) => dummy(),
        None => (),
    };
}

macro_rules! single_match {
    ($num:literal) => {
        match $num {
            15 => println!("15"),
            _ => (),
        }
    };
}

fn main() {
    single_match!(5);
}
