#![warn(clippy::single_match)]
#![allow(
    unused,
    clippy::uninlined_format_args,
    clippy::needless_if,
    clippy::redundant_guards,
    clippy::redundant_pattern_matching,
    clippy::manual_unwrap_or_default
)]
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
use Foo::*;
use std::borrow::Cow;

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

    // Don't lint
    let _ = match Some(0) {
        #[cfg(feature = "foo")]
        Some(10) => 11,
        Some(x) => x,
        _ => 0,
    };
}

fn issue_10808(bar: Option<i32>) {
    match bar {
        Some(v) => unsafe {
            let r = &v as *const i32;
            println!("{}", *r);
        },
        _ => {},
    }

    match bar {
        #[rustfmt::skip]
        Some(v) => {
            unsafe {
                let r = &v as *const i32;
                println!("{}", *r);
            }
        },
        _ => {},
    }
}

mod issue8634 {
    struct SomeError(i32, i32);

    fn foo(x: Result<i32, ()>) {
        match x {
            Ok(y) => {
                println!("Yay! {y}");
            },
            Err(()) => {
                // Ignore this error because blah blah blah.
            },
        }
    }

    fn bar(x: Result<i32, SomeError>) {
        match x {
            Ok(y) => {
                println!("Yay! {y}");
            },
            Err(_) => {
                // TODO: Process the error properly.
            },
        }
    }

    fn block_comment(x: Result<i32, SomeError>) {
        match x {
            Ok(y) => {
                println!("Yay! {y}");
            },
            Err(_) => {
                /*
                let's make sure that this also
                does not lint block comments.
                */
            },
        }
    }
}

fn issue11365() {
    enum Foo {
        A,
        B,
        C,
    }
    use Foo::{A, B, C};

    match Some(A) {
        Some(A | B | C) => println!(),
        None => {},
    }

    match Some(A) {
        Some(A | B) => println!(),
        Some { 0: C } | None => {},
    }

    match [A, A] {
        [A, _] => println!(),
        [_, A | B | C] => {},
    }

    match Ok::<_, u32>(Some(A)) {
        Ok(Some(A)) => println!(),
        Err(_) | Ok(None | Some(B | C)) => {},
    }

    match Ok::<_, u32>(Some(A)) {
        Ok(Some(A)) => println!(),
        Err(_) | Ok(None | Some(_)) => {},
    }

    match &Some(A) {
        Some(A | B | C) => println!(),
        None => {},
    }

    match &Some(A) {
        &Some(A | B | C) => println!(),
        None => {},
    }

    match &Some(A) {
        Some(A | B) => println!(),
        None | Some(_) => {},
    }
}

#[derive(Eq, PartialEq)]
pub struct Data([u8; 4]);

const DATA: Data = Data([1, 2, 3, 4]);
const CONST_I32: i32 = 1;

fn irrefutable_match() {
    match DATA {
        DATA => println!(),
        _ => {},
    }

    match CONST_I32 {
        CONST_I32 => println!(),
        _ => {},
    }

    let i = 0;
    match i {
        i => {
            let a = 1;
            let b = 2;
        },
        _ => {},
    }

    match i {
        i => {},
        _ => {},
    }

    match i {
        i => (),
        _ => (),
    }

    match CONST_I32 {
        CONST_I32 => println!(),
        _ => {},
    }
}
