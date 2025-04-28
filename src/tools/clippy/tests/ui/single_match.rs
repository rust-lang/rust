//@require-annotations-for-level: WARN
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
    //~^^^^^^ single_match

    let x = Some(1u8);
    match x {
        // Note the missing block braces.
        // We suggest `if let Some(y) = x { .. }` because the macro
        // is expanded before we can do anything.
        Some(y) => println!("{:?}", y),
        _ => (),
    }
    //~^^^^^^^ single_match
    //~| NOTE: you might want to preserve the comments from inside the `match`

    let z = (1u8, 1u8);
    match z {
        (2..=3, 7..=9) => dummy(),
        _ => {},
    };
    //~^^^^ single_match

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
    //~^^^^ single_match

    match y {
        Ok(y) => dummy(),
        Err(..) => (),
    };
    //~^^^^ single_match

    let c = Cow::Borrowed("");

    match c {
        Cow::Borrowed(..) => dummy(),
        Cow::Owned(..) => (),
    };
    //~^^^^ single_match

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
    //~^^^^ single_match

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
    //~^^^^ single_match

    const FOO_C: Foo = Foo::C(0);
    match x {
        FOO_C => println!(),
        _ => (),
    }
    //~^^^^ single_match

    match &&x {
        Foo::A => println!(),
        _ => (),
    }
    //~^^^^ single_match

    let x = &x;
    match &x {
        Foo::A => println!(),
        _ => (),
    }
    //~^^^^ single_match

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
    //~^^^^ single_match

    // issue #7038
    struct X;
    let x = Some(X);
    match x {
        None => println!(),
        _ => (),
    };
    //~^^^^ single_match
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
    //~^^^^ single_match

    // lint
    match x {
        (Some(E::V), _) => todo!(),
        (_, _) => {},
    }
    //~^^^^ single_match

    // lint
    match (Some(42), Some(E::V), Some(42)) {
        (.., Some(E::V), _) => {},
        (..) => {},
    }
    //~^^^^ single_match

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
    //~^^^^^^^ single_match

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
    //~^^^^^^^^^^ single_match
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
    //~^^^^ single_match

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
    //~^^^^ single_match
}

fn issue12758(s: &[u8]) {
    match &s[0..3] {
        b"foo" => println!(),
        _ => {},
    }
    //~^^^^ single_match
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
    //~^^^^ single_match

    match CONST_I32 {
        CONST_I32 => println!(),
        _ => {},
    }
    //~^^^^ single_match

    let i = 0;
    match i {
        i => {
            let a = 1;
            let b = 2;
        },
        _ => {},
    }
    //~^^^^^^^ single_match

    match i {
        i => {},
        _ => {},
    }
    //~^^^^ single_match

    match i {
        i => (),
        _ => (),
    }
    //~^^^^ single_match

    match CONST_I32 {
        CONST_I32 => println!(),
        _ => {},
    }
    //~^^^^ single_match

    let mut x = vec![1i8];

    match x.pop() {
        // bla
        Some(u) => println!("{u}"),
        // more comments!
        None => {},
    }
    //~^^^^^^ single_match
    //~| NOTE: you might want to preserve the comments from inside the `match`

    match x.pop() {
        // bla
        Some(u) => {
            // bla
            println!("{u}");
        },
        // bla
        None => {},
    }
    //~^^^^^^^^^ single_match
    //~| NOTE: you might want to preserve the comments from inside the `match`
}

fn issue_14493() {
    macro_rules! mac {
        (some) => {
            Some(42)
        };
        (any) => {
            _
        };
        (str) => {
            "foo"
        };
    }

    match mac!(some) {
        Some(u) => println!("{u}"),
        _ => (),
    }
    //~^^^^ single_match

    // When scrutinee comes from macro, do not tell that arm will always match
    // and suggest an equality check instead.
    match mac!(str) {
        "foo" => println!("eq"),
        _ => (),
    }
    //~^^^^ ERROR: for an equality check

    // Do not lint if any match arm come from expansion
    match Some(0) {
        mac!(some) => println!("eq"),
        mac!(any) => println!("neq"),
    }
    match Some(0) {
        Some(42) => println!("eq"),
        mac!(any) => println!("neq"),
    }
    match Some(0) {
        mac!(some) => println!("eq"),
        _ => println!("neq"),
    }
}
