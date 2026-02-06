// Self::Assoc and Self::Assoc(...) when Assoc is an associated type set to a struct.
// Unit struct: https://github.com/rust-lang/rust/issues/71054
// Tuple struct: https://github.com/rust-lang/rust/issues/120871
//
//@ run-pass
#![allow(unused)]

// --- Unit struct constructor (issue #71054) ---

trait Trait {
    type Associated;

    fn instance() -> Self::Associated;
}

struct Associated;
struct Struct;

impl Trait for Struct {
    type Associated = Associated;

    fn instance() -> Self::Associated {
        Self::Associated
    }
}

trait Trait2 {
    type Assoc;

    fn get() -> Self::Assoc;
}

struct Unit;
struct S2;

impl Trait2 for S2 {
    type Assoc = Unit;

    fn get() -> Self::Assoc {
        Self::Assoc {}
    }
}

fn _use_outer_scope() -> Associated {
    Associated
}

// --- Tuple struct constructor (issue #120871) ---

struct MyValue;
struct MyError(u32);

impl MyError {
    fn new(n: u32) -> Self {
        Self(n)
    }
}

impl std::str::FromStr for MyValue {
    type Err = MyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tuple" => Err(Self::Err(42)),
            "brace" => Err(Self::Err { 0: 42 }),
            "method" => Err(Self::Err::new(42)),
            "direct" => Err(MyError(42)),
            _ => Err(Self::Err(0)),
        }
    }
}

fn main() {
    assert_eq!("tuple".parse::<MyValue>().err().unwrap().0, 42);
    assert_eq!("brace".parse::<MyValue>().err().unwrap().0, 42);
    assert_eq!("method".parse::<MyValue>().err().unwrap().0, 42);
    assert_eq!("direct".parse::<MyValue>().err().unwrap().0, 42);
}
