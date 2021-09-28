#![deny(clippy::fallible_impl_from)]
#![allow(clippy::if_then_panic)]

// docs example
struct Foo(i32);
impl From<String> for Foo {
    fn from(s: String) -> Self {
        Foo(s.parse().unwrap())
    }
}

struct Valid(Vec<u8>);

impl<'a> From<&'a str> for Valid {
    fn from(s: &'a str) -> Valid {
        Valid(s.to_owned().into_bytes())
    }
}
impl From<usize> for Valid {
    fn from(i: usize) -> Valid {
        Valid(Vec::with_capacity(i))
    }
}

struct Invalid;

impl From<usize> for Invalid {
    fn from(i: usize) -> Invalid {
        if i != 42 {
            panic!();
        }
        Invalid
    }
}

impl From<Option<String>> for Invalid {
    fn from(s: Option<String>) -> Invalid {
        let s = s.unwrap();
        if !s.is_empty() {
            panic!("42");
        } else if s.parse::<u32>().unwrap() != 42 {
            panic!("{:?}", s);
        }
        Invalid
    }
}

trait ProjStrTrait {
    type ProjString;
}
impl<T> ProjStrTrait for Box<T> {
    type ProjString = String;
}
impl<'a> From<&'a mut <Box<u32> as ProjStrTrait>::ProjString> for Invalid {
    fn from(s: &'a mut <Box<u32> as ProjStrTrait>::ProjString) -> Invalid {
        if s.parse::<u32>().ok().unwrap() != 42 {
            panic!("{:?}", s);
        }
        Invalid
    }
}

struct Unreachable;

impl From<String> for Unreachable {
    fn from(s: String) -> Unreachable {
        if s.is_empty() {
            return Unreachable;
        }
        match s.chars().next() {
            Some(_) => Unreachable,
            None => unreachable!(), // do not lint the unreachable macro
        }
    }
}

fn main() {}
