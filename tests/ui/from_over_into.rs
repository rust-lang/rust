#![warn(clippy::from_over_into)]

// this should throw an error
struct StringWrapper(String);

impl Into<StringWrapper> for String {
    fn into(self) -> StringWrapper {
        StringWrapper(self)
    }
}

// this is fine
struct A(String);

impl From<String> for A {
    fn from(s: String) -> A {
        A(s)
    }
}

fn main() {}
