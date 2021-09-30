#![warn(clippy::to_string_in_display)]
#![allow(clippy::inherent_to_string_shadow_display, clippy::to_string_in_format_args)]

use std::fmt;

struct A;
impl A {
    fn fmt(&self) {
        self.to_string();
    }
}

trait B {
    fn fmt(&self) {}
}

impl B for A {
    fn fmt(&self) {
        self.to_string();
    }
}

impl fmt::Display for A {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

fn fmt(a: A) {
    a.to_string();
}

struct C;

impl C {
    fn to_string(&self) -> String {
        String::from("I am C")
    }
}

impl fmt::Display for C {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

enum D {
    E(String),
    F,
}

impl std::fmt::Display for D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::E(string) => write!(f, "E {}", string.to_string()),
            Self::F => write!(f, "F"),
        }
    }
}

fn main() {
    let a = A;
    a.to_string();
    a.fmt();
    fmt(a);

    let c = C;
    c.to_string();
}
