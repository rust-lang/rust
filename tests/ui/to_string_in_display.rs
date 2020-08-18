#![warn(clippy::to_string_in_display)]
#![allow(clippy::inherent_to_string_shadow_display)]

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

fn main() {
    let a = A;
    a.to_string();
    a.fmt();
    fmt(a);

    let c = C;
    c.to_string();
}
