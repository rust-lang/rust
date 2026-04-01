#![warn(clippy::recursive_format_impl)]
#![allow(
    clippy::borrow_deref_ref,
    clippy::deref_addrof,
    clippy::inherent_to_string_shadow_display,
    clippy::to_string_in_format_args,
    clippy::uninlined_format_args
)]

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
        //~^ recursive_format_impl
    }
}

fn fmt(a: A) {
    a.to_string();
}

struct C;

impl C {
    // Doesn't trigger if to_string defined separately
    // i.e. not using ToString trait (from Display)
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

// Check for use of self as Display, in Display impl
// Triggers on direct use of self
struct G;

impl std::fmt::Display for G {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
        //~^ recursive_format_impl
    }
}

// Triggers on reference to self
struct H;

impl std::fmt::Display for H {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self)
        //~^ recursive_format_impl
    }
}

impl std::fmt::Debug for H {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self)
        //~^ recursive_format_impl
    }
}

// Triggers on multiple reference to self
struct H2;

impl std::fmt::Display for H2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &&&self)
        //~^ recursive_format_impl
    }
}

// Doesn't trigger on correct deref
struct I;

impl std::ops::Deref for I {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for I {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &**self)
    }
}

impl std::fmt::Debug for I {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", &**self)
    }
}

// Doesn't trigger on multiple correct deref
struct I2;

impl std::ops::Deref for I2 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for I2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", **&&&**self)
    }
}

// Doesn't trigger on multiple correct deref
struct I3;

impl std::ops::Deref for I3 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for I3 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &&**&&&**self)
    }
}

// Does trigger when deref resolves to self
struct J;

impl std::ops::Deref for J {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for J {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &*self)
        //~^ recursive_format_impl
    }
}

impl std::fmt::Debug for J {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", &*self)
        //~^ recursive_format_impl
    }
}

struct J2;

impl std::ops::Deref for J2 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for J2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", *self)
        //~^ recursive_format_impl
    }
}

struct J3;

impl std::ops::Deref for J3 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for J3 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", **&&*self)
        //~^ recursive_format_impl
    }
}

struct J4;

impl std::ops::Deref for J4 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "test"
    }
}

impl std::fmt::Display for J4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &&**&&*self)
        //~^ recursive_format_impl
    }
}

// Doesn't trigger on Debug from Display
struct K;

impl std::fmt::Debug for K {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "test")
    }
}

impl std::fmt::Display for K {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// Doesn't trigger on Display from Debug
struct K2;

impl std::fmt::Debug for K2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for K2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "test")
    }
}

// Doesn't trigger on struct fields
struct L {
    field1: u32,
    field2: i32,
}

impl std::fmt::Display for L {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{},{}", self.field1, self.field2)
    }
}

impl std::fmt::Debug for L {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?},{:?}", self.field1, self.field2)
    }
}

// Doesn't trigger on nested enum matching
enum Tree {
    Leaf,
    Node(Vec<Tree>),
}

impl std::fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tree::Leaf => write!(f, "*"),
            Tree::Node(children) => {
                write!(f, "(")?;
                for child in children.iter() {
                    write!(f, "{},", child)?;
                }
                write!(f, ")")
            },
        }
    }
}

impl std::fmt::Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Tree::Leaf => write!(f, "*"),
            Tree::Node(children) => {
                write!(f, "(")?;
                for child in children.iter() {
                    write!(f, "{:?},", child)?;
                }
                write!(f, ")")
            },
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
