//@no-rustfix

#![warn(clippy::unconditional_recursion)]
#![allow(clippy::partialeq_ne_impl)]

enum Foo {
    A,
    B,
}

impl PartialEq for Foo {
    fn ne(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        self != other
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        self == other
    }
}

enum Foo2 {
    A,
    B,
}

impl PartialEq for Foo2 {
    fn ne(&self, other: &Self) -> bool {
        self != &Foo2::B // no error here
    }
    fn eq(&self, other: &Self) -> bool {
        self == &Foo2::B // no error here
    }
}

enum Foo3 {
    A,
    B,
}

impl PartialEq for Foo3 {
    fn ne(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        self.ne(other)
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        self.eq(other)
    }
}

enum Foo4 {
    A,
    B,
}

impl PartialEq for Foo4 {
    fn ne(&self, other: &Self) -> bool {
        self.eq(other) // no error
    }
    fn eq(&self, other: &Self) -> bool {
        self.ne(other) // no error
    }
}

enum Foo5 {
    A,
    B,
}

impl Foo5 {
    fn a(&self) -> bool {
        true
    }
}

impl PartialEq for Foo5 {
    fn ne(&self, other: &Self) -> bool {
        self.a() // no error
    }
    fn eq(&self, other: &Self) -> bool {
        self.a() // no error
    }
}

struct S;

// Check the order doesn't matter.
impl PartialEq for S {
    fn ne(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        other != self
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        other == self
    }
}

struct S2;

// Check that if the same element is compared, it's also triggering the lint.
impl PartialEq for S2 {
    fn ne(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        other != other
    }
    fn eq(&self, other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        other == other
    }
}

struct S3;

impl PartialEq for S3 {
    fn ne(&self, _other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        self != self
    }
    fn eq(&self, _other: &Self) -> bool {
        //~^ ERROR: function cannot return without recursing
        self == self
    }
}

// There should be no warning here!
#[derive(PartialEq)]
enum E {
    A,
    B,
}

#[derive(PartialEq)]
struct Bar<T: PartialEq>(T);

struct S4;

impl PartialEq for S4 {
    fn eq(&self, other: &Self) -> bool {
        // No warning here.
        Bar(self) == Bar(other)
    }
}

macro_rules! impl_partial_eq {
    ($ty:ident) => {
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self == other
            }
        }
    };
}

struct S5;

impl_partial_eq!(S5);
//~^ ERROR: function cannot return without recursing

fn main() {
    // test code goes here
}
