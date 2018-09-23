// Test that we don't assume that type aliases have the same type parameters
// as the type they alias and then panic when we see this.

#![feature(nll)]

type a<'a> = &'a isize;
type b<'a> = Box<a<'a>>;

struct c<'a> {
    f: Box<b<'a>>
}

trait FromBox<'a> {
    fn from_box(b: Box<b>) -> Self;
}

impl<'a> FromBox<'a> for c<'a> {
    fn from_box(b: Box<b>) -> Self {
        c { f: b } //~ ERROR
    }
}

trait FromTuple<'a> {
    fn from_tuple( b: (b,)) -> Self;
}

impl<'a> FromTuple<'a> for c<'a> {
    fn from_tuple(b: (b,)) -> Self {
        c { f: Box::new(b.0) } //~ ERROR
    }
}

fn main() {}
