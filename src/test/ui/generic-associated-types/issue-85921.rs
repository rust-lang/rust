// check-pass

#![feature(generic_associated_types)]

trait Trait {
    type Assoc<'a>;

    fn with_assoc(f: impl FnOnce(Self::Assoc<'_>));
}

impl Trait for () {
    type Assoc<'a> = i32;

    fn with_assoc(f: impl FnOnce(Self::Assoc<'_>)) {
        f(5i32)
    }
}

fn main() {}
