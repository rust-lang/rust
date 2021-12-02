// check-pass

#![feature(generic_associated_types)]

trait Trait {
    type Assoc<'a>;
}

fn f<T: Trait>(_: T, _: impl Fn(T::Assoc<'_>)) {}

struct Type;

impl Trait for Type {
    type Assoc<'a> = ();
}

fn main() {
    f(Type, |_|());
}
