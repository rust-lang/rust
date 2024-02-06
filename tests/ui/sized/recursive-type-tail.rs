//@ check-fail
//@ error-pattern: reached the recursion limit finding the struct tail

trait A { type Assoc; }

impl A for () {
    type Assoc = Foo<()>;
}
struct Foo<T: A>(T::Assoc);

fn main() {}
