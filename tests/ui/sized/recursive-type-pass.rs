//@ check-pass
trait A { type Assoc; }

impl A for () {
    // FIXME: it would be nice for this to at least cause a warning.
    type Assoc = Foo<()>;
}
struct Foo<T: A>(T::Assoc);

fn main() {}
