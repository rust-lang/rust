//@ check-pass
//@ compile-flags: -Znext-solver

trait Foo {
    type Gat<'a>
    where
        Self: 'a;
    fn bar(&self) -> Self::Gat<'_>;
}

enum Option<T> {
    Some(T),
    None,
}

impl<T> Option<T> {
    fn as_ref(&self) -> Option<&T> {
        match self {
            Option::Some(t) => Option::Some(t),
            Option::None => Option::None,
        }
    }

    fn map<U>(self, f: impl FnOnce(T) -> U) -> Option<U> {
        match self {
            Option::Some(t) => Option::Some(f(t)),
            Option::None => Option::None,
        }
    }
}

impl<T: Foo + 'static> Foo for Option<T> {
    type Gat<'a> = Option<<T as Foo>::Gat<'a>> where Self: 'a;

    fn bar(&self) -> Self::Gat<'_> {
        self.as_ref().map(Foo::bar)
    }
}

fn main() {}
