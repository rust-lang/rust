trait Trait<T> {
    fn foo<'a, K>(self, _: T, _: K) where T: 'a, K: 'a;
}

impl Trait<()> for () {
    fn foo<'a, K>(self, _: (), _: K) where { //~ ERROR E0195
        todo!();
    }
}

struct State;

trait Foo<T> {
    fn foo<'a>(&self, state: &'a State) -> &'a T
    where
        T: 'a;
}

impl<F, T> Foo<T> for F
where
    F: Fn(&State) -> &T,
{
    fn foo<'a>(&self, state: &'a State) -> &'a T { //~ ERROR E0195
        self(state)
    }
}

trait Bar {
    fn foo<'a>(&'a self) {}
}

impl Bar for () {
    fn foo<'a: 'a>(&'a self) {} //~ ERROR E0195
}

fn main() {
    ().foo((), ());
}
