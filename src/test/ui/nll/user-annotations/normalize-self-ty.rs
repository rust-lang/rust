// Regression test for #55183: check a case where the self type from
// the inherent impl requires normalization to be equal to the
// user-provided type.
//
// run-pass

trait Mirror {
    type Me;
}

impl<T> Mirror for T {
    type Me = T;
}

struct Foo<A, B>(A, B);

impl<A> Foo<A, <A as Mirror>::Me> {
    fn m(b: A) { }
}

fn main() {
    <Foo<&'static u32, &u32>>::m(&22);
}
