//@ check-pass
//@ compile-flags: -Znext-solver

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

type Foo<T> = <Option<T> as Mirror>::Assoc;

fn main() {
    let x = Foo::<i32>::None;
}
