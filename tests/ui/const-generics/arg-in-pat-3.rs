//@ check-pass
struct Foo<const N: usize>;

fn bindingp() {
    match Foo {
        mut x @ Foo::<3> => {
            let ref mut _x @ Foo::<3> = x;
        }
    }
}

struct Bar<const N: usize> {
    field: Foo<N>,
}

fn structp() {
    match todo!() {
        Bar::<3> {
            field: Foo::<3>,
        } => (),
    }
}

struct Baz<const N: usize>(Foo<N>);

fn tuplestructp() {
    match Baz(Foo) {
        Baz::<3>(Foo::<3>) => (),
    }
}

impl<const N: usize> Baz<N> {
    const ASSOC: usize = 3;
}

fn pathp() {
    match 3 {
        Baz::<3>::ASSOC => (),
        _ => (),
    }
}

fn main() {}
