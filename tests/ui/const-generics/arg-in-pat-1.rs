// check-pass
enum ConstGenericEnum<const N: usize> {
    Foo([i32; N]),
    Bar,
}

fn foo<const N: usize>(val: &ConstGenericEnum<N>) {
    if let ConstGenericEnum::<N>::Foo(field, ..) = val {}
}

fn bar<const N: usize>(val: &ConstGenericEnum<N>) {
    match val {
        ConstGenericEnum::<N>::Foo(field, ..) => (),
        ConstGenericEnum::<N>::Bar => (),
    }
}

fn main() {
    match ConstGenericEnum::Bar {
        ConstGenericEnum::<3>::Foo(field, ..) => (),
        ConstGenericEnum::<3>::Bar => (),
    }
}
