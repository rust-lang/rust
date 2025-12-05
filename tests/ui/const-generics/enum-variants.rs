//@ check-pass
enum Foo<const N: usize> {
    Variant,
    Variant2(),
    Variant3{},
}

struct Bar<const N: usize>;
struct Bar2<const N: usize>();
struct Bar3<const N: usize> {}

fn main() {
    let _ = Foo::Variant::<1>;
    let _ = Foo::Variant2::<1>();
    let _ = Foo::Variant3::<1>{};

    let _ = Foo::<1>::Variant;
    let _ = Foo::<1>::Variant2();
    let _ = Foo::<1>::Variant3{};

    let _ = Bar::<1>;
    let _ = Bar2::<1>();
    let _ = Bar3::<1>{};
}
