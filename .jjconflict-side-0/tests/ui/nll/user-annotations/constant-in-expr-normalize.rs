trait Mirror {
    type Me;
}

impl<T> Mirror for T {
    type Me = T;
}

trait Foo<'a> {
    const C: <&'a u32 as Mirror>::Me;
}

impl<'a, T> Foo<'a> for T {
    const C: &'a u32 = &22;
}

fn foo<'a>(_: &'a u32) -> &'static u32 {
    <() as Foo<'a>>::C //~ ERROR
}

fn main() {
}
