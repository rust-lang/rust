trait Foo<'a> {
    const C: &'a u32;
}

impl<'a, T> Foo<'a> for T {
    const C: &'a u32 = &22;
}

fn foo<'a>(_: &'a u32) -> &'static u32 {
    <() as Foo<'a>>::C //~ ERROR
}

fn main() {
}
