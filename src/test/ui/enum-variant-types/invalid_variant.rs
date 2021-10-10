enum Foo {
    Variant1,
    Variant2(u32),
}

pub fn main() {
    let x: Foo::Variant2 = Foo::Variant2(9);
    bar(x);
    //~^ ERROR mismatched types [E0308]
}

fn bar(x: Foo::Variant1) -> Foo {
    x
}
