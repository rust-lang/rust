// run-pass

#![allow(warnings, unused)]

enum Foo {
    Variant1,
    Variant2(u32),
}

fn main() {
    let f: Foo::Variant2 = Foo::Variant2(9);
    bar(f);
}

fn bar(f: Foo) {}
