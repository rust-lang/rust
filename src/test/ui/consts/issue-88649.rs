// check-pass
#![crate_type = "lib"]

enum Foo {
    Variant1(bool),
    Variant2(bool),
}

const _: () = {
    let mut n = 0;
    while n < 2 {
        match Foo::Variant1(true) {
            Foo::Variant1(x) | Foo::Variant2(x) if x => {}
            _ => {}
        }
        n += 1;
    }
};
