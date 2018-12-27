#[derive(Copy, Clone)]
union Foo {
    a: isize,
    b: (),
}

enum Bar {
    Boo = [unsafe { Foo { b: () }.a }; 4][3],
    //~^ ERROR it is undefined behavior to use this value
}

fn main() {
    assert_ne!(Bar::Boo as isize, 0);
}
