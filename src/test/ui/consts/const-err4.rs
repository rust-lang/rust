// stderr-per-bitwidth
#[derive(Copy, Clone)]
union Foo {
    a: isize,
    b: (),
}

enum Bar {
    Boo = [unsafe { Foo { b: () }.a }; 4][3],
    //~^ ERROR evaluation of constant value failed
    //~| uninitialized
}

fn main() {
    assert_ne!(Bar::Boo as isize, 0);
}
