#[derive(Copy, Clone)]
union Foo {
    a: isize,
    b: (),
}

enum Bar {
    Boo = [unsafe { Foo { b: () }.a }; 4][3],
    //~^ ERROR could not evaluate enum discriminant
}

fn main() {
    assert_ne!(Bar::Boo as isize, 0);
}
