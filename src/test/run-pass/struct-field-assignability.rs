struct Foo {
    x: &int
}

fn main() {
    let f = Foo { x: @3 };
    assert *f.x == 3;
}

