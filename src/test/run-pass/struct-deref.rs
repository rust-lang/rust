struct Foo(int);

fn main() {
    let x: Foo = Foo(2);
    assert *x == 2;
}

