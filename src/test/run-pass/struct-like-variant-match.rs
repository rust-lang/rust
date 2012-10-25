enum Foo {
    Bar {
        x: int,
        y: int
    },
    Baz {
        x: float,
        y: float
    }
}

fn f(x: &Foo) {
    match *x {
        Baz { x: x, y: y } => {
            assert x == 1.0;
            assert y == 2.0;
        }
        Bar { y: y, x: x } => {
            assert x == 1;
            assert y == 2;
        }
    }
}

fn main() {
    let x = Bar { x: 1, y: 2 };
    f(&x);
    let y = Baz { x: 1.0, y: 2.0 };
    f(&y);
}

