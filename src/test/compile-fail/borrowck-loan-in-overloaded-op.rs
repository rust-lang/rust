// xfail-test #3387

enum foo = ~uint;

impl foo: Add<foo, foo> {
    pure fn add(f: foo) -> foo {
        foo(~(**self + **f))
    }
}

fn main() {
    let x = foo(~3);
    let _y = x + move x;
    //~^ ERROR moving out of immutable local variable prohibited due to outstanding loan
}
