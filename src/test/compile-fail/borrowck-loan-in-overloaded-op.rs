// xfail-test #3387

enum foo = ~uint;

#[cfg(stage0)]
impl foo: Add<foo, foo> {
    pure fn add(f: foo) -> foo {
        foo(~(**self + **f))
    }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl foo : Add<foo, foo> {
    pure fn add(f: &foo) -> foo {
        foo(~(**self + **(*f)))
    }
}

fn main() {
    let x = foo(~3);
    let _y = x + move x;
    //~^ ERROR moving out of immutable local variable prohibited due to outstanding loan
}
