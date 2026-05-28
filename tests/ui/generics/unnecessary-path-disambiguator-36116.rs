// https://github.com/rust-lang/rust/issues/36116
// Unnecessary path disambiguator is ok

//@ check-pass

macro_rules! m {
    ($p: path) => {
        let _ = $p(0);
        let _: $p;
    }
}

struct Foo<T> {
    _a: T,
}

struct S<T>(T);

fn f() {
    let f = Some(Foo { _a: 42 }).map(|a| a as Foo::<i32>);
    let g: Foo::<i32> = Foo { _a: 42 };

    m!(S::<u8>);
}

fn main() {}
