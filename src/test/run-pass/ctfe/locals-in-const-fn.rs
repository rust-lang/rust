// run-pass

// https://github.com/rust-lang/rust/issues/48821

const fn foo(i: usize) -> usize {
    let x = i;
    x
}

static FOO: usize = foo(42);

const fn bar(mut i: usize) -> usize {
    i += 8;
    let x = &i;
    *x
}

static BAR: usize = bar(42);

const fn boo(mut i: usize) -> usize {
    {
        let mut x = i;
        x += 10;
        i = x;
    }
    i
}

static BOO: usize = boo(42);

fn main() {
    assert!(FOO == 42);
    assert!(BAR == 50);
    assert!(BOO == 52);
}
