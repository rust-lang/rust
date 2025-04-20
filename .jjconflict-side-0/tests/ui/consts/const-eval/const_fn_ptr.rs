//@ run-pass
//@ compile-flags: -Zunleash-the-miri-inside-of-you

fn double(x: usize) -> usize { x * 2 }
const fn double_const(x: usize) -> usize { x * 2 }

const X: fn(usize) -> usize = double;
const X_CONST: fn(usize) -> usize = double_const;

const fn bar(x: usize) -> usize {
    X(x)
}

const fn bar_const(x: usize) -> usize {
    X_CONST(x)
}

const fn foo(x: fn(usize) -> usize, y: usize)  -> usize {
    x(y)
}

fn main() {
    const Y: usize = bar_const(2);
    assert_eq!(Y, 4);
    let y = bar_const(2);
    assert_eq!(y, 4);
    let y = bar(2);
    assert_eq!(y, 4);

    const Z: usize = foo(double_const, 2);
    assert_eq!(Z, 4);
    let z = foo(double_const, 2);
    assert_eq!(z, 4);
    let z = foo(double, 2);
    assert_eq!(z, 4);
}

//~? WARN skipping const checks
