//@ run-pass
// Ensure non-capturing Closure passing CoerceMany work correctly.
fn foo(_: usize) -> usize {
    0
}

fn bar(_: usize) -> usize {
    1
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    // Coerce result check

    type FnPointer = fn(usize) -> usize;

    let c = |x| x;
    let c_pointer: FnPointer = c;
    assert_eq!(c_pointer(42), 42);

    let f = match 0 {
        0 => foo,
        1 => |_| 1,
        _ => unimplemented!(),
    };
    assert_eq!(f(42), 0);

    let f = match 2 {
        2 => |_| 2,
        0 => foo,
        _ => unimplemented!(),
    };
    assert_eq!(f(42), 2);

    let f = match 1 {
        0 => foo,
        1 => bar,
        2 => |_| 2,
        _ => unimplemented!(),
    };
    assert_eq!(f(42), 1);

    let clo0 = |_: usize| 0;
    let clo1 = |_| 1;
    let clo2 = |_| 2;
    let f = match 0 {
        0 => clo0,
        1 => clo1,
        2 => clo2,
        _ => unimplemented!(),
    };
    assert_eq!(f(42), 0);

    let funcs = [add, |a, b| (a - b) as i32];
    assert_eq!([funcs[0](5, 5), funcs[1](5, 5)], [10, 0]);
}
