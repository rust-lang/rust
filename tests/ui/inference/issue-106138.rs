// check-pass

fn myfunction(x: &Vec<bool>, y: &Vec<i32> ) {
    let one = |i, a: &Vec<bool>| {
        a[i]  // ok
    };

    let two = |i, a: &Vec<bool>| {
        !a[i] // cannot infer type
    };

    let three = |i, b: &Vec<i32>| {
        -b[i] // ok
    };

    let r = one(0, x);
    assert_eq!(r, x[0]);
    let r = two(0, x);
    assert_eq!(r, !x[0]);
    let r = three(0, y);
    assert_eq!(r, -y[0]);
}

fn bools(x: &Vec<bool>) {
    let binary = |i, a: &Vec<bool>| {
        a[i] && a[i+1] // ok
    };

    let unary = |i, a: &Vec<bool>| {
        !a[i] // cannot infer type
    };

    let r = binary(0, x);
    assert_eq!(r, x[0] && x[1]);

    let r = unary(0, x);
    assert_eq!(r, !x[0]);
}

fn ints(x: &Vec<i32>) {
    let binary = |i, a: &Vec<i32>| {
        a[i] + a[i+1] // ok
    };
    let unary = |i, a: &Vec<i32>| {
        -a[i] // cannot infer type
    };

    let r = binary(0, x);
    assert_eq!(r, x[0] + x[1]);
    let r = unary(0, x);
    assert_eq!(r, -x[0]);
}

fn main() {
    let x = vec![true, false];
    let y = vec![1, 2, 3];
    myfunction(&x, &y);
    bools(&x);
    ints(&y);
}
