//@ run-pass

// Regression test for #47096.

fn foo(s: &[i32]) -> &[i32] {
    let &[ref xs @ ..] = s;
    xs
}

fn main() {
    let x = [1, 2, 3];
    let y = foo(&x);
    assert_eq!(x, y);
}
