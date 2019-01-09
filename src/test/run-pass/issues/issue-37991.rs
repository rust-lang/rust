// run-pass

const fn foo() -> i64 {
    3
}

const fn bar(x: i64) -> i64 {
    x*2
}

fn main() {
    let val = &(foo() % 2);
    assert_eq!(*val, 1);

    let val2 = &(bar(1+1) % 3);
    assert_eq!(*val2, 1);
}
