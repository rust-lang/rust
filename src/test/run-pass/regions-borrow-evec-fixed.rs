// xfail-test

fn foo(x: &[int]) -> int {
    x[0]
}

fn main() {
    let p = [1,2,3,4,5];
    assert foo(p) == 1;
}
