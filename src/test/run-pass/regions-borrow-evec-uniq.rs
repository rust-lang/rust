fn foo(x: &[int]) -> int {
    x[0]
}

fn main() {
    let p = ~[1,2,3,4,5];
    let r = foo(p);
    assert r == 1;

    let p = ~[5,4,3,2,1];
    let r = foo(p);
    assert r == 5;
}
