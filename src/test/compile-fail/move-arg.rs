// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// error-pattern: Unsatisfied precondition constraint
fn test(foo: -int) {
    assert (foo == 10);
}

fn main() {
    let x = 10;
    test(x);
    log x;
}