// xfail-stage0
// error-pattern:Wrong type in main function: found fn(rec(int x
fn main(foo: {x: int, y: int}) { }