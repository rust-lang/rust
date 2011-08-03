fn add(i: int, j: int) -> int { ret i + j; }
fn binder(n: int) -> fn() -> int {
    let f = bind add(n, _);
    ret bind f(2);
}
fn main() {
    binder(5);
    let f = binder(1);
    assert(f() == 3);
    assert(binder(8)() == 10);
}
