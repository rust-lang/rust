fn f1(a: {mutable x: int}, b: &mutable int, c: -int) -> int {
    let r = a.x + b + c;
    a.x = 0;
    b = 10;
    c = 20;
    ret r;
}

fn f2(a: int, f: block(int)) -> int { f(1); ret a; }

fn main() {
    let a = {mutable x: 1}, b = 2, c = 3;
    assert (f1(a, b, c) == 6);
    assert (a.x == 0);
    assert (b == 10);
    assert (f2(a.x, {|x| a.x = 50; }) == 0);
    assert (a.x == 50);
}
