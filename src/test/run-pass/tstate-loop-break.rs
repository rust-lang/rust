// xfail-test

pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn test() {
    let v = 4;
    loop {
        check is_even(v);
        break;
    }
    even(v);
}

fn main() {
    test();
}
