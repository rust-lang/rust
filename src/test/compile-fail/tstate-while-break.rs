pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn test(cond: bool) {
    let v = 4;
    while cond {
        check is_even(v);
        break;
    }
    even(v); //~ ERROR unsatisfied precondition
}

fn main() {
    test(true);
}
