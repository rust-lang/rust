pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn f() {
    let mut x: int = 10;
    while 1 == 1 { x = 10; }
    even(x); //! ERROR unsatisfied precondition
}

fn main() { f(); }
