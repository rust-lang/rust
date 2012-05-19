pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn main() {
    let x: int = 4;
    while even(x) != 0 { } //! ERROR unsatisfied precondition
}
