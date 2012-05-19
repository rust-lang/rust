pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }
fn main() {
    let i: int = 4;
    log(debug, false || { check is_even(i); true });
    even(i); //! ERROR unsatisfied precondition
}
