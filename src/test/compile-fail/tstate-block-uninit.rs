pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn force(f: fn()) { f(); }

fn main() {
    let x: int = 4;
    force(fn&() {
        even(x); //~ ERROR unsatisfied precondition
    });
}
