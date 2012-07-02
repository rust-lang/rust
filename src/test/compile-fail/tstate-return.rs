pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn f() -> int {
	let x: int = 4;
	ret even(x); //~ ERROR unsatisfied precondition
}

fn main() { f(); }
