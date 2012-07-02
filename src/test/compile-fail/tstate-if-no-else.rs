pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn foo(x: int) { log(debug, x); }

fn main() {
	let x: int = 10;
        if 1 > 2 { check is_even(x); }
        even(x); //~ ERROR unsatisfied precondition
}
