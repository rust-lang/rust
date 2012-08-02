fn f() -> int {
	let x: int;
	return x; //~ ERROR use of possibly uninitialized variable: `x`
}

fn main() { f(); }
