fn f() -> int {
	let x: int;
	ret x; //! ERROR use of possibly uninitialized variable: `x`
}

fn main() { f(); }
