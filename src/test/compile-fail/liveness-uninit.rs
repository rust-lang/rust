fn foo(x: int) { log(debug, x); }

fn main() {
	let x: int;
	foo(x); //~ ERROR use of possibly uninitialized variable: `x`
}
