#![feature(plugin)]
#![plugin(clippy)]

#![deny(redundant_closure_call)]

fn main() {
	let a = (|| 42)();
	//~^ ERROR Try not to call a closure in the expression where it is declared.
	//~| HELP Try doing something like:
	//~| SUGGESTION let a = 42;

	let mut i = 1;
	let k = (|m| m+1)(i); //~ERROR Try not to call a closure in the expression where it is declared.

	k = (|a,b| a*b)(1,5); //~ERROR Try not to call a closure in the expression where it is declared.

	let closure = || 32;
	i = closure(); //~ERROR Closure called just once immediately after it was declared

	let closure = |i| i+1;
	i = closure(3); //~ERROR Closure called just once immediately after it was declared

	i = closure(4);
}

