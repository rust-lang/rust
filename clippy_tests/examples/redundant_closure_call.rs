#![feature(plugin)]
#![plugin(clippy)]

#![warn(redundant_closure_call)]

fn main() {
	let a = (|| 42)();

	let mut i = 1;
	let mut k = (|m| m+1)(i);

	k = (|a,b| a*b)(1,5);

	let closure = || 32;
	i = closure();

	let closure = |i| i+1;
	i = closure(3);

	i = closure(4);
}
