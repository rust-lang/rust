#![feature(plugin)]
#![plugin(clippy)]

#![deny(overflow_check_conditional)]

fn main() {
	let a: u32 = 1;
	let b: u32 = 2;
	let c: u32 = 3;
	if a + b < a { //~ERROR You are trying to use classic C overflow conditons that will fail in Rust.

	}
	if a + b < b { //~ERROR You are trying to use classic C overflow conditons that will fail in Rust.

	}
	if a - b > b { //~ERROR You are trying to use classic C underflow conditons that will fail in Rust.

	}
	if a - b > a { //~ERROR You are trying to use classic C underflow conditons that will fail in Rust.

	}
	if a + b < c {

	}
	if a - b < c {

	}
	let i = 1.1;
	let j = 2.2;
	if i + j < i {

	}
	if i - j < i {

	}
}

