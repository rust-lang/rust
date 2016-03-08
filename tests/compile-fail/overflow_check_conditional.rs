#![feature(plugin)]
#![plugin(clippy)]

#![allow(many_single_char_names)]
#![deny(overflow_check_conditional)]

fn main() {
	let a: u32 = 1;
	let b: u32 = 2;
	let c: u32 = 3;
	if a + b < a { //~ERROR You are trying to use classic C overflow conditions that will fail in Rust.

	}
	if a > a + b { //~ERROR You are trying to use classic C overflow conditions that will fail in Rust.

	}
	if a + b < b { //~ERROR You are trying to use classic C overflow conditions that will fail in Rust.

	}
	if b > a + b { //~ERROR You are trying to use classic C overflow conditions that will fail in Rust.

	}
	if a - b > b { //~ERROR You are trying to use classic C underflow conditions that will fail in Rust.

	}
	if b < a - b { //~ERROR You are trying to use classic C underflow conditions that will fail in Rust.

	}
	if a - b > a { //~ERROR You are trying to use classic C underflow conditions that will fail in Rust.

	}
	if a < a - b { //~ERROR You are trying to use classic C underflow conditions that will fail in Rust.

	}
	if a + b < c {

	}
	if c > a + b {

	}
	if a - b < c {

	}
	if c > a - b {

	}
	let i = 1.1;
	let j = 2.2;
	if i + j < i {

	}
	if i - j < i {

	}
	if i > i + j {

	}
	if i - j < i {

	}
}
