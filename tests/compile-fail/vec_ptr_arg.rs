#![feature(plugin)]
#![plugin(clippy)]

#[deny(vec_ptr_arg)]
#[allow(unused)]
fn go(x: &Vec<i64>) { //~ERROR: Writing '&Vec<_>' instead of '&[_]'
	//Nothing here
}


fn main() {
	let x = vec![1i64, 2, 3];
	go(&x);
}
