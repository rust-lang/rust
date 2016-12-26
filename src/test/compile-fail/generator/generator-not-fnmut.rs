// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators)]

fn main() {
	let mut a = Vec::<bool>::new();

	let mut test = || {
		let _: () = gen arg;
		yield 3;
		a.push(true);
		2
	};

	let a1 = test();
	let a2 = test(); //~ ERROR use of moved value
}
