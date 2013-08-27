// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run
// debugger:finish

// debugger:print a
// check:$1 = 10101
// debugger:continue

// debugger:finish
// debugger:print b
// check:$2 = 20202
// debugger:continue

// debugger:finish
// debugger:print c
// check:$3 = 30303

#[allow(unused_variable)];

fn function_one() {
	let a = 10101;
	zzz();
}

fn function_two() {
	let b = 20202;
	zzz();
}


fn function_three() {
	let c = 30303;
	zzz();
}


fn main() {
	function_one();
	function_two();
	function_three();
}

fn zzz() {()}
