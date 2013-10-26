// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




pub fn main() {
    let mut sum: int = 0;
    do first_ten |i| { info!("main"); info!("{}", i); sum = sum + i; }
    info!("sum");
    info!("{}", sum);
    assert_eq!(sum, 45);
}

fn first_ten(it: &fn(int)) {
    let mut i: int = 0;
    while i < 10 { info!("first_ten"); it(i); i = i + 1; }
}
