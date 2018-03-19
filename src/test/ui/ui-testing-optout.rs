// disable-ui-testing-normalization

// Line number < 10
type A = B; //~ ERROR

// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Line number >=10, <100
type C = D; //~ ERROR



















































































// Line num >=100
type E = F; //~ ERROR

fn main() {}
