// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
fn checktrue(rs: bool) -> bool { assert!((rs)); return true; }

pub fn main() { let k = checktrue; evenk(42, k); oddk(45, k); }

fn evenk(n: int, k: extern fn(bool) -> bool) -> bool {
    info!("evenk");
    info!(n);
    if n == 0 { return k(true); } else { return oddk(n - 1, k); }
}

fn oddk(n: int, k: extern fn(bool) -> bool) -> bool {
    info!("oddk");
    info!(n);
    if n == 0 { return k(false); } else { return evenk(n - 1, k); }
}
