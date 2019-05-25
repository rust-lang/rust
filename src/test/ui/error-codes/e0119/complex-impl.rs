// aux-build:complex_impl_support.rs

extern crate complex_impl_support;

use complex_impl_support::{External, M};

struct Q;

impl<R> External for (Q, R) {} //~ ERROR must be used
//~^ ERROR conflicting implementations of trait

fn main() {}
