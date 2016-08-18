// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// In expression position, but not statement position, when we expand a macro,
// we replace the span of the expanded expression with that of the call site.

macro_rules! nested_expr {
    () => (fake)
}

macro_rules! call_nested_expr {
    () => (nested_expr!())
}

macro_rules! call_nested_expr_sum {
    () => { 1 + nested_expr!(); }
}

fn main() {
    1 + call_nested_expr!();
    call_nested_expr_sum!();
}
