// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(macro_rules)];

pub fn main() {
    macro_rules! count_arguments(
        (constant) => ( #(1) ); //~ ERROR: expected ident, found `1`
        (sequence $($e:expr)*) => ( #($($e)*) ); //~ ERROR: expected ident, found `(`
        (delimited_sequence $($e:expr),*) => ( #($($e)*) ); //~ ERROR: expected ident, found `(`
    )

    count_arguments!(constant);
    count_arguments!(sequence 1 2 3 4 5);
    count_arguments!(delimited_sequence 1, 2, 3, 4, 5);
}
