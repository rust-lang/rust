// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we are able to distinguish when loans borrow different
// anonymous fields of a tuple vs the same anonymous field.

struct Y(usize, usize);

fn distinct_variant() {
    let mut y = Y(1, 2);

    let a = match y {
        Y(ref mut a, _) => a
    };

    let b = match y {
        Y(_, ref mut b) => b
    };

    *a += 1;
    *b += 1;
}

fn same_variant() {
    let mut y = Y(1, 2);

    let a = match y {
        Y(ref mut a, _) => a
    };

    let b = match y {
        Y(ref mut b, _) => b //~ ERROR cannot borrow
    };

    *a += 1;
    *b += 1;
}

fn main() {
}
