// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `&mut T` implements `DerefMut<T>`

use std::ops::{Deref, DerefMut};

fn inc<T: Deref<Target=int> + DerefMut>(mut t: T) {
    *t += 1;
}

fn main() {
    let mut x: int = 5;
    inc(&mut x);
    assert_eq!(x, 6);
}
