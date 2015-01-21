// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let mut x: Option<isize> = None;
    match x {
      None => {
          // Note: on this branch, no borrow has occurred.
          x = Some(0);
      }
      Some(ref i) => {
          // But on this branch, `i` is an outstanding borrow
          x = Some(*i+1); //~ ERROR cannot assign to `x`
      }
    }
    x.clone(); // just to prevent liveness warnings
}
