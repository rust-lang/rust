// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let mut v = vec!(1is);
    let mut f = |&mut:| v.push(2is);
    let _w = v; //~ ERROR: cannot move out of `v`

    f();
}
