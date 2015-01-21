// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Obj<F> where F: FnMut() -> u32 {
    closure: F,
}

fn main() {
    let o = Obj { closure: || 42 };
    o.closure(); //~ ERROR does not implement any method in scope named `closure`
    //~^ NOTE use `(s.closure)(...)` if you meant to call the function stored in the `closure` field
}
