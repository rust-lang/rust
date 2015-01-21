// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[no_link]
extern crate libc;

fn main() {
    unsafe {
        libc::abs(0);  //~ ERROR Use of undeclared type or module `libc`
                      //~^ ERROR unresolved name `libc::abs`
    }
}
