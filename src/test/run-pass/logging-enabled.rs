// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_LOG=logging-enabled=info

#[macro_use]
extern crate log;

pub fn main() {
    if log_enabled!(log::DEBUG) {
        panic!("what?! debugging?");
    }
    if !log_enabled!(log::INFO) {
        panic!("what?! no info?");
    }
}
