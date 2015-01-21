// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_LOG=conditional-debug-macro-on=4

pub fn main() {
    // exits early if println! evaluates its arguments, otherwise it
    // will hit the panic.
    println!("{:?}", { if true { return; } });

    panic!();
}
