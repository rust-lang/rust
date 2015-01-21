// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we generate obsolete syntax errors around usages of `proc`.

fn foo(p: proc()) { } //~ ERROR obsolete syntax: the `proc` type

fn bar() { proc() 1; } //~ ERROR obsolete syntax: `proc` expression

fn main() { }
