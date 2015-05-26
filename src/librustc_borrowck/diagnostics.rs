// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_long_diagnostics! {

E0381: r##"
It is not allowed to use or capture an uninitialized variable. For example:

```
fn main() {
    let x: i32;
    let y = x; // error, use of possibly uninitialized variable
```

To fix this, ensure that any declared variables are initialized before being
used.
"##

}

register_diagnostics! {
    E0373, // closure may outlive current fn, but it borrows {}, which is owned by current fn
    E0382, // use of partially/collaterally moved value
    E0383, // partial reinitialization of uninitialized structure
    E0384, // reassignment of immutable variable
    E0385, // {} in an aliasable location
    E0386, // {} in an immutable container
    E0387, // {} in a captured outer variable in an `Fn` closure
    E0388, // {} in a static location
    E0389  // {} in a `&` reference
}
