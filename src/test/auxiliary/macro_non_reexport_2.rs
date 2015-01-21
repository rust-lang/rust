// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "dylib"]

// Since we load a serialized macro with all its attributes, accidentally
// re-exporting a `#[macro_export] macro_rules!` is something of a concern!
//
// We avoid it at the moment only because of the order in which we do things.

#[macro_use] #[no_link]
extern crate macro_reexport_1;
