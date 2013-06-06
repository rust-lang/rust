// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Top-level module for network-related functionality.

Basically, including this module gives you:

* `net_tcp`
* `net_ip`
* `net_url`

See each of those three modules for documentation on what they do.
*/

pub use tcp = net_tcp;
pub use ip = net_ip;
pub use url = net_url;
