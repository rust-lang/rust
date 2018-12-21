// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-D bogus -D dead_cod

// error-pattern:unknown lint: `bogus`
// error-pattern:requested on the command line with `-D bogus`
// error-pattern:unknown lint: `dead_cod`
// error-pattern:requested on the command line with `-D dead_cod`
// error-pattern:did you mean: `dead_code`

fn main() { }
