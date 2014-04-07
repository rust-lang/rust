// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test

// error-pattern:library 'm' already added: can't specify link_args.

/* I think it should undefined to have multiple modules that link in the same
  library, but provide different link arguments. Unfortunately we don't track
  link_args by module -- they are just appended as discovered into the crate
  store -- but for now, it should be an error to provide link_args on a module
  that's already been included (with or without link_args). */

#[link_name= "m"]
#[link_args="-foo"]             // this could have been elided.
extern {
}

#[link_name= "m"]
#[link_args="-bar"]             // this is the actual error trigger.
extern {
}
