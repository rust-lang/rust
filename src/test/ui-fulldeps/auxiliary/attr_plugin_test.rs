// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar)]
#![feature(rustc_private)]

extern crate syntax;

extern crate rustc;
extern crate rustc_plugin;

use syntax::feature_gate::AttributeType;
use rustc_plugin::Registry;



#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_attribute("foo".to_owned(), AttributeType::Normal);
    reg.register_attribute("bar".to_owned(), AttributeType::CrateLevel);
    reg.register_attribute("baz".to_owned(), AttributeType::Whitelisted);
}
