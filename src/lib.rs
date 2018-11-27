// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:cargo-clippy
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![allow(clippy::missing_docs_in_private_items)]
#![warn(rust_2018_idioms)]

// FIXME: switch to something more ergonomic here, once available.
// (currently there is no way to opt into sysroot crates w/o `extern crate`)
#[allow(unused_extern_crates)]
extern crate rustc_plugin;
use self::rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry<'_>) {
    reg.sess.lint_store.with_read_lock(|lint_store| {
        for (lint, _, _) in lint_store.get_lint_groups() {
            reg.sess
                .struct_warn(
                    "the clippy plugin is being deprecated, please use cargo clippy or rls with the clippy feature",
                )
                .emit();
            if lint == "clippy" {
                // cargo clippy run on a crate that also uses the plugin
                return;
            }
        }
    });

    let conf = clippy_lints::read_conf(reg);
    clippy_lints::register_plugins(reg, &conf);
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
