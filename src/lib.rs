// error-pattern:cargo-clippy
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![feature(macro_vis_matcher)]
#![allow(unknown_lints)]
#![allow(missing_docs_in_private_items)]

extern crate rustc_plugin;
use rustc_plugin::Registry;

extern crate clippy_lints;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.sess.lint_store.with_read_lock(|lint_store| {
        for (lint, _, _) in lint_store.get_lint_groups() {
            if lint == "clippy" {
                reg.sess
                    .struct_warn("running cargo clippy on a crate that also imports the clippy plugin")
                    .emit();
                return;
            }
        }
    });

    clippy_lints::register_plugins(reg);
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
