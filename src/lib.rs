// error-pattern:cargo-clippy
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![warn(rust_2018_idioms)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
#[allow(unused_extern_crates)]
extern crate rustc_driver;
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
