// error-pattern:cargo-clippy
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![warn(rust_2018_idioms)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
#[allow(unused_extern_crates)]
extern crate rustc_driver;
use self::rustc_driver::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry<'_>) {
    for (lint, _, _) in reg.lint_store.get_lint_groups() {
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

    let conf = clippy_lints::read_conf(reg.args(), &reg.sess);
    clippy_lints::register_plugins(&mut reg.lint_store, &reg.sess, &conf);
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
