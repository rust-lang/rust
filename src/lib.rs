// error-pattern:cargo-clippy
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![allow(unknown_lints)]
#![feature(borrow_state)]

extern crate rustc_plugin;
use rustc_plugin::Registry;

extern crate clippy_lints;

pub use clippy_lints::*;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    if reg.sess.lint_store.borrow_state() == std::cell::BorrowState::Unused && reg.sess.lint_store.borrow().get_lint_groups().iter().any(|&(s, _, _)| s == "clippy") {
        reg.sess.struct_warn("running cargo clippy on a crate that also imports the clippy plugin").emit();
    } else {
        register_plugins(reg);
    }
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code, print_stdout)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
