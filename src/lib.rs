#![feature(globs, phase, plugin_registrar)]

#![allow(unused_imports)]

#[phase(plugin, link)]
extern crate syntax;
#[phase(plugin, link)]
extern crate rustc;

// Only for the compile time checking of paths
extern crate collections;

use rustc::plugin::Registry;
use rustc::lint::LintPassObject;

pub mod types;
pub mod misc;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_lint_pass(box types::TypePass as LintPassObject);
    reg.register_lint_pass(box misc::MiscPass as LintPassObject);
    reg.register_lint_pass(box misc::StrToStringPass as LintPassObject);
    reg.register_lint_pass(box misc::TopLevelRefPass as LintPassObject);
    reg.register_lint_group("clippy", vec![types::CLIPPY_BOX_VEC, types::CLIPPY_DLIST,
                                           misc::CLIPPY_SINGLE_MATCH, misc::CLIPPY_STR_TO_STRING,
                                           misc::CLIPPY_TOPLEVEL_REF_ARG]);
}
