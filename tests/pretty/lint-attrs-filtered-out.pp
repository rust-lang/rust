extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:lint-attrs-filtered-out.pp

// `does_not_exist` is filtered out on the hir
#[attr = LintCheck([LintCheck {lint_name_normalized: LintName { tool: None, name: "dead_code" },
kind: Deny}])]
fn main() { }
