//@ ignore-auxiliary (used by `./lint-pre-expansion-extern-module-levels.rs`)

pub fn try() {}

#[path = "lint_pre_expansion_extern_module_levels_nested_aux.rs"]
pub mod nested;
