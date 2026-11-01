//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:lint-attrs-filtered-out.pp

// `does_not_exist` is filtered out on the hir
#[deny(dead_code, does_not_exist)]
fn main() {}
