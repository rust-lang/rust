// Error on invalid --remap-path-scope arguments

//@ revisions: foo underscore
//@ compile-flags: -Zunstable-options
//@ [foo]compile-flags: --remap-path-scope=foo
//@ [underscore]compile-flags: --remap-path-scope=macro_object

//~? RAW argument for `--remap-path-scope

fn main() {}
