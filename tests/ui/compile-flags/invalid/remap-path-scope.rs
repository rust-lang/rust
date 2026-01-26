// Error on invalid --remap-path-scope arguments

//@ revisions: foo underscore
//@ [foo]compile-flags: --remap-path-scope=foo
//@ [underscore]compile-flags: --remap-path-scope=macro_object

//~? ERROR argument for `--remap-path-scope

fn main() {}
