// Check that invalid --check-cfg are rejected
//
// check-fail
// revisions: anything_else names_simple_ident values_simple_ident values_string_literals
// [anything_else]compile-flags: -Z unstable-options --check-cfg=anything_else(...)
// [names_simple_ident]compile-flags: -Z unstable-options --check-cfg=names("NOT_IDENT")
// [values_simple_ident]compile-flags: -Z unstable-options --check-cfg=values("NOT_IDENT")
// [values_string_literals]compile-flags: -Z unstable-options --check-cfg=values(test,12)

fn main() {}
