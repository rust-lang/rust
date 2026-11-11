//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-pretty-attr.pp

#[repr(C, packed(4))]
#[repr(transparent)]
struct Example {}
