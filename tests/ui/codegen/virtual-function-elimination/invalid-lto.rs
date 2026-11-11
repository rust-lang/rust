//@ revisions: lto_missing lto_no lto_thin
//@[lto_missing] compile-flags: -Zvirtual-function-elimination
//@[lto_no] compile-flags: -Zvirtual-function-elimination -Clto=no
//@[lto_thin] compile-flags: -Zvirtual-function-elimination -Clto=thin
//@ no-prefer-dynamic
//@ ignore-backends: gcc
#![crate_type = "staticlib"]

//~? ERROR `-Zvirtual-function-elimination` requires `-Clto=fat`
