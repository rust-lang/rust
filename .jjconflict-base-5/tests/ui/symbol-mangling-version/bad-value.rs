//@ revisions: no-value blank bad
//@ [no-value] compile-flags: -Csymbol-mangling-version
//@ [blank] compile-flags: -Csymbol-mangling-version=
//@ [bad] compile-flags: -Csymbol-mangling-version=bad-value

fn main() {}

//[no-value]~? ERROR codegen option `symbol-mangling-version` requires one of
//[blank]~? ERROR incorrect value `` for codegen option `symbol-mangling-version`
//[bad]~? ERROR incorrect value `bad-value` for codegen option `symbol-mangling-version`
