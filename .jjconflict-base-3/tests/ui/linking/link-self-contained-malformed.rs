//! Check that malformed `-Clink-self-contained` invocations are properly rejected.

//@ revisions: no_value
//@[no_value] compile-flags: -Clink-self-contained=
//[no_value]~? ERROR incorrect value `` for codegen option `link-self-contained`

//@ revisions: invalid_modifier
//@[invalid_modifier] compile-flags: -Clink-self-contained=*lld
//[invalid_modifier]~? ERROR incorrect value `*lld` for codegen option `link-self-contained`

//@ revisions: unknown_value
//@[unknown_value] compile-flags: -Clink-self-contained=unknown
//[unknown_value]~? ERROR incorrect value `unknown` for codegen option `link-self-contained`

//@ revisions: unknown_modifier_value
//@[unknown_modifier_value] compile-flags: -Clink-self-contained=-unknown
//[unknown_modifier_value]~? ERROR incorrect value `-unknown` for codegen option `link-self-contained`

//@ revisions: unknown_boolean
//@[unknown_boolean] compile-flags: -Clink-self-contained=maybe
//[unknown_boolean]~? ERROR incorrect value `maybe` for codegen option `link-self-contained`

fn main() {}
