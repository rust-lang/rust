//! Check that malformed `-Clinker-features` flags are properly rejected.

//@ revisions: no_value
//@[no_value] compile-flags: -Clinker-features=
//[no_value]~? ERROR incorrect value `` for codegen option `linker-features`

//@ revisions: invalid_modifier
//@[invalid_modifier] compile-flags: -Clinker-features=*lld
//[invalid_modifier]~? ERROR incorrect value `*lld` for codegen option `linker-features`

//@ revisions: unknown_value
//@[unknown_value] compile-flags: -Clinker-features=unknown
//[unknown_value]~? ERROR incorrect value `unknown` for codegen option `linker-features`

//@ revisions: unknown_modifier_value
//@[unknown_modifier_value] compile-flags: -Clinker-features=-unknown
//[unknown_modifier_value]~? ERROR incorrect value `-unknown` for codegen option `linker-features`

//@ revisions: unknown_boolean
//@[unknown_boolean] compile-flags: -Clinker-features=maybe
//[unknown_boolean]~? ERROR incorrect value `maybe` for codegen option `linker-features`

//@ revisions: invalid_separator
//@[invalid_separator] compile-flags: -Clinker-features=-lld@+lld
//[invalid_separator]~? ERROR incorrect value `-lld@+lld` for codegen option `linker-features`

fn main() {}
