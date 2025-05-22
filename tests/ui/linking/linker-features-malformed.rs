//! Check that malformed `-Zlinker-features` flags are properly rejected.

//@ revisions: no_value
//@[no_value] compile-flags: -Zlinker-features=
//[no_value]~? ERROR incorrect value `` for unstable option `linker-features`

//@ revisions: invalid_modifier
//@[invalid_modifier] compile-flags: -Zlinker-features=*lld
//[invalid_modifier]~? ERROR incorrect value `*lld` for unstable option `linker-features`

//@ revisions: unknown_value
//@[unknown_value] compile-flags: -Zlinker-features=unknown
//[unknown_value]~? ERROR incorrect value `unknown` for unstable option `linker-features`

//@ revisions: unknown_modifier_value
//@[unknown_modifier_value] compile-flags: -Zlinker-features=-unknown
//[unknown_modifier_value]~? ERROR incorrect value `-unknown` for unstable option `linker-features`

//@ revisions: unknown_boolean
//@[unknown_boolean] compile-flags: -Zlinker-features=maybe
//[unknown_boolean]~? ERROR incorrect value `maybe` for unstable option `linker-features`

//@ revisions: invalid_separator
//@[invalid_separator] compile-flags: -Zlinker-features=-lld@+lld
//[invalid_separator]~? ERROR incorrect value `-lld@+lld` for unstable option `linker-features`

fn main() {}
