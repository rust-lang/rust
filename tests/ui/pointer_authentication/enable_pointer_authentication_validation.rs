//@ revisions: empty unprefixed all_unknown all_known mixed

//@[empty] compile-flags: -Zpointer-authentication=
//@[unprefixed] compile-flags: -Zpointer-authentication=auth-traps
//@[all_unknown] compile-flags: -Zpointer-authentication=+I,+do,-not,-exist
//@[all_known] check-pass
//@[all_known] compile-flags: -Zpointer-authentication=+elf-got,-init-fini
//@[mixed] compile-flags: -Zpointer-authentication=+elf-got,-imaginary

fn main() {}

//[empty]~? ERROR incorrect value `` for unstable option `pointer-authentication`
//[unprefixed]~? ERROR incorrect value `auth-traps` for unstable option `pointer-authentication`
//[all_unknown]~? ERROR incorrect value `+I,+do,-not,-exist` for unstable option `pointer-authentication`
//[mixed]~? ERROR incorrect value `+elf-got,-imaginary` for unstable option `pointer-authentication`
