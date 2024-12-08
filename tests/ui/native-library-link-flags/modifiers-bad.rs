//@ edition: 2021
//@ revisions: blank no-prefix prefix-only unknown

//@[blank] compile-flags: -l static:=foo
//@[no-prefix] compile-flags: -l static:bundle=foo
//@[prefix-only] compile-flags: -l static:+=foo
//@[unknown] compile-flags: -l static:+ferris=foo

// Tests various illegal values for the "modifier" part of an `-l` flag.

fn main() {}
