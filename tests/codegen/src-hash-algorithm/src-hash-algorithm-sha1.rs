//@ compile-flags: -g -Z src-hash-algorithm=sha1 -Copt-level=0

#![crate_type = "lib"]

pub fn test() {}
// CHECK: checksumkind: CSK_SHA1
