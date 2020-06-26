// compile-flags: -g -Z src-hash-algorithm=sha1

#![crate_type = "lib"]

pub fn test() {}
// CHECK: checksumkind: CSK_SHA1
