// compile-flags: -g -Z src-hash-algorithm=sha256

#![crate_type = "lib"]

pub fn test() {}
// CHECK: checksumkind: CSK_SHA256
