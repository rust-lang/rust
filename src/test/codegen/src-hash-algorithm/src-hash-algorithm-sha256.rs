// compile-flags: -g -Z src-hash-algorithm=sha256
// min-llvm-version: 11.0

#![crate_type = "lib"]

pub fn test() {}
// CHECK: checksumkind: CSK_SHA256
