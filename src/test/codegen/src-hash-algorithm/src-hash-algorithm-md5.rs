// compile-flags: -g -Z src-hash-algorithm=md5

#![crate_type = "lib"]

pub fn test() {}
// CHECK: checksumkind: CSK_MD5
