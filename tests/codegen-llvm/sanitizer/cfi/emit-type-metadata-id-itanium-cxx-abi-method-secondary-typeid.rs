// Verifies that a secondary type metadata identifier is assigned to methods with their concrete
// self so they can be used as function pointers.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

trait Trait1 {
    fn foo(&self);
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {}
    // CHECK: define{{.*}}3foo{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u6regionEEE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type1EE"}
