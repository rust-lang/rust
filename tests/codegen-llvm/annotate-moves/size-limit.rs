//@ compile-flags: -Z annotate-moves=100 -Copt-level=0 -g
// Test that custom size limits work correctly
#![crate_type = "lib"]

struct MediumStruct {
    data: [u64; 10], // 80 bytes - below custom 100-byte threshold
}

const _: () = { assert!(std::mem::size_of::<MediumStruct>() == 80) };

impl Clone for MediumStruct {
    // CHECK-LABEL: <size_limit::MediumStruct as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // Should NOT be annotated since 80 < 100
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#MEDIUM_COPY_LOC:]]
        MediumStruct { data: self.data }
    }
}

struct LargeStruct {
    data: [u64; 20], // 160 bytes - above custom 100-byte threshold
}

const _: () = { assert!(std::mem::size_of::<LargeStruct>() == 160) };

impl Clone for LargeStruct {
    // CHECK-LABEL: <size_limit::LargeStruct as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#LARGE_COPY_LOC:]]
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#LARGE_RETURN_LOC:]]
        LargeStruct { data: self.data }
    }
}

// CHECK-LABEL: size_limit::test_medium_copy
pub fn test_medium_copy() {
    let medium = MediumStruct { data: [42; 10] };
    let _copy = medium.clone();
}

// CHECK-LABEL: size_limit::test_large_copy
pub fn test_large_copy() {
    let large = LargeStruct { data: [42; 20] };
    let _copy = large.clone();
}

// CHECK-LABEL: size_limit::test_medium_move
pub fn test_medium_move() {
    let medium = MediumStruct { data: [42; 10] };
    // Should NOT be annotated
    // CHECK-NOT: compiler_move
    let _moved = medium;
}

// CHECK-LABEL: size_limit::test_large_move
pub fn test_large_move() {
    let large = LargeStruct { data: [42; 20] };
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#LARGE_MOVE_LOC:]]
    let _moved = large;
}

// The scope for no-annotated is clone function itself
// CHECK-DAG: ![[#MEDIUM_COPY_LOC]] = !DILocation({{.*}}scope: ![[#MEDIUM_COPY_SCOPE:]]
// CHECK-DAG: ![[#MEDIUM_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "clone",

// Clone itself is copy, but return is move.
// CHECK-DAG: ![[#LARGE_COPY_LOC]] = !DILocation({{.*}}scope: ![[#LARGE_COPY_SCOPE:]]
// CHECK-DAG: ![[#LARGE_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<[u64; 20], 160>"
// CHECK-DAG: ![[#LARGE_RETURN_LOC]] = !DILocation({{.*}}scope: ![[#LARGE_RETURN_SCOPE:]]
// CHECK-DAG: ![[#LARGE_RETURN_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// Assignment is move
// CHECK-DAG: ![[#LARGE_MOVE_LOC]] = !DILocation({{.*}}scope: ![[#LARGE_MOVE_SCOPE:]]
// CHECK-DAG: ![[#LARGE_MOVE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"
