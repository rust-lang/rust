//@ compile-flags: -Z annotate-moves=100 -Copt-level=0 -g
// Test that custom size limits work correctly
#![crate_type = "lib"]

struct Struct99 {
    data: [u8; 99], // just below custom 100-byte threshold
}

const _: () = { assert!(size_of::<Struct99>() == 99) };

impl Clone for Struct99 {
    // CHECK-LABEL: <size_limit::Struct99 as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // Should NOT be annotated since 99 < 100
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ99_COPY_LOC:]]
        Struct99 { data: self.data }
    }
}

// CHECK-LABEL: size_limit::test_99_copy
pub fn test_99_copy() {
    let sz99 = Struct99 { data: [42; 99] };
    let _copy = sz99.clone();
}

// CHECK-LABEL: size_limit::test_99_move
pub fn test_99_move() {
    let sz99 = Struct99 { data: [42; 99] };
    // Should NOT be annotated
    // CHECK-NOT: compiler_move
    let _moved = sz99;
}

struct Struct100 {
    data: [u8; 100], // 100 bytes - equal to custom 100-byte threshold
}

const _: () = { assert!(size_of::<Struct100>() == 100) };

impl Clone for Struct100 {
    // CHECK-LABEL: <size_limit::Struct100 as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ100_COPY_LOC:]]
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ100_RETURN_LOC:]]
        Struct100 { data: self.data }
    }
}

// CHECK-LABEL: size_limit::test_100_copy
pub fn test_100_copy() {
    let sz100 = Struct100 { data: [42; 100] };
    let _copy = sz100.clone();
}

// CHECK-LABEL: size_limit::test_100_move
pub fn test_100_move() {
    let sz100 = Struct100 { data: [42; 100] };
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ100_MOVE_LOC:]]
    let _moved = sz100;
}

struct Struct101 {
    data: [u8; 101], // 101 bytes - above custom 100-byte threshold
}

const _: () = { assert!(size_of::<Struct101>() == 101) };

impl Clone for Struct101 {
    // CHECK-LABEL: <size_limit::Struct101 as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ101_COPY_LOC:]]
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ101_RETURN_LOC:]]
        Struct101 { data: self.data }
    }
}

// CHECK-LABEL: size_limit::test_101_copy
pub fn test_101_copy() {
    let sz101 = Struct101 { data: [42; 101] };
    let _copy = sz101.clone();
}

// CHECK-LABEL: size_limit::test_101_move
pub fn test_101_move() {
    let sz101 = Struct101 { data: [42; 101] };
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#SZ101_MOVE_LOC:]]
    let _moved = sz101;
}

// The scope for no-annotated is clone function itself
// CHECK-DAG: ![[#SZ99_COPY_LOC]] = !DILocation({{.*}}scope: ![[#SZ99_COPY_SCOPE:]]
// CHECK-DAG: ![[#SZ99_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "clone",

// Clone itself is copy, but return is move.
// CHECK-DAG: ![[#SZ100_COPY_LOC]] = !DILocation({{.*}}scope: ![[#SZ100_COPY_SCOPE:]]
// CHECK-DAG: ![[#SZ100_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<[u8; 100], 100>"
// CHECK-DAG: ![[#SZ100_RETURN_LOC]] = !DILocation({{.*}}scope: ![[#SZ100_RETURN_SCOPE:]]
// CHECK-DAG: ![[#SZ100_RETURN_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u8; 100], 100>"

// Assignment is move
// CHECK-DAG: ![[#SZ100_MOVE_LOC]] = !DILocation({{.*}}scope: ![[#SZ100_MOVE_SCOPE:]]
// CHECK-DAG: ![[#SZ100_MOVE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u8; 100], 100>"

// Clone itself is copy, but return is move.
// CHECK-DAG: ![[#SZ101_COPY_LOC]] = !DILocation({{.*}}scope: ![[#SZ101_COPY_SCOPE:]]
// CHECK-DAG: ![[#SZ101_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<[u8; 101], 101>"
// CHECK-DAG: ![[#SZ101_RETURN_LOC]] = !DILocation({{.*}}scope: ![[#SZ101_RETURN_SCOPE:]]
// CHECK-DAG: ![[#SZ101_RETURN_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u8; 101], 101>"

// Assignment is move
// CHECK-DAG: ![[#SZ101_MOVE_LOC]] = !DILocation({{.*}}scope: ![[#SZ101_MOVE_SCOPE:]]
// CHECK-DAG: ![[#SZ101_MOVE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u8; 101], 101>"
