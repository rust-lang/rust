//@ compile-flags: -Z annotate-moves=1 -Copt-level=0 -g

#![crate_type = "lib"]

// Test with large array (non-struct type, Copy)
type LargeArray = [u64; 20]; // 160 bytes

#[derive(Clone, Default)]
struct NonCopyU64(u64);

// Test with Copy implementation
#[derive(Copy)]
struct ExplicitCopy {
    data: [u64; 20], // 160 bytes
}

impl Clone for ExplicitCopy {
    // CHECK-LABEL: <integration::ExplicitCopy as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#EXPLICIT_COPY_LOC:]]
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#EXPLICIT_RETURN_LOC:]]
        Self { data: self.data }
    }
}

// Test with hand-implemented Clone (non-Copy)
struct NonCopyStruct {
    data: [u64; 20], // 160 bytes
}

impl Clone for NonCopyStruct {
    // CHECK-LABEL: <integration::NonCopyStruct as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CLONE_COPY_LOC:]]
        // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CLONE_RETURN_LOC:]]
        NonCopyStruct { data: self.data }
    }
}

// CHECK-LABEL: integration::test_pure_assignment_move
pub fn test_pure_assignment_move() {
    let arr: LargeArray = [42; 20];
    // Arrays are initialized with a loop
    // CHECK-NOT: call void @llvm.memcpy{{.*}}, !dbg ![[#]]
    let _moved = arr;
}

// CHECK-LABEL: integration::test_pure_assignment_copy
pub fn test_pure_assignment_copy() {
    let s = ExplicitCopy { data: [42; 20] };
    // Arrays are initialized with a loop
    // CHECK-NOT: call void @llvm.memcpy{{.*}}, !dbg ![[#]]
    let _copied = s;
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#ASSIGN_COPY2_LOC:]]
    let _copied_2 = s;
}

#[derive(Default)]
struct InitializeStruct {
    field1: String,
    field2: String,
    field3: String,
}

// CHECK-LABEL: integration::test_init_struct
pub fn test_init_struct() {
    let mut s = InitializeStruct::default();

    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#INIT_STRUCT_LOC:]]
    s = InitializeStruct {
        field1: String::from("Hello"),
        field2: String::from("from"),
        field3: String::from("Rust"),
    };
}

// CHECK-LABEL: integration::test_tuple_of_scalars
pub fn test_tuple_of_scalars() {
    // Tuple of scalars (even if large) may use scalar-pair repr, so may not be annotated
    let t: (u64, u64, u64, u64) = (1, 2, 3, 4); // 32 bytes
    // Copied with explicit stores
    // CHECK-NOT: call void @llvm.memcpy{{.*}}, !dbg ![[#]]
    let _moved = t;
}

// CHECK-LABEL: integration::test_tuple_of_structs
pub fn test_tuple_of_structs() {
    let s1 = NonCopyStruct { data: [1; 20] };
    let s2 = NonCopyStruct { data: [2; 20] };
    let tuple = (s1, s2); // Large tuple containing structs (320 bytes)
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#TUPLE_MOVE_LOC:]]
    let _moved = tuple;
}

// CHECK-LABEL: integration::test_tuple_mixed
pub fn test_tuple_mixed() {
    let s = NonCopyStruct { data: [1; 20] };
    let tuple = (42u64, s); // Mixed tuple (168 bytes: 8 for u64 + 160 for struct)
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#MIXED_TUPLE_LOC:]]
    let _moved = tuple;
}

// CHECK-LABEL: integration::test_explicit_copy_assignment
pub fn test_explicit_copy_assignment() {
    let c1 = ExplicitCopy { data: [1; 20] };
    // Initialized with loop
    // CHECK-NOT: call void @llvm.memcpy{{.*}}, !dbg ![[#]]
    let c2 = c1;
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#COPY2_LOC:]]
    let _c3 = c1; // Can still use c1 (it was copied)
    let _ = c2;
}

// CHECK-LABEL: integration::test_array_move
pub fn test_array_move() {
    let arr: [String; 20] = std::array::from_fn(|i| i.to_string());

    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#ARRAY_MOVE_LOC:]]
    let _moved = arr;
}

// CHECK-LABEL: integration::test_array_in_struct_field
pub fn test_array_in_struct_field() {
    let s = NonCopyStruct { data: [1; 20] };
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#FIELD_MOVE_LOC:]]
    let data = s.data; // Move array field out of struct
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#FIELD_MOVE2_LOC:]]
    let _moved = data;
}

// CHECK-LABEL: integration::test_clone_noncopy
pub fn test_clone_noncopy() {
    let s = NonCopyStruct { data: [1; 20] };
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CALL_CLONE_NONCOPY_LOC:]]
    let _cloned = s.clone(); // The copy happens inside the clone() impl above
}

// CHECK-LABEL: integration::test_clone_explicit_copy
pub fn test_clone_explicit_copy() {
    let c = ExplicitCopy { data: [1; 20] };
    // Derived Clone on Copy type - the copy happens inside the generated clone impl
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#CALL_CLONE_COPY_LOC:]]
    let _cloned = c.clone();
}

// CHECK-LABEL: integration::test_copy_ref
pub fn test_copy_ref(x: &ExplicitCopy) {
    // CHECK: call void @llvm.memcpy{{.*}}, !dbg ![[#LOCAL_COPY_LOC:]]
    let _local = *x;
}

// CHECK-DAG: ![[#EXPLICIT_COPY_LOC]] = !DILocation({{.*}}scope: ![[#EXPLICIT_COPY_SCOPE:]]
// CHECK-DAG: ![[#EXPLICIT_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<[u64; 20], 160>"
// CHECK-DAG: ![[#EXPLICIT_RETURN_LOC]] = !DILocation({{.*}}scope: ![[#EXPLICIT_RETURN_SCOPE:]]
// CHECK-DAG: ![[#EXPLICIT_RETURN_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#CLONE_COPY_LOC]] = !DILocation({{.*}}scope: ![[#CLONE_COPY_SCOPE:]]
// CHECK-DAG: ![[#CLONE_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<[u64; 20], 160>"
// CHECK-DAG: ![[#CLONE_RETURN_LOC]] = !DILocation({{.*}}scope: ![[#CLONE_RETURN_SCOPE:]]
// CHECK-DAG: ![[#CLONE_RETURN_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#ASSIGN_COPY2_LOC]] = !DILocation({{.*}}scope: ![[#ASSIGN_COPY2_SCOPE:]]
// CHECK-DAG: ![[#ASSIGN_COPY2_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#INIT_STRUCT_LOC]] = !DILocation({{.*}}scope: ![[#INIT_STRUCT_SCOPE:]]
// CHECK-DAG: ![[#INIT_STRUCT_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<alloc::string::String, [[#]]>"

// CHECK-DAG: ![[#TUPLE_MOVE_LOC]] = !DILocation({{.*}}scope: ![[#TUPLE_MOVE_SCOPE:]]
// CHECK-DAG: ![[#TUPLE_MOVE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#MIXED_TUPLE_LOC]] = !DILocation({{.*}}scope: ![[#MIXED_TUPLE_SCOPE:]]
// CHECK-DAG: ![[#MIXED_TUPLE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#COPY2_LOC]] = !DILocation({{.*}}scope: ![[#COPY2_SCOPE:]]
// CHECK-DAG: ![[#COPY2_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#ARRAY_MOVE_LOC]] = !DILocation({{.*}}scope: ![[#ARRAY_MOVE_SCOPE:]]
// CHECK-DAG: ![[#ARRAY_MOVE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[alloc::string::String; 20], [[#]]>"

// CHECK-DAG: ![[#FIELD_MOVE_LOC]] = !DILocation({{.*}}scope: ![[#FIELD_MOVE_SCOPE:]]
// CHECK-DAG: ![[#FIELD_MOVE_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"
// CHECK-DAG: ![[#FIELD_MOVE2_LOC]] = !DILocation({{.*}}scope: ![[#FIELD_MOVE2_SCOPE:]]
// CHECK-DAG: ![[#FIELD_MOVE2_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<[u64; 20], 160>"

// CHECK-DAG: ![[#CALL_CLONE_NONCOPY_LOC]] = !DILocation({{.*}}scope: ![[#CALL_CLONE_NONCOPY_SCOPE:]]
// CHECK-DAG: ![[#CALL_CLONE_NONCOPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#CALL_CLONE_COPY_LOC]] = !DILocation({{.*}}scope: ![[#CALL_CLONE_COPY_SCOPE:]]
// CHECK-DAG: ![[#CALL_CLONE_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_move<[u64; 20], 160>"

// CHECK-DAG: ![[#LOCAL_COPY_LOC]] = !DILocation({{.*}}scope: ![[#LOCAL_COPY_SCOPE:]]
// CHECK-DAG: ![[#LOCAL_COPY_SCOPE]] = {{(distinct )?}}!DISubprogram(name: "compiler_copy<integration::ExplicitCopy, 160>"
