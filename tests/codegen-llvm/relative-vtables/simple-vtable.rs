//@ compile-flags: -Zexperimental-relative-rust-abi-vtables=y

#![crate_type = "lib"]

// CHECK:      @vtable.0 = private {{.*}}constant [5 x i32] [
// CHECK-SAME:   i32 0,
// CHECK-SAME:   i32 4,
// CHECK-SAME:   i32 4,
// CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent [[STRUCT2_FOO:@".*Struct2\$.*foo.*"]] to i64), i64 ptrtoint (ptr @vtable.0 to i64)) to i32),
// CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent [[STRUCT2_BAR:@".*Struct2\$.*bar.*"]] to i64), i64 ptrtoint (ptr @vtable.0 to i64)) to i32)
// CHECK-SAME: ], align 4

// CHECK:      @vtable.1 = private {{.*}}constant [5 x i32] [
// CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent [[VT1_DROP_IN_PLACE:\@".*drop_in_place[^\"]*"]] to i64), i64 ptrtoint (ptr @vtable.1 to i64)) to i32),
// CHECK-SAME:   i32 24,
// CHECK-SAME:   i32 8,
// CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent [[STRUCT_FOO:@".*Struct\$.*foo.*"]] to i64), i64 ptrtoint (ptr @vtable.1 to i64)) to i32),
// CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent [[STRUCT_BAR:@".*Struct\$.*bar.*"]] to i64), i64 ptrtoint (ptr @vtable.1 to i64)) to i32)
// CHECK-SAME: ], align 4

// CHECK-DAG: define {{.*}}void [[STRUCT2_FOO]](ptr
// CHECK-DAG: define {{.*}}i32 [[STRUCT2_BAR]](ptr
// CHECK-DAG: define {{.*}}void [[STRUCT_FOO]](ptr
// CHECK-DAG: define {{.*}}i32 [[STRUCT_BAR]](ptr

trait MyTrait {
    fn foo(&self);
    fn bar(&self) -> u32;
}

struct Struct {
    s: String,
}

struct Struct2 {
    u: u32,
}

impl MyTrait for Struct {
    fn foo(&self) {
        println!("Struct foo {}", self.s);
    }
    fn bar(&self) -> u32 {
        42
    }
}

impl MyTrait for Struct2 {
    fn foo(&self) {
        println!("Struct2 foo {}", self.bar());
    }
    fn bar(&self) -> u32 {
        self.u
    }
}

/// This is only here to manifest the vtables.
pub fn create_struct(b: bool) -> Box<dyn MyTrait> {
    if b { Box::new(Struct { s: "abc".to_string() }) } else { Box::new(Struct2 { u: 1 }) }
}

// CHECK-DAG:   define void @_ZN13simple_vtable10invoke_foo{{[a-zA-Z0-9]*}}(
// CHECK-SAME:    ptr noundef nonnull align 1 [[DATA:%.*]],
// CHECK-SAME:    ptr noalias noundef readonly align 4 dereferenceable(20) [[VTABLE:%.*]]) {{.*}}{
// CHECK-NEXT:  start:
// CHECK-NEXT:    [[FUNC:%.*]] = tail call ptr @llvm.load.relative.i32(ptr nonnull [[VTABLE]], i32 12), !invariant.load
// CHECK-NEXT:    tail call void [[FUNC]](ptr noundef nonnull align 1 [[DATA]])
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
pub fn invoke_foo(x: &dyn MyTrait) {
    x.foo();
}

// CHECK-DAG:   define void @_ZN13simple_vtable11invoke_drop{{[a-zA-Z0-9]*}}(i1 noundef zeroext %b) {{.*}}{
// CHECK:         [[BOX:%.*]] = tail call { ptr, ptr } @_ZN13simple_vtable13create_struct{{.*}}(i1 noundef zeroext %b)
pub fn invoke_drop(b: bool) {
    let bx = create_struct(b);
    invoke_foo(&*bx);
}
