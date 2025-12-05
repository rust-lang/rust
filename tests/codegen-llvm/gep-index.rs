//! Check that index and offset use the same getelementptr format.

//@ revisions: NO-OPT OPT
//@[NO-OPT] compile-flags: -Copt-level=0
//@[OPT] compile-flags: -Copt-level=1

#![crate_type = "lib"]

struct Foo(i32, i32);

// CHECK-LABEL: @index_on_struct(
#[no_mangle]
fn index_on_struct(a: &[Foo], index: usize) -> &Foo {
    // CHECK: getelementptr inbounds{{( nuw)?}} %Foo, ptr %a.0, {{i64|i32}} %index
    &a[index]
}

// CHECK-LABEL: @offset_on_struct(
#[no_mangle]
fn offset_on_struct(a: *const Foo, index: usize) -> *const Foo {
    // CHECK: getelementptr inbounds{{( nuw)?}} %Foo, ptr %a, {{i64|i32}} %index
    unsafe { a.add(index) }
}

// CHECK-LABEL: @index_on_i32(
#[no_mangle]
fn index_on_i32(a: &[i32], index: usize) -> &i32 {
    // CHECK: getelementptr inbounds{{( nuw)?}} i32, ptr %a.0, {{i64|i32}} %index
    &a[index]
}

// CHECK-LABEL: @offset_on_i32(
#[no_mangle]
fn offset_on_i32(a: *const i32, index: usize) -> *const i32 {
    // CHECK: getelementptr inbounds{{( nuw)?}} i32, ptr %a, {{i64|i32}} %index
    unsafe { a.add(index) }
}
