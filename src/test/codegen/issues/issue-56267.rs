// compile-flags: -C no-prepopulate-passes

#![crate_type="rlib"]

#[allow(dead_code)]
pub struct Foo<T> {
    foo: u64,
    bar: T,
}

// The store writing to bar.1 should have alignment 4. Not checking
// other stores here, as the alignment will be platform-dependent.

// CHECK: store i32 [[TMP1:%.+]], i32* [[TMP2:%.+]], align 4
#[no_mangle]
pub fn test(x: (i32, i32)) -> Foo<(i32, i32)> {
    Foo { foo: 0, bar: x }
}
