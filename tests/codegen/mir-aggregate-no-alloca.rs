//@ compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

#[repr(transparent)]
struct Transparent32(u32);

// CHECK: i32 @make_transparent(i32 noundef %x)
#[no_mangle]
pub fn make_transparent(x: u32) -> Transparent32 {
    // CHECK: %a = alloca i32
    // CHECK: store i32 %x, ptr %a
    // CHECK: %[[TEMP:.+]] = load i32, ptr %a
    // CHECK: ret i32 %[[TEMP]]
    let a = Transparent32(x);
    a
}

// CHECK: i32 @make_closure(i32 noundef %x)
#[no_mangle]
pub fn make_closure(x: i32) -> impl Fn(i32) -> i32 {
    // CHECK: %[[ALLOCA:.+]] = alloca i32
    // CHECK: store i32 %x, ptr %[[ALLOCA]]
    // CHECK: %[[TEMP:.+]] = load i32, ptr %[[ALLOCA]]
    // CHECK: ret i32 %[[TEMP]]
    move |y| x + y
}

// CHECK-LABEL: { i32, i32 } @make_2_tuple(i32 noundef %x)
#[no_mangle]
pub fn make_2_tuple(x: u32) -> (u32, u32) {
    // CHECK: %pair = alloca { i32, i32 }
    // CHECK: store i32
    // CHECK: store i32
    // CHECK: load i32
    // CHECK: load i32
    let pair = (x, x);
    pair
}
