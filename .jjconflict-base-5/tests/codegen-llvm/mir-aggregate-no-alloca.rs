// 32-bit systems will return 128bit values using a return area pointer.
//@ revisions: bit32 bit64
//@[bit32] only-32bit
//@[bit64] only-64bit
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z randomize-layout=no

#![crate_type = "lib"]

#[repr(transparent)]
pub struct Transparent32(u32);

// CHECK: i32 @make_transparent(i32{{.*}} %x)
#[no_mangle]
pub fn make_transparent(x: u32) -> Transparent32 {
    // CHECK-NOT: alloca
    // CHECK: ret i32 %x
    let a = Transparent32(x);
    a
}

// CHECK: i32 @make_closure(i32{{.*}} %x)
#[no_mangle]
pub fn make_closure(x: i32) -> impl Fn(i32) -> i32 {
    // CHECK-NOT: alloca
    // CHECK: ret i32 %x
    move |y| x + y
}

#[repr(transparent)]
pub struct TransparentPair((), (u16, u16), ());

// CHECK: { i16, i16 } @make_transparent_pair(i16 noundef %x.0, i16 noundef %x.1)
#[no_mangle]
pub fn make_transparent_pair(x: (u16, u16)) -> TransparentPair {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP0:.+]] = insertvalue { i16, i16 } poison, i16 %x.0, 0
    // CHECK: %[[TEMP1:.+]] = insertvalue { i16, i16 } %[[TEMP0]], i16 %x.1, 1
    // CHECK: ret { i16, i16 } %[[TEMP1]]
    let a = TransparentPair((), x, ());
    a
}

// CHECK-LABEL: { i32, i32 } @make_2_tuple(i32{{.*}} %x)
#[no_mangle]
pub fn make_2_tuple(x: u32) -> (u32, u32) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP0:.+]] = insertvalue { i32, i32 } poison, i32 %x, 0
    // CHECK: %[[TEMP1:.+]] = insertvalue { i32, i32 } %[[TEMP0]], i32 %x, 1
    // CHECK: ret { i32, i32 } %[[TEMP1]]
    let pair = (x, x);
    pair
}

// CHECK-LABEL: i8 @make_cell_of_bool(i1 noundef zeroext %b)
#[no_mangle]
pub fn make_cell_of_bool(b: bool) -> std::cell::Cell<bool> {
    // CHECK: %[[BYTE:.+]] = zext i1 %b to i8
    // CHECK: ret i8 %[[BYTE]]
    std::cell::Cell::new(b)
}

// CHECK-LABEL: { i8, i16 } @make_cell_of_bool_and_short(i1 noundef zeroext %b, i16{{.*}} %s)
#[no_mangle]
pub fn make_cell_of_bool_and_short(b: bool, s: u16) -> std::cell::Cell<(bool, u16)> {
    // CHECK-NOT: alloca
    // CHECK: %[[BYTE:.+]] = zext i1 %b to i8
    // CHECK: %[[TEMP0:.+]] = insertvalue { i8, i16 } poison, i8 %[[BYTE]], 0
    // CHECK: %[[TEMP1:.+]] = insertvalue { i8, i16 } %[[TEMP0]], i16 %s, 1
    // CHECK: ret { i8, i16 } %[[TEMP1]]
    std::cell::Cell::new((b, s))
}

// CHECK-LABEL: { i1, i1 } @make_tuple_of_bools(i1 noundef zeroext %a, i1 noundef zeroext %b)
#[no_mangle]
pub fn make_tuple_of_bools(a: bool, b: bool) -> (bool, bool) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP0:.+]] = insertvalue { i1, i1 } poison, i1 %a, 0
    // CHECK: %[[TEMP1:.+]] = insertvalue { i1, i1 } %[[TEMP0]], i1 %b, 1
    // CHECK: ret { i1, i1 } %[[TEMP1]]
    (a, b)
}

pub struct Struct0();

// CHECK-LABEL: void @make_struct_0()
#[no_mangle]
pub fn make_struct_0() -> Struct0 {
    // CHECK: ret void
    let s = Struct0();
    s
}

pub struct Struct1(i32);

// CHECK-LABEL: i32 @make_struct_1(i32{{.*}} %a)
#[no_mangle]
pub fn make_struct_1(a: i32) -> Struct1 {
    // CHECK: ret i32 %a
    let s = Struct1(a);
    s
}

pub struct Struct2Asc(i16, i64);

// bit32-LABEL: void @make_struct_2_asc({{.*}} sret({{[^,]*}}) {{.*}} %s,
// bit64-LABEL: { i64, i16 } @make_struct_2_asc(
// CHECK-SAME: i16{{.*}} %a, i64 noundef %b)
#[no_mangle]
pub fn make_struct_2_asc(a: i16, b: i64) -> Struct2Asc {
    // CHECK-NOT: alloca
    // bit32: %[[GEP:.+]] = getelementptr inbounds i8, ptr %s, i32 8
    // bit32: store i16 %a, ptr %[[GEP]]
    // bit32: store i64 %b, ptr %s
    // bit64: %[[TEMP0:.+]] = insertvalue { i64, i16 } poison, i64 %b, 0
    // bit64: %[[TEMP1:.+]] = insertvalue { i64, i16 } %[[TEMP0]], i16 %a, 1
    // bit64: ret { i64, i16 } %[[TEMP1]]
    let s = Struct2Asc(a, b);
    s
}

pub struct Struct2Desc(i64, i16);

// bit32-LABEL: void @make_struct_2_desc({{.*}} sret({{[^,]*}}) {{.*}} %s,
// bit64-LABEL: { i64, i16 } @make_struct_2_desc(
// CHECK-SAME: i64 noundef %a, i16{{.*}} %b)
#[no_mangle]
pub fn make_struct_2_desc(a: i64, b: i16) -> Struct2Desc {
    // CHECK-NOT: alloca
    // bit32: store i64 %a, ptr %s
    // bit32: %[[GEP:.+]] = getelementptr inbounds i8, ptr %s, i32 8
    // bit32: store i16 %b, ptr %[[GEP]]
    // bit64: %[[TEMP0:.+]] = insertvalue { i64, i16 } poison, i64 %a, 0
    // bit64: %[[TEMP1:.+]] = insertvalue { i64, i16 } %[[TEMP0]], i16 %b, 1
    // bit64: ret { i64, i16 } %[[TEMP1]]
    let s = Struct2Desc(a, b);
    s
}
