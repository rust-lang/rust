// compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

// FIXME(eddyb) all of these tests show memory stores and loads, even after a
// scalar `bitcast`, more special-casing is required to remove `alloca` usage.

// CHECK-LABEL: define{{.*}}i32 @f32_to_bits(float %x)
// CHECK: store i32 %{{.*}}, {{.*}} %0
// CHECK-NEXT: %[[RES:.*]] = load i32, {{.*}} %0
// CHECK: ret i32 %[[RES]]
#[no_mangle]
pub fn f32_to_bits(x: f32) -> u32 {
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: define{{.*}}i8 @bool_to_byte(i1 noundef zeroext %b)
// CHECK: %1 = zext i1 %b to i8
// CHECK-NEXT: store i8 %1, {{.*}} %0
// CHECK-NEXT: %2 = load i8, {{.*}} %0
// CHECK: ret i8 %2
#[no_mangle]
pub fn bool_to_byte(b: bool) -> u8 {
    unsafe { std::mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}noundef zeroext i1 @byte_to_bool(i8 %byte)
// CHECK: %1 = trunc i8 %byte to i1
// CHECK-NEXT: %2 = zext i1 %1 to i8
// CHECK-NEXT: store i8 %2, {{.*}} %0
// CHECK-NEXT: %3 = load i8, {{.*}} %0
// CHECK-NEXT: %4 = trunc i8 %3 to i1
// CHECK: ret i1 %4
#[no_mangle]
pub unsafe fn byte_to_bool(byte: u8) -> bool {
    std::mem::transmute(byte)
}

// CHECK-LABEL: define{{.*}}{{i8\*|ptr}} @ptr_to_ptr({{i16\*|ptr}} %p)
// CHECK: store {{i8\*|ptr}} %{{.*}}, {{.*}} %0
// CHECK-NEXT: %[[RES:.*]] = load {{i8\*|ptr}}, {{.*}} %0
// CHECK: ret {{i8\*|ptr}} %[[RES]]
#[no_mangle]
pub fn ptr_to_ptr(p: *mut u16) -> *mut u8 {
    unsafe { std::mem::transmute(p) }
}

// HACK(eddyb) scalar `transmute`s between pointers and non-pointers are
// currently not special-cased like other scalar `transmute`s, because
// LLVM requires specifically `ptrtoint`/`inttoptr` instead of `bitcast`.
//
// Tests below show the non-special-cased behavior (with the possible
// future special-cased instructions in the "NOTE(eddyb)" comments).

// CHECK: define{{.*}}[[USIZE:i[0-9]+]] @ptr_to_int({{i16\*|ptr}} %p)

// NOTE(eddyb) see above, the following two CHECK lines should ideally be this:
//        %2 = ptrtoint i16* %p to [[USIZE]]
//             store [[USIZE]] %2, [[USIZE]]* %0
// CHECK: store {{i16\*|ptr}} %p, {{.*}}

// CHECK-NEXT: %[[RES:.*]] = load [[USIZE]], {{.*}} %0
// CHECK: ret [[USIZE]] %[[RES]]
#[no_mangle]
pub fn ptr_to_int(p: *mut u16) -> usize {
    unsafe { std::mem::transmute(p) }
}

// CHECK: define{{.*}}{{i16\*|ptr}} @int_to_ptr([[USIZE]] %i)

// NOTE(eddyb) see above, the following two CHECK lines should ideally be this:
//        %2 = inttoptr [[USIZE]] %i to i16*
//             store i16* %2, i16** %0
// CHECK: store [[USIZE]] %i, {{.*}}

// CHECK-NEXT: %[[RES:.*]] = load {{i16\*|ptr}}, {{.*}} %0
// CHECK: ret {{i16\*|ptr}} %[[RES]]
#[no_mangle]
pub fn int_to_ptr(i: usize) -> *mut u16 {
    unsafe { std::mem::transmute(i) }
}
