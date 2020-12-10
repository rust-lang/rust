// compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

// FIXME(eddyb) all of these tests show memory stores and loads, even after a
// scalar `bitcast`, more special-casing is required to remove `alloca` usage.

// CHECK: define i32 @f32_to_bits(float %x)
// CHECK: %2 = bitcast float %x to i32
// CHECK-NEXT: store i32 %2, i32* %0
// CHECK-NEXT: %3 = load i32, i32* %0
// CHECK: ret i32 %3
#[no_mangle]
pub fn f32_to_bits(x: f32) -> u32 {
    unsafe { std::mem::transmute(x) }
}

// CHECK: define i8 @bool_to_byte(i1 zeroext %b)
// CHECK: %1 = zext i1 %b to i8
// CHECK-NEXT: store i8 %1, i8* %0
// CHECK-NEXT: %2 = load i8, i8* %0
// CHECK: ret i8 %2
#[no_mangle]
pub fn bool_to_byte(b: bool) -> u8 {
    unsafe { std::mem::transmute(b) }
}

// CHECK: define zeroext i1 @byte_to_bool(i8 %byte)
// CHECK: %1 = trunc i8 %byte to i1
// CHECK-NEXT: %2 = zext i1 %1 to i8
// CHECK-NEXT: store i8 %2, i8* %0
// CHECK-NEXT: %3 = load i8, i8* %0
// CHECK-NEXT: %4 = trunc i8 %3 to i1
// CHECK: ret i1 %4
#[no_mangle]
pub unsafe fn byte_to_bool(byte: u8) -> bool {
    std::mem::transmute(byte)
}

// CHECK: define i8* @ptr_to_ptr(i16* %p)
// CHECK: %2 = bitcast i16* %p to i8*
// CHECK-NEXT: store i8* %2, i8** %0
// CHECK-NEXT: %3 = load i8*, i8** %0
// CHECK: ret i8* %3
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

// CHECK: define [[USIZE:i[0-9]+]] @ptr_to_int(i16* %p)

// NOTE(eddyb) see above, the following two CHECK lines should ideally be this:
//        %2 = ptrtoint i16* %p to [[USIZE]]
//             store [[USIZE]] %2, [[USIZE]]* %0
// CHECK: %2 = bitcast [[USIZE]]* %0 to i16**
// CHECK-NEXT: store i16* %p, i16** %2

// CHECK-NEXT: %3 = load [[USIZE]], [[USIZE]]* %0
// CHECK: ret [[USIZE]] %3
#[no_mangle]
pub fn ptr_to_int(p: *mut u16) -> usize {
    unsafe { std::mem::transmute(p) }
}

// CHECK: define i16* @int_to_ptr([[USIZE]] %i)

// NOTE(eddyb) see above, the following two CHECK lines should ideally be this:
//        %2 = inttoptr [[USIZE]] %i to i16*
//             store i16* %2, i16** %0
// CHECK: %2 = bitcast i16** %0 to [[USIZE]]*
// CHECK-NEXT: store [[USIZE]] %i, [[USIZE]]* %2

// CHECK-NEXT: %3 = load i16*, i16** %0
// CHECK: ret i16* %3
#[no_mangle]
pub fn int_to_ptr(i: usize) -> *mut u16 {
    unsafe { std::mem::transmute(i) }
}
