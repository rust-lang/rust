// unit-test: LowerIntrinsics
// ignore-wasm32 compiled with panic=abort by default

#![feature(core_intrinsics, intrinsics, rustc_attrs)]
#![crate_type = "lib"]

// EMIT_MIR lower_intrinsics.wrapping.LowerIntrinsics.diff
pub fn wrapping(a: i32, b: i32) {
    let _x = core::intrinsics::wrapping_add(a, b);
    let _y = core::intrinsics::wrapping_sub(a, b);
    let _z = core::intrinsics::wrapping_mul(a, b);
}

// EMIT_MIR lower_intrinsics.unchecked.LowerIntrinsics.diff
pub unsafe fn unchecked(a: i32, b: i32) {
    let _x = core::intrinsics::unchecked_div(a, b);
    let _y = core::intrinsics::unchecked_rem(a, b);
}

// EMIT_MIR lower_intrinsics.size_of.LowerIntrinsics.diff
pub fn size_of<T>() -> usize {
    core::intrinsics::size_of::<T>()
}

// EMIT_MIR lower_intrinsics.align_of.LowerIntrinsics.diff
pub fn align_of<T>() -> usize {
    core::intrinsics::min_align_of::<T>()
}

// EMIT_MIR lower_intrinsics.forget.LowerIntrinsics.diff
pub fn forget<T>(t: T) {
    core::intrinsics::forget(t)
}

// EMIT_MIR lower_intrinsics.unreachable.LowerIntrinsics.diff
pub fn unreachable() -> ! {
    unsafe { core::intrinsics::unreachable() };
}

// EMIT_MIR lower_intrinsics.non_const.LowerIntrinsics.diff
pub fn non_const<T>() -> usize {
    // Check that lowering works with non-const operand as a func.
    let size_of_t = core::intrinsics::size_of::<T>;
    size_of_t()
}

// EMIT_MIR lower_intrinsics.transmute_inhabited.LowerIntrinsics.diff
pub fn transmute_inhabited(c: std::cmp::Ordering) -> i8 {
    unsafe { std::mem::transmute(c) }
}

// EMIT_MIR lower_intrinsics.transmute_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_uninhabited(u: ()) -> Never {
    unsafe { std::mem::transmute::<(), Never>(u) }
}

// EMIT_MIR lower_intrinsics.transmute_ref_dst.LowerIntrinsics.diff
pub unsafe fn transmute_ref_dst<T: ?Sized>(u: &T) -> *const T {
    unsafe { std::mem::transmute(u) }
}

// EMIT_MIR lower_intrinsics.transmute_to_ref_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_to_ref_uninhabited() -> ! {
    let x: &Never = std::mem::transmute(1usize);
    match *x {}
}

// EMIT_MIR lower_intrinsics.transmute_to_mut_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_to_mut_uninhabited() -> ! {
    let x: &mut Never = std::mem::transmute(1usize);
    match *x {}
}

// EMIT_MIR lower_intrinsics.transmute_to_box_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_to_box_uninhabited() -> ! {
    let x: Box<Never> = std::mem::transmute(1usize);
    match *x {}
}

pub enum E {
    A,
    B,
    C,
}

// EMIT_MIR lower_intrinsics.discriminant.LowerIntrinsics.diff
pub fn discriminant<T>(t: T) {
    core::intrinsics::discriminant_value(&t);
    core::intrinsics::discriminant_value(&0);
    core::intrinsics::discriminant_value(&());
    core::intrinsics::discriminant_value(&E::B);
}

extern "rust-intrinsic" {
    // Cannot use `std::intrinsics::copy_nonoverlapping` as that is a wrapper function
    #[rustc_nounwind]
    fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
}

// EMIT_MIR lower_intrinsics.f_copy_nonoverlapping.LowerIntrinsics.diff
pub fn f_copy_nonoverlapping() {
    let src = ();
    let mut dst = ();
    unsafe {
        copy_nonoverlapping(&src as *const _ as *const i32, &mut dst as *mut _ as *mut i32, 0);
    }
}

// EMIT_MIR lower_intrinsics.assume.LowerIntrinsics.diff
pub fn assume() {
    unsafe {
        std::intrinsics::assume(true);
    }
}

// EMIT_MIR lower_intrinsics.with_overflow.LowerIntrinsics.diff
pub fn with_overflow(a: i32, b: i32) {
    let _x = core::intrinsics::add_with_overflow(a, b);
    let _y = core::intrinsics::sub_with_overflow(a, b);
    let _z = core::intrinsics::mul_with_overflow(a, b);
}

// EMIT_MIR lower_intrinsics.read_via_copy_primitive.LowerIntrinsics.diff
pub fn read_via_copy_primitive(r: &i32) -> i32 {
    unsafe { core::intrinsics::read_via_copy(r) }
}

// EMIT_MIR lower_intrinsics.read_via_copy_uninhabited.LowerIntrinsics.diff
pub fn read_via_copy_uninhabited(r: &Never) -> Never {
    unsafe { core::intrinsics::read_via_copy(r) }
}

// EMIT_MIR lower_intrinsics.write_via_move_string.LowerIntrinsics.diff
pub fn write_via_move_string(r: &mut String, v: String) {
    unsafe { core::intrinsics::write_via_move(r, v) }
}

pub enum Never {}

// EMIT_MIR lower_intrinsics.option_payload.LowerIntrinsics.diff
pub fn option_payload(o: &Option<usize>, p: &Option<String>) {
    unsafe {
        let _x = core::intrinsics::option_payload_ptr(o);
        let _y = core::intrinsics::option_payload_ptr(p);
    }
}

// EMIT_MIR lower_intrinsics.ptr_offset.LowerIntrinsics.diff
pub unsafe fn ptr_offset(p: *const i32, d: isize) -> *const i32 {
    core::intrinsics::offset(p, d)
}
