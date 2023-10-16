// unit-test: LowerIntrinsics
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(core_intrinsics, intrinsics, rustc_attrs)]
#![crate_type = "lib"]

// EMIT_MIR lower_intrinsics.wrapping.LowerIntrinsics.diff
pub fn wrapping(a: i32, b: i32) {
    // CHECK-LABEL: fn wrapping(
    // CHECK: {{_.*}} = Add(
    // CHECK: {{_.*}} = Sub(
    // CHECK: {{_.*}} = Mul(
    let _x = core::intrinsics::wrapping_add(a, b);
    let _y = core::intrinsics::wrapping_sub(a, b);
    let _z = core::intrinsics::wrapping_mul(a, b);
}

// EMIT_MIR lower_intrinsics.unchecked.LowerIntrinsics.diff
pub unsafe fn unchecked(a: i32, b: i32) {
    // CHECK-LABEL: fn unchecked(
    // CHECK: {{_.*}} = AddUnchecked(
    // CHECK: {{_.*}} = SubUnchecked(
    // CHECK: {{_.*}} = MulUnchecked(
    // CHECK: {{_.*}} = Div(
    // CHECK: {{_.*}} = Rem(
    // CHECK: {{_.*}} = ShlUnchecked(
    // CHECK: {{_.*}} = ShrUnchecked(
    let _a = core::intrinsics::unchecked_add(a, b);
    let _b = core::intrinsics::unchecked_sub(a, b);
    let _c = core::intrinsics::unchecked_mul(a, b);
    let _x = core::intrinsics::unchecked_div(a, b);
    let _y = core::intrinsics::unchecked_rem(a, b);
    let _i = core::intrinsics::unchecked_shl(a, b);
    let _j = core::intrinsics::unchecked_shr(a, b);
}

// EMIT_MIR lower_intrinsics.size_of.LowerIntrinsics.diff
pub fn size_of<T>() -> usize {
    // CHECK-LABEL: fn size_of(
    // CHECK: {{_.*}} = SizeOf(T);
    core::intrinsics::size_of::<T>()
}

// EMIT_MIR lower_intrinsics.align_of.LowerIntrinsics.diff
pub fn align_of<T>() -> usize {
    // CHECK-LABEL: fn align_of(
    // CHECK: {{_.*}} = AlignOf(T);
    core::intrinsics::min_align_of::<T>()
}

// EMIT_MIR lower_intrinsics.forget.LowerIntrinsics.diff
pub fn forget<T>(t: T) {
    // CHECK-LABEL: fn forget(
    // CHECK-NOT: Drop(
    // CHECK: return;
    // CHECK-NOT: Drop(
    core::intrinsics::forget(t)
}

// EMIT_MIR lower_intrinsics.unreachable.LowerIntrinsics.diff
pub fn unreachable() -> ! {
    // CHECK-LABEL: fn unreachable(
    // CHECK: unreachable;
    unsafe { core::intrinsics::unreachable() };
}

// EMIT_MIR lower_intrinsics.non_const.LowerIntrinsics.diff
pub fn non_const<T>() -> usize {
    // CHECK-LABEL: fn non_const(
    // CHECK: SizeOf(T);

    // Check that lowering works with non-const operand as a func.
    let size_of_t = core::intrinsics::size_of::<T>;
    size_of_t()
}

// EMIT_MIR lower_intrinsics.transmute_inhabited.LowerIntrinsics.diff
pub fn transmute_inhabited(c: std::cmp::Ordering) -> i8 {
    // CHECK-LABEL: fn transmute_inhabited(
    // CHECK: {{_.*}} = {{.*}} as i8 (Transmute);

    unsafe { std::mem::transmute(c) }
}

// EMIT_MIR lower_intrinsics.transmute_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_uninhabited(u: ()) -> Never {
    // CHECK-LABEL: fn transmute_uninhabited(
    // CHECK: {{_.*}} = {{.*}} as Never (Transmute);
    // CHECK: unreachable;

    unsafe { std::mem::transmute::<(), Never>(u) }
}

// EMIT_MIR lower_intrinsics.transmute_ref_dst.LowerIntrinsics.diff
pub unsafe fn transmute_ref_dst<T: ?Sized>(u: &T) -> *const T {
    // CHECK-LABEL: fn transmute_ref_dst(
    // CHECK: {{_.*}} = {{.*}} as *const T (Transmute);

    unsafe { std::mem::transmute(u) }
}

// EMIT_MIR lower_intrinsics.transmute_to_ref_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_to_ref_uninhabited() -> ! {
    // CHECK-LABEL: fn transmute_to_ref_uninhabited(
    // CHECK: {{_.*}} = {{.*}} as &Never (Transmute);
    // CHECK: unreachable;

    let x: &Never = std::mem::transmute(1usize);
    match *x {}
}

// EMIT_MIR lower_intrinsics.transmute_to_mut_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_to_mut_uninhabited() -> ! {
    // CHECK-LABEL: fn transmute_to_mut_uninhabited(
    // CHECK: {{_.*}} = {{.*}} as &mut Never (Transmute);
    // CHECK: unreachable;

    let x: &mut Never = std::mem::transmute(1usize);
    match *x {}
}

// EMIT_MIR lower_intrinsics.transmute_to_box_uninhabited.LowerIntrinsics.diff
pub unsafe fn transmute_to_box_uninhabited() -> ! {
    // CHECK-LABEL: fn transmute_to_box_uninhabited(
    // CHECK: {{_.*}} = {{.*}} as std::boxed::Box<Never> (Transmute);
    // CHECK: unreachable;

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
    // CHECK-LABEL: fn discriminant(
    // CHECK: {{_.*}} = discriminant(
    // CHECK: {{_.*}} = discriminant(
    // CHECK: {{_.*}} = discriminant(
    // CHECK: {{_.*}} = discriminant(

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
    // CHECK-LABEL: fn f_copy_nonoverlapping(
    // CHECK: copy_nonoverlapping({{.*}});

    let src = ();
    let mut dst = ();
    unsafe {
        copy_nonoverlapping(&src as *const _ as *const i32, &mut dst as *mut _ as *mut i32, 0);
    }
}

// EMIT_MIR lower_intrinsics.assume.LowerIntrinsics.diff
pub fn assume() {
    // CHECK-LABEL: fn assume(
    // CHECK: assume({{.*}});

    unsafe {
        std::intrinsics::assume(true);
    }
}

// EMIT_MIR lower_intrinsics.with_overflow.LowerIntrinsics.diff
pub fn with_overflow(a: i32, b: i32) {
    // CHECK-LABEL: fn with_overflow(
    // CHECK: CheckedAdd(
    // CHECK: CheckedSub(
    // CHECK: CheckedMul(

    let _x = core::intrinsics::add_with_overflow(a, b);
    let _y = core::intrinsics::sub_with_overflow(a, b);
    let _z = core::intrinsics::mul_with_overflow(a, b);
}

// EMIT_MIR lower_intrinsics.read_via_copy_primitive.LowerIntrinsics.diff
pub fn read_via_copy_primitive(r: &i32) -> i32 {
    // CHECK-LABEL: fn read_via_copy_primitive(
    // CHECK: [[tmp:_.*]] = &raw const (*_1);
    // CHECK: _0 = (*[[tmp]]);
    // CHECK: return;

    unsafe { core::intrinsics::read_via_copy(r) }
}

// EMIT_MIR lower_intrinsics.read_via_copy_uninhabited.LowerIntrinsics.diff
pub fn read_via_copy_uninhabited(r: &Never) -> Never {
    // CHECK-LABEL: fn read_via_copy_uninhabited(
    // CHECK: [[tmp:_.*]] = &raw const (*_1);
    // CHECK: _0 = (*[[tmp]]);
    // CHECK: unreachable;

    unsafe { core::intrinsics::read_via_copy(r) }
}

// EMIT_MIR lower_intrinsics.write_via_move_string.LowerIntrinsics.diff
pub fn write_via_move_string(r: &mut String, v: String) {
    // CHECK-LABEL: fn write_via_move_string(
    // CHECK: [[ptr:_.*]] = &raw mut (*_1);
    // CHECK: [[tmp:_.*]] = move _2;
    // CHECK: (*[[ptr]]) = move [[tmp]];
    // CHECK: return;

    unsafe { core::intrinsics::write_via_move(r, v) }
}

pub enum Never {}

// EMIT_MIR lower_intrinsics.option_payload.LowerIntrinsics.diff
pub fn option_payload(o: &Option<usize>, p: &Option<String>) {
    // CHECK-LABEL: fn option_payload(
    // CHECK: {{_.*}} = &raw const (((*{{_.*}}) as Some).0: usize);
    // CHECK: {{_.*}} = &raw const (((*{{_.*}}) as Some).0: std::string::String);

    unsafe {
        let _x = core::intrinsics::option_payload_ptr(o);
        let _y = core::intrinsics::option_payload_ptr(p);
    }
}

// EMIT_MIR lower_intrinsics.ptr_offset.LowerIntrinsics.diff
pub unsafe fn ptr_offset(p: *const i32, d: isize) -> *const i32 {
    // CHECK-LABEL: fn ptr_offset(
    // CHECK: _0 = Offset(

    core::intrinsics::offset(p, d)
}
