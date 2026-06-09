//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes
//@ compile-flags: -Cpanic=abort -Csymbol-mangling-version=v0
#![feature(custom_mir, core_intrinsics, unboxed_closures)]
#![crate_type = "lib"]
extern crate core;
use core::intrinsics::mir::*;
use std::cell::Cell;
use std::hint::black_box;
use std::mem::ManuallyDrop;

pub struct Big {
    pub blah: [i32; 1024],
}

pub struct BigCell {
    pub blah: [Cell<i32>; 1024],
}

pub struct BigDrop {
    pub blah: [u8; 1024],
}

impl Drop for BigDrop {
    #[inline(never)]
    fn drop(&mut self) {}
}

// CHECK-LABEL: @mutate(
// CHECK-NOT: readonly
// CHECK-SAME: %b)
#[unsafe(no_mangle)]
pub fn mutate(mut b: Big) {
    b.blah[987] = 654;
    black_box(&b);
}

// CHECK-LABEL: @deref_mut({{.*}}readonly {{.*}}captures(none) {{.*}}%c)
#[unsafe(no_mangle)]
pub fn deref_mut(c: (BigCell, &mut usize)) {
    *c.1 = 42;
}

// CHECK-LABEL: @call_copy_arg(ptr {{.*}}readonly {{.*}}captures(none){{.*}})
#[unsafe(no_mangle)]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn call_copy_arg(a: Big) {
    mir! {
        {
            Call(RET = call_copy_arg(a), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

// CHECK-LABEL: @call_move_arg(
// CHECK-NOT:   readonly
// CHECK-SAME:  captures(address)
// CHECK-SAME:  )
#[unsafe(no_mangle)]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn call_move_arg(a: Big) {
    mir! {
        {
            Call(RET = call_move_arg(Move(a)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

fn shared_borrow<T>(a: T) {
    black_box(&a);
}

// Freeze parameter cannot be mutated through a shared borrow.
//
// CHECK-LABEL: ; deduced_param_attrs::shared_borrow::<deduced_param_attrs::Big>
// CHECK-NEXT:  ;
// CHECK-NEXT:  (ptr {{.*}}readonly {{.*}}captures(address) {{.*}}%a)
pub static A0: fn(Big) = shared_borrow;

// !Freeze parameter can be mutated through a shared borrow.
//
// CHECK-LABEL: ; deduced_param_attrs::shared_borrow::<deduced_param_attrs::BigCell>
// CHECK-NEXT:  ;
// CHECK-NOT:   readonly
// CHECK-NEXT:  %a)
pub static A1: fn(BigCell) = shared_borrow;

// The parameter can be mutated through a raw const borrow.
//
// CHECK-LABEL: ; deduced_param_attrs::raw_const_borrow
// CHECK-NEXT:  ;
// CHECK-NOT:   readonly
// CHECK-NEXT:  %a)
#[inline(never)]
pub fn raw_const_borrow(a: Big) {
    black_box(&raw const a);
}

fn consume<T>(_: T) {}

// The parameter doesn't need to be dropped.
//
// CHECK-LABEL: ; deduced_param_attrs::consume::<deduced_param_attrs::BigCell>
// CHECK-NEXT:  ;
// CHECK-NEXT:  (ptr {{.*}}readonly {{.*}}captures(none) {{.*}})
pub static B0: fn(BigCell) = consume;

// The parameter needs to be dropped.
//
// CHECK-LABEL: ; deduced_param_attrs::consume::<deduced_param_attrs::BigDrop>
// CHECK-NEXT:  ;
// CHECK-NEXT:  (ptr {{.*}}captures(address) {{.*}})
pub static B1: fn(BigDrop) = consume;

fn consume_parts<T>(t: (T, T)) {
    let (_t0, ..) = t;
}

// In principle it would be possible to deduce readonly here.
//
// CHECK-LABEL: ; deduced_param_attrs::consume_parts::<[u8; 40]>
// CHECK-NEXT:  ;
// CHECK-NOT:   readonly
// CHECK-NEXT:  %t)
pub static C1: fn(([u8; 40], [u8; 40])) = consume_parts;

// The inner field of ManuallyDrop<BigDrop> needs to be dropped.
//
// CHECK-LABEL: @manually_drop_field(
// CHECK-NOT:   readonly
// CHECK-SAME:  %b)
#[unsafe(no_mangle)]
pub fn manually_drop_field(a: fn() -> BigDrop, mut b: ManuallyDrop<BigDrop>) {
    // FIXME(tmiasko) replace with custom MIR, instead of expecting MIR optimizations to turn this
    // into: drop((_2.0: BigDrop))
    *b = a();
    unsafe { core::intrinsics::unreachable() }
}

// `readonly` is omitted from the return place, even when applicable.
//
// CHECK-LABEL: @never_returns(
// CHECK-NOT:   readonly
// CHECK-SAME:  %_0)
#[unsafe(no_mangle)]
pub fn never_returns() -> [u8; 80] {
    loop {}
}

// CHECK-LABEL: @not_captured_return_place(ptr{{.*}} captures(none) {{.*}}%_0)
#[unsafe(no_mangle)]
pub fn not_captured_return_place() -> [u8; 80] {
    [0u8; 80]
}

// CHECK-LABEL: @captured_return_place(ptr{{.*}} captures(address) {{.*}}%_0)
#[unsafe(no_mangle)]
pub fn captured_return_place() -> [u8; 80] {
    black_box([0u8; 80])
}

// Arguments spread at ABI level are unsupported.
//
// CHECK-LABEL: @spread_arg(
// CHECK-NOT: readonly
// CHECK-SAME: )
#[no_mangle]
pub extern "rust-call" fn spread_arg(_: (Big, Big, Big)) {}
