//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(allocator_api)]

use std::marker::PhantomPinned;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ptr::NonNull;

pub struct S {
    _field: [i32; 8],
}

pub struct UnsafeInner {
    _field: std::cell::UnsafeCell<i16>,
}

pub struct NotUnpin {
    _field: i32,
    _marker: PhantomPinned,
}

pub enum MyBool {
    True,
    False,
}

// CHECK: noundef zeroext i1 @boolean(i1 noundef zeroext %x)
#[no_mangle]
pub fn boolean(x: bool) -> bool {
    x
}

// CHECK: i8 @maybeuninit_boolean(i8{{.*}} %x)
#[no_mangle]
pub fn maybeuninit_boolean(x: MaybeUninit<bool>) -> MaybeUninit<bool> {
    x
}

// CHECK: noundef zeroext i1 @enum_bool(i1 noundef zeroext %x)
#[no_mangle]
pub fn enum_bool(x: MyBool) -> MyBool {
    x
}

// CHECK: i8 @maybeuninit_enum_bool(i8{{.*}} %x)
#[no_mangle]
pub fn maybeuninit_enum_bool(x: MaybeUninit<MyBool>) -> MaybeUninit<MyBool> {
    x
}

// CHECK: noundef{{( range\(i32 0, 1114112\))?}} i32 @char(i32{{.*}}{{( range\(i32 0, 1114112\))?}} %x)
#[no_mangle]
pub fn char(x: char) -> char {
    x
}

// CHECK: i32 @maybeuninit_char(i32{{.*}} %x)
#[no_mangle]
pub fn maybeuninit_char(x: MaybeUninit<char>) -> MaybeUninit<char> {
    x
}

// CHECK: noundef i64 @int(i64 noundef %x)
#[no_mangle]
pub fn int(x: u64) -> u64 {
    x
}

// CHECK: noundef{{( range\(i64 1, 0\))?}} i64 @nonzero_int(i64 noundef{{( range\(i64 1, 0\))?}} %x)
#[no_mangle]
pub fn nonzero_int(x: NonZero<u64>) -> NonZero<u64> {
    x
}

// CHECK: noundef i64 @option_nonzero_int(i64 noundef %x)
#[no_mangle]
pub fn option_nonzero_int(x: Option<NonZero<u64>>) -> Option<NonZero<u64>> {
    x
}

// CHECK: @readonly_borrow(ptr noalias noundef readonly align 4{{( captures\(address, read_provenance\))?}} dereferenceable(4) %_1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn readonly_borrow(_: &i32) {}

// CHECK: noundef align 4 dereferenceable(4) ptr @readonly_borrow_ret()
#[no_mangle]
pub fn readonly_borrow_ret() -> &'static i32 {
    loop {}
}

// CHECK: @static_borrow(ptr noalias noundef readonly align 4{{( captures\(address, read_provenance\))?}} dereferenceable(4) %_1)
// static borrow may be captured
#[no_mangle]
pub fn static_borrow(_: &'static i32) {}

// CHECK: @named_borrow(ptr noalias noundef readonly align 4{{( captures\(address, read_provenance\))?}} dereferenceable(4) %_1)
// borrow with named lifetime may be captured
#[no_mangle]
pub fn named_borrow<'r>(_: &'r i32) {}

// CHECK: @unsafe_borrow(ptr noundef nonnull align 2 %_1)
// unsafe interior means this isn't actually readonly and there may be aliases ...
#[no_mangle]
pub fn unsafe_borrow(_: &UnsafeInner) {}

// CHECK: @mutable_unsafe_borrow(ptr noalias noundef align 2 dereferenceable(2) %_1)
// ... unless this is a mutable borrow, those never alias
#[no_mangle]
pub fn mutable_unsafe_borrow(_: &mut UnsafeInner) {}

// CHECK: @mutable_borrow(ptr noalias noundef align 4 dereferenceable(4) %_1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn mutable_borrow(_: &mut i32) {}

// CHECK: noundef align 4 dereferenceable(4) ptr @mutable_borrow_ret()
#[no_mangle]
pub fn mutable_borrow_ret() -> &'static mut i32 {
    loop {}
}

#[no_mangle]
// CHECK: @mutable_notunpin_borrow(ptr noundef nonnull align 4 %_1)
// This one is *not* `noalias` because it might be self-referential.
// It is also not `dereferenceable` due to
// <https://github.com/rust-lang/unsafe-code-guidelines/issues/381>.
pub fn mutable_notunpin_borrow(_: &mut NotUnpin) {}

// CHECK: @notunpin_borrow(ptr noalias noundef readonly align 4{{( captures\(address, read_provenance\))?}} dereferenceable(4) %_1)
// But `&NotUnpin` behaves perfectly normal.
#[no_mangle]
pub fn notunpin_borrow(_: &NotUnpin) {}

// CHECK: @indirect_struct(ptr{{( dead_on_return)?}} noalias noundef readonly align 4{{( captures\(address\))?}} dereferenceable(32) %_1)
#[no_mangle]
pub fn indirect_struct(_: S) {}

// CHECK: @borrowed_struct(ptr noalias noundef readonly align 4{{( captures\(address, read_provenance\))?}} dereferenceable(32) %_1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn borrowed_struct(_: &S) {}

// CHECK: @option_borrow(ptr noalias noundef readonly align 4{{( captures\(address, read_provenance\))?}} dereferenceable_or_null(4) %_x)
#[no_mangle]
pub fn option_borrow(_x: Option<&i32>) {}

// CHECK: @option_borrow_mut(ptr noalias noundef align 4 dereferenceable_or_null(4) %_x)
#[no_mangle]
pub fn option_borrow_mut(_x: Option<&mut i32>) {}

// Function that must NOT have `dereferenceable` or `align`.
#[rustc_layout_scalar_valid_range_start(16)]
pub struct RestrictedAddress(&'static i16);
enum E {
    A(RestrictedAddress),
    B,
    C,
}
// If the `nonnull` ever goes missing, you might have to tweak the
// scalar_valid_range on `RestrictedAddress` to get it back. You
// might even have to add a `rustc_layout_scalar_valid_range_end`.
// CHECK: @nonnull_and_nondereferenceable(ptr noundef nonnull %_x)
#[no_mangle]
pub fn nonnull_and_nondereferenceable(_x: E) {}

// CHECK: @raw_struct(ptr noundef %_1)
#[no_mangle]
pub fn raw_struct(_: *const S) {}

// CHECK: @raw_option_nonnull_struct(ptr noundef %_1)
#[no_mangle]
pub fn raw_option_nonnull_struct(_: Option<NonNull<S>>) {}

// `Box` can get deallocated during execution of the function, so it should
// not get `dereferenceable`.
// CHECK: noundef nonnull align 4 ptr @_box(ptr noalias noundef nonnull align 4 %x)
#[no_mangle]
pub fn _box(x: Box<i32>) -> Box<i32> {
    x
}

// With a custom allocator, it should *not* have `noalias`. (See
// <https://github.com/rust-lang/miri/issues/3341> for why.) The second argument is the allocator,
// which is a reference here that still carries `noalias` as usual.
// CHECK: @_box_custom(ptr noundef nonnull align 4 %x.0, ptr noalias noundef nonnull readonly align 1{{( captures\(address, read_provenance\))?}} %x.1)
#[no_mangle]
pub fn _box_custom(x: Box<i32, &std::alloc::Global>) {
    drop(x)
}

// CHECK: noundef nonnull align 4 ptr @notunpin_box(ptr noundef nonnull align 4 %x)
#[no_mangle]
pub fn notunpin_box(x: Box<NotUnpin>) -> Box<NotUnpin> {
    x
}

// CHECK: @struct_return(ptr{{( dead_on_unwind)?}} noalias noundef{{( writable)?}} sret([32 x i8]) align 4{{( captures\(address\))?}} dereferenceable(32){{( %_0)?}})
#[no_mangle]
pub fn struct_return() -> S {
    S { _field: [0, 0, 0, 0, 0, 0, 0, 0] }
}

// Hack to get the correct size for the length part in slices
// CHECK: @helper([[USIZE:i[0-9]+]] noundef %_1)
#[no_mangle]
pub fn helper(_: usize) {}

// CHECK: @slice(ptr noalias noundef nonnull readonly align 1{{( captures\(address, read_provenance\))?}} %_1.0, [[USIZE]] noundef %_1.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn slice(_: &[u8]) {}

// CHECK: @mutable_slice(ptr noalias noundef nonnull align 1 %_1.0, [[USIZE]] noundef %_1.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn mutable_slice(_: &mut [u8]) {}

// CHECK: @unsafe_slice(ptr noundef nonnull align 2 %_1.0, [[USIZE]] noundef %_1.1)
// unsafe interior means this isn't actually readonly and there may be aliases ...
#[no_mangle]
pub fn unsafe_slice(_: &[UnsafeInner]) {}

// CHECK: @raw_slice(ptr noundef %_1.0, [[USIZE]] noundef %_1.1)
#[no_mangle]
pub fn raw_slice(_: *const [u8]) {}

// CHECK: @str(ptr noalias noundef nonnull readonly align 1{{( captures\(address, read_provenance\))?}} %_1.0, [[USIZE]] noundef %_1.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn str(_: &[u8]) {}

// CHECK: @trait_borrow(ptr noundef nonnull align 1 %_1.0, {{.+}} noalias noundef readonly align {{.*}} dereferenceable({{.*}}) %_1.1)
// FIXME #25759 This should also have `nocapture`
#[no_mangle]
pub fn trait_borrow(_: &dyn Drop) {}

// CHECK: @option_trait_borrow(ptr noundef align 1 %x.0, ptr %x.1)
#[no_mangle]
pub fn option_trait_borrow(x: Option<&dyn Drop>) {}

// CHECK: @option_trait_borrow_mut(ptr noundef align 1 %x.0, ptr %x.1)
#[no_mangle]
pub fn option_trait_borrow_mut(x: Option<&mut dyn Drop>) {}

// CHECK: @trait_raw(ptr noundef %_1.0, {{.+}} noalias noundef readonly align {{.*}} dereferenceable({{.*}}) %_1.1)
#[no_mangle]
pub fn trait_raw(_: *const dyn Drop) {}

// CHECK: @trait_box(ptr noalias noundef nonnull align 1{{( %0)?}}, {{.+}} noalias noundef readonly align {{.*}} dereferenceable({{.*}}){{( %1)?}})
#[no_mangle]
pub fn trait_box(_: Box<dyn Drop + Unpin>) {}

// CHECK: { ptr, ptr } @trait_option(ptr noalias noundef align 1 %x.0, ptr %x.1)
#[no_mangle]
pub fn trait_option(x: Option<Box<dyn Drop + Unpin>>) -> Option<Box<dyn Drop + Unpin>> {
    x
}

// CHECK: { ptr, [[USIZE]] } @return_slice(ptr noalias noundef nonnull readonly align 2{{( captures\(address, read_provenance\))?}} %x.0, [[USIZE]] noundef %x.1)
#[no_mangle]
pub fn return_slice(x: &[u16]) -> &[u16] {
    x
}

// CHECK: { i16, i16 } @enum_id_1(i16 noundef{{( range\(i16 0, 3\))?}} %x.0, i16 %x.1)
#[no_mangle]
pub fn enum_id_1(x: Option<Result<u16, u16>>) -> Option<Result<u16, u16>> {
    x
}

// CHECK: { i1, i8 } @enum_id_2(i1 noundef zeroext %x.0, i8 %x.1)
#[no_mangle]
pub fn enum_id_2(x: Option<u8>) -> Option<u8> {
    x
}
