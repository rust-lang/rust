// normalize-stderr-test "\(\d+ byte\)" -> "(N byte)"
// normalize-stderr-test "\(limit: \d+ byte\)" -> "(limit: N byte)"

#![warn(clippy::large_types_passed_by_value)]

pub struct Large([u8; 2048]);

#[derive(Clone, Copy)]
pub struct LargeAndCopy([u8; 2048]);

pub struct Small([u8; 4]);

#[derive(Clone, Copy)]
pub struct SmallAndCopy([u8; 4]);

fn small(a: Small, b: SmallAndCopy) {}
fn not_copy(a: Large) {}
fn by_ref(a: &Large, b: &LargeAndCopy) {}
fn mutable(mut a: LargeAndCopy) {}
fn bad(a: LargeAndCopy) {}
pub fn bad_but_pub(a: LargeAndCopy) {}

impl LargeAndCopy {
    fn self_is_ok(self) {}
    fn other_is_not_ok(self, other: LargeAndCopy) {}
    fn unless_other_can_change(self, mut other: LargeAndCopy) {}
    pub fn or_were_in_public(self, other: LargeAndCopy) {}
}

trait LargeTypeDevourer {
    fn devoure_array(&self, array: [u8; 6666]);
    fn devoure_tuple(&self, tup: (LargeAndCopy, LargeAndCopy));
    fn devoure_array_and_tuple_wow(&self, array: [u8; 6666], tup: (LargeAndCopy, LargeAndCopy));
}

pub trait PubLargeTypeDevourer {
    fn devoure_array_in_public(&self, array: [u8; 6666]);
}

struct S {}
impl LargeTypeDevourer for S {
    fn devoure_array(&self, array: [u8; 6666]) {
        todo!();
    }
    fn devoure_tuple(&self, tup: (LargeAndCopy, LargeAndCopy)) {
        todo!();
    }
    fn devoure_array_and_tuple_wow(&self, array: [u8; 6666], tup: (LargeAndCopy, LargeAndCopy)) {
        todo!();
    }
}

#[inline(always)]
fn foo_always(x: LargeAndCopy) {
    todo!();
}
#[inline(never)]
fn foo_never(x: LargeAndCopy) {
    todo!();
}
#[inline]
fn foo(x: LargeAndCopy) {
    todo!();
}

fn main() {}
