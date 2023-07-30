// Make sure we don't include vtable entries for methods that take self by-value,
// see <https://github.com/rust-lang/rust/issues/114007>.
//
// build-fail
#![crate_type = "lib"]
#![feature(rustc_attrs)]

use std::ops::*;

#[rustc_dump_vtable]
pub trait Simple {
    //~^ error: vtable entries for `<() as Simple>`: [
    fn f(self);
    fn g(self: Self);
}

impl Simple for () {
    fn f(self) {}
    fn g(self: Self) {}
}

#[rustc_dump_vtable]
pub trait RefNum<Base>: NumOps<Base, Base> + for<'r> NumOps<&'r Base, Base> {}
//~^ error: vtable entries for `<u32 as RefNum<u32>>`: [

impl<T, Base> RefNum<Base> for T where T: NumOps<Base, Base> + for<'r> NumOps<&'r Base, Base> {}

pub trait NumOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Rem<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> NumOps<Rhs, Output> for T where
    T: Add<Rhs, Output = Output>
        + Sub<Rhs, Output = Output>
        + Mul<Rhs, Output = Output>
        + Div<Rhs, Output = Output>
        + Rem<Rhs, Output = Output>
{
}

pub fn require_vtables() {
    fn require_vtables(_: &dyn RefNum<u32>, _: &dyn Simple) {}

    require_vtables(&1u32, &())
}
