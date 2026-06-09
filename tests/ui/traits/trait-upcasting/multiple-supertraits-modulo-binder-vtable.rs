#![feature(rustc_attrs)]

// Test for <https://github.com/rust-lang/rust/issues/135316>.

trait Supertrait<T> {
    fn _print_numbers(&self, mem: &[usize; 100]) {
    }
}
impl<T> Supertrait<T> for () {}

trait Trait<T, U>: Supertrait<T> + Supertrait<U> {
    fn say_hello(&self, _: &usize) {
    }
}
impl<T, U> Trait<T, U> for () {}

// We should observe compatibility between these two vtables.

#[rustc_dump_vtable]
type First = dyn for<'a> Trait<&'static (), &'a ()>;
//~^ ERROR vtable entries

#[rustc_dump_vtable]
type Second = dyn Trait<&'static (), &'static ()>;
//~^ ERROR vtable entries

fn main() {}
