//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_variances]
struct Ref<'a, T> {
    r: &'a T,
}

#[rustc_dump_variances]
struct RefMut<'a, T> {
    r: &'a mut T,
}

#[rustc_dump_variances]
struct CellRef<'a, T> {
    r: &'a core::cell::UnsafeCell<T>,
}

#[rustc_dump_variances]
fn x<T, U>(_t: T) -> U {
    todo!()
}
