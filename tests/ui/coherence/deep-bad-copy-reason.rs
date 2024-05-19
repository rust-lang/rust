#![feature(extern_types)]

extern "Rust" {
    type OpaqueListContents;
}

pub struct ListS<T> {
    //~^ NOTE: required because it appears within the type
    len: usize,
    data: [T; 0],
    opaque: OpaqueListContents,
}

pub struct Interned<'a, T>(&'a T);
//~^ NOTE: required by an implicit `Sized`
//~| NOTE: required by the implicit `Sized`

impl<'a, T> Clone for Interned<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for Interned<'a, T> {}

pub struct List<'tcx, T>(Interned<'tcx, ListS<T>>);
//~^ NOTE this field does not implement `Copy`
//~| NOTE the `Copy` impl for `Interned<'tcx, ListS<T>>` requires that `OpaqueListContents: Sized`
//~| NOTE: doesn't have a size known at compile-time
//~| ERROR: cannot be known at compilation time

impl<'tcx, T> Clone for List<'tcx, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'tcx, T> Copy for List<'tcx, T> {}
//~^ ERROR the trait `Copy` cannot be implemented for this type

fn assert_is_copy<T: Copy>() {}

fn main() {
    assert_is_copy::<List<'static, ()>>();
}
