//@ check-pass
//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(ptr_metadata)]

use std::ptr::{self, Pointee};

fn cast_same_meta<T: ?Sized, U: ?Sized>(ptr: *const T) -> *const U
where
    T: Pointee<Metadata = <U as Pointee>::Metadata>,
{
    let (thin, meta) = ptr.to_raw_parts();
    ptr::from_raw_parts(thin, meta)
}

struct Wrapper<T: ?Sized>(T);

// normalize `Wrapper<T>::Metadata` -> `T::Metadata`
fn wrapper_to_tail<T: ?Sized>(ptr: *const T) -> *const Wrapper<T> {
    cast_same_meta(ptr)
}

// normalize `Wrapper<T>::Metadata` -> `T::Metadata` -> `()`
fn wrapper_to_unit<T>(ptr: *const ()) -> *const Wrapper<T> {
    cast_same_meta(ptr)
}

trait Project {
    type Assoc: ?Sized;
}

struct WrapperProject<T: ?Sized + Project>(T::Assoc);

// normalize `WrapperProject<T>::Metadata` -> `T::Assoc::Metadata`
fn wrapper_project_tail<T: ?Sized + Project>(ptr: *const T::Assoc) -> *const WrapperProject<T> {
    cast_same_meta(ptr)
}

// normalize `WrapperProject<T>::Metadata` -> `T::Assoc::Metadata` -> `()`
fn wrapper_project_unit<T: ?Sized + Project>(ptr: *const ()) -> *const WrapperProject<T>
where
    T::Assoc: Sized,
{
    cast_same_meta(ptr)
}

// normalize `<[T] as Pointee>::Metadata` -> `usize`, even if `[T]: Sized`
fn sized_slice<T>(ptr: *const [T]) -> *const str
where
    [T]: Sized,
{
    cast_same_meta(ptr)
}

fn main() {}
