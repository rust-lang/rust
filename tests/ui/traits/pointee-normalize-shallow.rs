//@ check-pass

#![feature(ptr_metadata)]

use std::ptr::Thin;

struct Wrapper<T: ?Sized>(T);

fn check_thin<T: ?Sized + Thin>() {}

// Test that normalization of `<Wrapper<[T]> as Pointee>::Metadata` respects the
// `<[T] as Pointee>::Metadata == ()` bound from the param env.
fn foo<T>()
where
    [T]: Thin,
{
    check_thin::<Wrapper<[T]>>();
}

fn main() {}
