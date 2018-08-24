// Test that a covariant struct permits the lifetime of a reference to
// be shortened.

#![allow(dead_code)]
#![feature(rustc_attrs)]

struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{
    v
}

#[rustc_error] fn main() { } //~ ERROR compilation successful
