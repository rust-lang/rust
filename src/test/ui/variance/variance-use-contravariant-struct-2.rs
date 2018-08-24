// Test various uses of structs with distint variances to make sure
// they permit lifetimes to be approximated as expected.

#![allow(dead_code)]
#![feature(rustc_attrs)]

struct SomeStruct<T>(fn(T));

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{
    v
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
