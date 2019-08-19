// Test various uses of structs with distint variances to make sure
// they permit lifetimes to be approximated as expected.

#![allow(dead_code)]
// build-pass (FIXME(62277): could be check-pass?)

struct SomeStruct<T>(fn(T));

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{
    v
}


fn main() { }
