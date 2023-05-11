// Test that a covariant struct permits the lifetime of a reference to
// be shortened.

#![allow(dead_code)]
// build-pass (FIXME(62277): could be check-pass?)

struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{
    v
}

fn main() { }
