//@ compile-flags: -Z mir-opt-level=4
//@ build-pass

// This used to ICE in const-prop due
// to an empty ParamEnv being used during normalization
// of a generic type


fn main() {
    join_all::<u32>();
}

trait Foo {
    type Item;
}

impl Foo for u32 {
    type Item = u8;
}

trait Bar {
    type Item2;
}

impl Bar for u8 {
    type Item2 = u64;
}

fn join_all<I>()
where I: Foo,
    I::Item: Bar
{
    Vec::<<I::Item as Bar>::Item2>::new(); // ICE occurs processing this line
}
