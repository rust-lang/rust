//@ check-pass

#![feature(type_changing_struct_update)]

use std::borrow::Cow;
use std::marker::PhantomData;

#[derive(Default)]
struct NonGeneric {
    field1: usize,
}

#[derive(Default)]
struct Generic<T, U> {
    field1: T,
    field2: U,
}

#[derive(Default)]
struct MoreGeneric<'a, const N: usize> {
    // If only `for<const N: usize> [u32; N]: Default`...
    field1: PhantomData<[u32; N]>,
    field2: Cow<'a, str>,
}

fn main() {
    let default1 = NonGeneric { ..Default::default() };
    let default2: Generic<i32, f32> = Generic { ..Default::default() };
    let default3: MoreGeneric<'static, 12> = MoreGeneric { ..Default::default() };
}
