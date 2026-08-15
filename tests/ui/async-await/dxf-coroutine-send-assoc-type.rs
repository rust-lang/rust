//@ edition: 2024
//@ revisions: no_dxf dxf dxf_aob
//@[no_dxf] compile-flags: -Znext-solver -Zassumptions-on-binders
//@[no_dxf] check-pass
//@[dxf] compile-flags: -Znext-solver -Zdxf
//@[dxf] known-bug: #114046
//@[dxf_aob] compile-flags: -Znext-solver -Zdxf -Zassumptions-on-binders
//@[dxf_aob] check-pass

// Adapted from the pattern in issue #114046 (higher-ranked-auto-trait-13).
// Associated types create lifetime dependencies that make Send conditional
// on lifetime constraints from NLL SCC data.

#![allow(dead_code)]
use std::marker::PhantomData;

trait Callable<'a>: Send + Sync {
    fn callable(data: &'a [u8]);
}

trait Getter<'a>: Send + Sync {
    type ItemSize: Send + Sync;
    fn get(data: &'a [u8]);
}

struct List<'a, A: Getter<'a>> {
    data: &'a [u8],
    item_size: A::ItemSize,
    phantom: PhantomData<A>,
}

struct GetterImpl<'a, T: Callable<'a> + 'a> {
    p: PhantomData<&'a T>,
}

impl<'a, T: Callable<'a> + 'a> Getter<'a> for GetterImpl<'a, T> {
    type ItemSize = ();
    fn get(data: &'a [u8]) {
        <T>::callable(data);
    }
}

struct Impl<'a> {
    _data: &'a [u8],
}

impl<'a> Callable<'a> for Impl<'a> {
    fn callable(_: &'a [u8]) {}
}

struct StructWithLifetime<'a> {
    marker: &'a PhantomData<u8>,
}

async fn yield_point() {}

fn assert_send(_: impl Send) {}

async fn my_method(s: &mut StructWithLifetime<'_>, data: &[u8]) {
    let _named = List::<'_, GetterImpl<Impl<'_>>> {
        data,
        item_size: (),
        phantom: PhantomData,
    };
    yield_point().await;
    drop(_named);
}

fn main() {
    let ph = PhantomData;
    let mut s = StructWithLifetime { marker: &ph };
    let data = vec![1u8];
    assert_send(my_method(&mut s, &data));
}
