// Repro for <https://github.com/rust-lang/rust/issues/114046#issue-1819720359>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] known-bug: unknown
//@[no_assumptions] known-bug: #110338

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
    item_size: A::ItemSize, // Removing this member causes the code to compile
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

struct ConstructableImpl<'a> {
    _data: &'a [u8],
}

impl<'a> Callable<'a> for ConstructableImpl<'a> {
    fn callable(_: &'a [u8]) {}
}

struct StructWithLifetime<'a> {
    marker: &'a PhantomData<u8>,
}

async fn async_method() {}

fn assert_send(_: impl Send + Sync) {}

// This async method ought to be send, but is not
async fn my_send_async_method(_struct_with_lifetime: &mut StructWithLifetime<'_>, data: &Vec<u8>) {
    let _named =
        List::<'_, GetterImpl<ConstructableImpl<'_>>> { data, item_size: (), phantom: PhantomData };
    // Moving the await point above the constructed of _named, causes
    // the method to become send, even though _named is Send + Sync
    async_method().await;
    assert_send(_named);
}

fn dummy(struct_with_lifetime: &mut StructWithLifetime<'_>, data: &Vec<u8>) {
    assert_send(my_send_async_method(struct_with_lifetime, data));
}

fn main() {}
