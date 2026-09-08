//@ check-pass
//@ edition: 2021
//@ compile-flags: --crate-type=lib -Znext-solver=globally -Clink-dead-code

#![allow(unused)]

mod bilrost {
    pub struct DecodeError;

    pub mod bytes {
        pub trait Buf {}
        pub trait BufMut {}
    }

    pub mod buf {
        pub trait ReverseBuf {}
    }

    pub mod encoding {
        use std::marker::PhantomData;

        pub struct General;
        pub struct TagWriter;
        pub struct TagRevWriter;
        pub struct WireType;
        pub struct Capped<B: ?Sized>(PhantomData<B>);

        pub trait DecodeContext {}
        pub trait TagMeasurer {}

        pub trait ForOverwrite<E, T> {
            fn for_overwrite() -> T;
        }

        pub trait EmptyState<E, T> {
            fn is_empty(val: &T) -> bool;
            fn clear(val: &mut T);
        }

        pub trait Encoder<E, T> {}
        pub trait Decoder<E, T> {}
        pub trait BorrowDecoder<'a, E, T> {}

        pub trait ValueEncoder<E, T> {}
        pub trait ValueDecoder<E, T> {}
        pub trait ValueBorrowDecoder<'a, E, T> {}

        pub trait Oneof {
            const FIELD_TAGS: &'static [u32];

            fn empty() -> Self;
            fn is_empty(&self) -> bool;
            fn clear(&mut self);
            fn oneof_current_tag(&self) -> Option<u32>;
            fn oneof_variant_name(tag: u32) -> (&'static str, &'static str);
        }

        pub trait OneofDecoder {}
        pub trait OneofBorrowDecoder<'a> {}

        pub trait RawMessage {
            fn empty() -> Self
            where
                Self: Sized;

            fn is_empty(&self) -> bool;
            fn clear(&mut self);
        }

        pub trait RawMessageDecoder {}
        pub trait RawMessageBorrowDecoder<'a> {}

        impl<T> ForOverwrite<General, T> for ()
        where
            (): ForOverwrite<(), T>,
        {
            fn for_overwrite() -> T {
                <() as ForOverwrite<(), T>>::for_overwrite()
            }
        }

        impl<T> EmptyState<General, Vec<T>> for ()
        where
            (): ValueEncoder<General, T>,
        {
            fn is_empty(_: &Vec<T>) -> bool {
                false
            }

            fn clear(_: &mut Vec<T>) {}
        }

        impl<T> Encoder<General, Vec<T>> for ()
        where
            (): ValueEncoder<General, T>,
        {}

        impl<T> ValueEncoder<General, T> for ()
        where
            T: RawMessage,
        {}
    }
}

use bilrost::encoding::*;

struct Value<'a>(&'a ());
struct ListValue<'a>(&'a ());

impl<'a> Oneof for Value<'a>
where
    (): ForOverwrite<General, ListValue<'a>>,
    (): ValueEncoder<General, ListValue<'a>>,
{
    const FIELD_TAGS: &'static [u32] = &[1, 2];

    fn empty() -> Self {
        panic!()
    }

    fn is_empty(&self) -> bool {
        panic!()
    }

    fn clear(&mut self) {
        panic!()
    }

    fn oneof_current_tag(&self) -> Option<u32> {
        panic!()
    }

    fn oneof_variant_name(_: u32) -> (&'static str, &'static str) {
        panic!()
    }
}

impl<'a> RawMessage for Value<'a>
where
    Value<'a>: Oneof,
{
    fn empty() -> Self {
        panic!()
    }

    fn is_empty(&self) -> bool {
        panic!()
    }

    fn clear(&mut self) {
        panic!()
    }
}

impl<'a> RawMessage for ListValue<'a>
where
    (): EmptyState<General, Vec<Value<'a>>>,
    (): Encoder<General, Vec<Value<'a>>>,
{
    fn empty() -> Self {
        panic!()
    }

    fn is_empty(&self) -> bool {
        panic!()
    }

    fn clear(&mut self) {
        panic!()
    }
}

impl<'a> ForOverwrite<(), ListValue<'a>> for ()
where
    (): EmptyState<General, Vec<Value<'a>>>,
    (): Encoder<General, Vec<Value<'a>>>,
{
    fn for_overwrite() -> ListValue<'a> {
        <ListValue<'a> as RawMessage>::empty()
    }
}

impl<'a> EmptyState<(), ListValue<'a>> for ()
where
    (): EmptyState<General, Vec<Value<'a>>>,
    (): Encoder<General, Vec<Value<'a>>>,
{
    fn is_empty(val: &ListValue<'a>) -> bool {
        <ListValue<'a> as RawMessage>::is_empty(val)
    }

    fn clear(val: &mut ListValue<'a>) {
        <ListValue<'a> as RawMessage>::clear(val);
    }
}
