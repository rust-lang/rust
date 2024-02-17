//@ build-pass
//@ compile-flags: --crate-type=lib

use std::marker::PhantomData;

pub trait StreamOnce {
    type Token;
}

impl StreamOnce for &str {
    type Token = ();
}

pub trait Parser<Input: StreamOnce> {
    type PartialState: Default;
    fn parse_mode(&self, _state: &Self::PartialState) {}
    fn parse_mode_impl() {}
}

pub fn parse_bool<'a>() -> impl Parser<&'a str> {
    pub struct TokensCmp<C, Input>
    where
        Input: StreamOnce,
    {
        _cmp: C,
        _marker: PhantomData<Input>,
    }

    impl<Input, C> Parser<Input> for TokensCmp<C, Input>
    where
        C: FnMut(Input::Token),
        Input: StreamOnce,
    {
        type PartialState = ();
    }

    TokensCmp { _cmp: |_| (), _marker: PhantomData }
}

pub struct ParseBool;

impl<'a> Parser<&'a str> for ParseBool
where
    &'a str: StreamOnce,
{
    type PartialState = ();

    fn parse_mode_impl() {
        parse_bool().parse_mode(&Default::default())
    }
}
