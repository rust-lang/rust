#![warn(clippy::serde_api_misuse)]
#![allow(dead_code, clippy::needless_lifetimes)]

extern crate serde;

struct A;

impl<'de> serde::de::Visitor<'de> for A {
    type Value = ();

    fn expecting(&self, _: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }

    fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        unimplemented!()
    }

    fn visit_string<E>(self, _v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        unimplemented!()
    }
}

struct B;

impl<'de> serde::de::Visitor<'de> for B {
    type Value = ();

    fn expecting(&self, _: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }

    fn visit_string<E>(self, _v: String) -> Result<Self::Value, E>
    //~^ ERROR: you should not implement `visit_string` without also implementing `visit_s
    //~| NOTE: `-D clippy::serde-api-misuse` implied by `-D warnings`
    where
        E: serde::de::Error,
    {
        unimplemented!()
    }
}

fn main() {}
