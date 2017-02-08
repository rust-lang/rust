#![feature(plugin)]
#![plugin(clippy)]
#![deny(serde_api_misuse)]
#![allow(dead_code)]

extern crate serde;

struct A;

impl serde::de::Visitor for A {
    type Value = ();

    fn expecting(&self, _: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }

    fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E>
        where E: serde::de::Error,
    {
        unimplemented!()
    }

    fn visit_string<E>(self, _v: String) -> Result<Self::Value, E>
        where E: serde::de::Error,
    {
        unimplemented!()
    }
}

struct B;

impl serde::de::Visitor for B {
    type Value = ();

    fn expecting(&self, _: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }

    fn visit_string<E>(self, _v: String) -> Result<Self::Value, E>

        where E: serde::de::Error,
    {
        unimplemented!()
    }
}

fn main() {
}
