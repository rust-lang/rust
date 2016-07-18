#![feature(plugin)]
#![plugin(clippy)]
#![deny(serde_api_misuse)]
#![allow(dead_code)]

extern crate serde;

struct A;

impl serde::de::Visitor for A {
    type Value = ();
    fn visit_str<E>(&mut self, _v: &str) -> Result<Self::Value, E>
        where E: serde::Error,
    {
        unimplemented!()
    }

    fn visit_string<E>(&mut self, _v: String) -> Result<Self::Value, E>
        where E: serde::Error,
    {
        unimplemented!()
    }
}

struct B;

impl serde::de::Visitor for B {
    type Value = ();

    fn visit_string<E>(&mut self, _v: String) -> Result<Self::Value, E>
    //~^ ERROR you should not implement `visit_string` without also implementing `visit_str`
        where E: serde::Error,
    {
        unimplemented!()
    }
}

fn main() {
}
