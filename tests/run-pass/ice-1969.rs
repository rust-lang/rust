#![feature(plugin)]
#![plugin(clippy)]
#![allow(clippy)]

fn main() { }

pub trait Convert {
    type Action: From<*const f64>;

    fn convert(val: *const f64) -> Self::Action {
        val.into()
    }
}
