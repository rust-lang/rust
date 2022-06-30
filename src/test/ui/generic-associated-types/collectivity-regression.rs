// Regression test from https://github.com/rust-lang/rust/pull/98109
//
// check-pass

#![feature(generic_associated_types)]

pub trait Get {
    type Value<'a>
    where
        Self: 'a;
}

fn multiply_at<T>(x: T)
where
    for<'a> T: Get<Value<'a> = ()>,
{
    || {
        let _x = x;
    };
}

fn main() {}
