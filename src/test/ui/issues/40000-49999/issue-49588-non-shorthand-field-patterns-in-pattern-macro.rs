// run-pass
#![allow(unused_variables)]
#![deny(non_shorthand_field_patterns)]

pub struct Value<A> { pub value: A }

#[macro_export]
macro_rules! pat {
    ($a:pat) => {
        Value { value: $a }
    };
}

fn main() {
    let pat!(value) = Value { value: () };
}
