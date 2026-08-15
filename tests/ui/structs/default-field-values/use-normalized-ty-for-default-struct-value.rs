//@ check-pass

#![feature(default_field_values)]

struct Value<const VALUE: u8>;

impl<const VALUE: u8> Value<VALUE> {
    pub const VALUE: Self = Self;
}

pub struct WithUse {
    _use: Value<{ 0 + 0 }> = Value::VALUE
}

const _: WithUse = WithUse { .. };

fn main() {}
