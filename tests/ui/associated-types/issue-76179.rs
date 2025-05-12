//@ check-pass

#![feature(associated_type_defaults)]

use std::io::Read;

trait View {
    type Deserializers: Deserializer<Item = Self::RequestParams>;
    type RequestParams = DefaultRequestParams;
}

struct DefaultRequestParams;

trait Deserializer {
    type Item;
    fn deserialize(r: impl Read) -> Self::Item;
}

fn main() {}
