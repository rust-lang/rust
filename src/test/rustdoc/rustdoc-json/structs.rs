use std::collections::HashMap;

pub struct PlainEmpty {}

pub struct Tuple(u32, String);

pub struct Unit;

pub struct WithPrimitives<'a> {
    num: u32,
    s: &'a str,
}

pub struct WithGenerics<T, U> {
    stuff: Vec<T>,
    things: HashMap<U, U>,
}
