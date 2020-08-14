use std::collections::HashMap;

pub struct Normal {}

pub struct Tuple();

pub struct Unit;

pub struct WithPrimitives<'a> {
    num: u32,
    s: &'a str,
}

pub struct WithGenerics<T, U> {
    stuff: Vec<T>,
    things: HashMap<U, U>,
}
