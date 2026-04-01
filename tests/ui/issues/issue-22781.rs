//@ check-pass
#![allow(unused_variables)]
use std::collections::HashMap;
use std::collections::hash_map::Entry::Vacant;

pub fn foo() {
    type F = Box<dyn Fn(&()) + 'static>;
    let mut map: HashMap<(), F> = HashMap::new();
    let x: &mut F = match map.entry(()) {
        Vacant(_) => unimplemented!(),
        _ => unimplemented!()
    };
}

fn main() {}
