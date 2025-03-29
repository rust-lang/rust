//@ check-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::rc::Rc;

struct Struct;

fn main() {
    let vec: Vec<u32> = Vec::new();
    match vec {
        deref!([..]) => {}
        [..] => {}
        _ => {}
    }
    match Box::new(true) {
        deref!(true) => {}
        true => {}
        _ => {}
    }
    match &Box::new(true) {
        deref!(true) => {}
        true => {}
        _ => {}
    }
    match &Rc::new(0) {
        deref!(1..) => {}
        1.. => {}
        _ => {}
    }
    let _: &Struct = match &Rc::new(Struct) {
        deref!(x) => x,
        Struct => &Struct,
        _ => unreachable!(),
    };
    let _: &[Struct] = match &Rc::new(vec![Struct]) {
        deref!(deref!(x)) => x,
        [Struct] => &[Struct],
        _ => unreachable!(),
    };
}
