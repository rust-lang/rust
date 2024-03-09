//@ check-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::rc::Rc;

struct Struct;

fn main() {
    let vec: Vec<u32> = Vec::new();
    match vec {
        box [..] => {}
        _ => {}
    }
    match Box::new(true) {
        box true => {}
        _ => {}
    }
    match &Box::new(true) {
        box true => {}
        _ => {}
    }
    match &Rc::new(0) {
        box (1..) => {}
        _ => {}
    }
    let _: &Struct = match &Rc::new(Struct) {
        box x => x,
        _ => unreachable!(),
    };
    let _: &[Struct] = match &Rc::new(vec![Struct]) {
        box (box x) => x,
        _ => unreachable!(),
    };
}
