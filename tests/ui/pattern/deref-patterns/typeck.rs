//@ check-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::rc::Rc;

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
    // FIXME(deref_patterns): fails to typecheck because `"foo"` has type &str but deref creates a
    // place of type `str`.
    // match "foo".to_string() {
    //     box "foo" => {}
    //     _ => {}
    // }
}
