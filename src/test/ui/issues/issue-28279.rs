// compile-pass
#![allow(dead_code)]
use std::rc::Rc;

fn test1() -> Rc<for<'a> Fn(&'a usize) + 'static> {
    if let Some(_) = Some(1) {
        loop{}
    } else {
        loop{}
    }
}

fn test2() -> *mut (for<'a> Fn(&'a usize) + 'static) {
    if let Some(_) = Some(1) {
        loop{}
    } else {
        loop{}
    }
}

fn main() {}
