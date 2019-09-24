// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
use std::rc::Rc;

fn test1() -> Rc<dyn for<'a> Fn(&'a usize) + 'static> {
    if let Some(_) = Some(1) {
        loop{}
    } else {
        loop{}
    }
}

fn test2() -> *mut (dyn for<'a> Fn(&'a usize) + 'static) {
    if let Some(_) = Some(1) {
        loop{}
    } else {
        loop{}
    }
}

fn main() {}
