// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// This test ensures that each pointer type `P<X>` is covariant in `X`.

use std::rc::Rc;
use std::sync::Arc;

fn a<'r>(x: Box<&'static str>) -> Box<&'r str> {
    x
}

fn b<'r, 'w>(x: &'w &'static str) -> &'w &'r str {
    x
}

fn c<'r>(x: Arc<&'static str>) -> Arc<&'r str> {
    x
}

fn d<'r>(x: Rc<&'static str>) -> Rc<&'r str> {
    x
}

fn main() {}
