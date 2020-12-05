// run-pass

#![feature(box_syntax)]

use std::collections::HashMap;

trait Graph<Node, Edge> {
    fn f(&self, _: Edge);
    fn g(&self, _: Node);
}

impl<E> Graph<isize, E> for HashMap<isize, isize> {
    fn f(&self, _e: E) {
        panic!();
    }
    fn g(&self, _e: isize) {
        panic!();
    }
}

pub fn main() {
    let g : Box<HashMap<isize,isize>> = box HashMap::new();
    let _g2 : Box<dyn Graph<isize,isize>> = g as Box<dyn Graph<isize,isize>>;
}
