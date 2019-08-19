// run-pass
#![allow(unused_must_use)]
#![allow(non_upper_case_globals)]

// pretty-expanded FIXME #23616
// ignore-emscripten no threads

use std::thread::Builder;

static generations: usize = 1024+256+128+49;

fn spawn(mut f: Box<dyn FnMut() + 'static + Send>) {
    Builder::new().stack_size(32 * 1024).spawn(move|| f());
}

fn child_no(x: usize) -> Box<dyn FnMut() + 'static + Send> {
    Box::new(move|| {
        if x < generations {
            spawn(child_no(x+1));
        }
    })
}

pub fn main() {
    spawn(child_no(0));
}
