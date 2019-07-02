// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
macro_rules! expr { () => { () } }

enum A {}

impl A {
    const A: () = expr!();
}

fn main() {}
