//@ run-pass
#![allow(path_statements)]
#![allow(dead_code)]

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

fn reborrow_from_closure(r: &mut ()) -> &mut () {
    let d = WithDrop;
    (move || { d; &mut *r })()
}

fn main() {}
