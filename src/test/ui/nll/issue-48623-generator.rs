// run-pass
#![allow(path_statements)]
#![allow(dead_code)]

#![feature(generators, generator_trait)]

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

fn reborrow_from_generator(r: &mut ()) {
    let d = WithDrop;
    move || { d; yield; &mut *r };
}

fn main() {}
