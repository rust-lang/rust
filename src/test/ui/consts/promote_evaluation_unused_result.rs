//compile-pass

#![feature(nll)]

fn main() {

    let _: &'static usize = &(loop {}, 1).1;
}
