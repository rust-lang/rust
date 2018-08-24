// error-pattern: thread 'main' panicked at 'explicit panic'

#![feature(nll)]

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
