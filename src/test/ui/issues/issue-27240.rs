// run-pass
#![allow(unused_assignments)]
#![allow(unused_variables)]
use std::fmt;
struct NoisyDrop<T: fmt::Debug>(#[allow(unused_tuple_struct_fields)] T);
impl<T: fmt::Debug> Drop for NoisyDrop<T> {
    fn drop(&mut self) {}
}

struct Bar<T: fmt::Debug>(#[allow(unused_tuple_struct_fields)] [*const NoisyDrop<T>; 2]);

fn fine() {
    let (u,b);
    u = vec![43];
    b = Bar([&NoisyDrop(&u), &NoisyDrop(&u)]);
}

#[allow(unused_tuple_struct_fields)]
struct Bar2<T: fmt::Debug>(*const NoisyDrop<T>, *const NoisyDrop<T>);

fn lolwut() {
    let (u,v);
    u = vec![43];
    v = Bar2(&NoisyDrop(&u), &NoisyDrop(&u));
}

fn main() { fine(); lolwut() }
