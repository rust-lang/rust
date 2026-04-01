//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
struct UsizeRef<'a> {
    a: &'a usize
}

type RefTo = Box<dyn for<'r> Fn(&'r Vec<usize>) -> UsizeRef<'r>>;

fn ref_to<'a>(vec: &'a Vec<usize>) -> UsizeRef<'a> {
    UsizeRef{ a: &vec[0]}
}

fn main() {
    // Regression test: this was causing ICEs; it should compile.
    let a: RefTo = Box::new(|vec: &Vec<usize>| {
        UsizeRef{ a: &vec[0] }
    });
}
