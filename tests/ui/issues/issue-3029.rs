// run-fail
//@error-in-other-file:so long
//@ignore-target-emscripten no processes

#![allow(unreachable_code)]

fn main() {
    let mut x = Vec::new();
    let y = vec![3];
    panic!("so long");
    x.extend(y.into_iter());
}
