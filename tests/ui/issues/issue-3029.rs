// run-fail
// error-pattern:so long
// ignore-emscripten no processes

#![allow(unreachable_code)]

fn main() {
    let mut x = Vec::new();
    let y = vec![3];
    panic!("so long");
    x.extend(y.into_iter());
}
