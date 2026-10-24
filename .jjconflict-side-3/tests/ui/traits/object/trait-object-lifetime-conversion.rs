// https://github.com/rust-lang/rust/issues/47638
//@ run-pass
#![allow(unused_variables)]
fn id<'c, 'b>(f: &'c &'b dyn Fn(&i32)) -> &'c &'b dyn Fn(&'static i32) {
    f
}

fn main() {
    let f: &dyn Fn(&i32) = &|x| {};
    id(&f);
}
