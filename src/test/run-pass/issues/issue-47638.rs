// run-pass
#![allow(unused_variables)]
fn id<'c, 'b>(f: &'c &'b Fn(&i32)) -> &'c &'b Fn(&'static i32) {
    f
}

fn main() {
    let f: &Fn(&i32) = &|x| {};
    id(&f);
}
