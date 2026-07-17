// https://github.com/rust-lang/rust/issues/47638
//
// Converting `dyn for<'a> Fn(&'a i32)` to `dyn Fn(&'static i32)` is no longer
// subtyping (higher-ranked subtyping has been removed) but a coercion. Here the
// trait objects sit behind two references, so there is no coercion site and the
// conversion is correctly rejected.
#![allow(unused_variables)]
fn id<'c, 'b>(f: &'c &'b dyn Fn(&i32)) -> &'c &'b dyn Fn(&'static i32) {
    f
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn main() {
    let f: &dyn Fn(&i32) = &|x| {};
    id(&f);
}
