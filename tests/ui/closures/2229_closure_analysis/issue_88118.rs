// Regression test for #88118. Used to ICE.
//
// check-pass

#![allow(incomplete_features)]
#![feature(capture_disjoint_fields)]

fn foo<MsU>(handler: impl FnOnce() -> MsU + Clone + 'static) {
    Box::new(move |value| {
        (|_| handler.clone()())(value);
        None
    }) as Box<dyn Fn(i32) -> Option<i32>>;
}

fn main() {}
