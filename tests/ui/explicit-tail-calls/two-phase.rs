// regression test for <https://github.com/rust-lang/rust/issues/112788>.
// this test used to ICE because we tried to run drop glue of `x`
// if dropping `_y` (happening at the `become` site) panicked and caused an unwind.
//
//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn f(x: &mut ()) {
    let _y = String::new();
    become f(x);
}

fn main() {}
