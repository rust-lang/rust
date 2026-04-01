//@ run-pass
//@ revisions: mir codegen
//@[mir] compile-flags: -Zmir-opt-level=3
//@[codegen] compile-flags: -Zmir-opt-level=0

// A regression test for #107205

const X: for<'b> fn(&'b ()) = |&()| ();
fn main() {
    let dyn_debug = Box::new(X) as Box<fn(&'static ())> as Box<dyn Send>;
    drop(dyn_debug)
}
