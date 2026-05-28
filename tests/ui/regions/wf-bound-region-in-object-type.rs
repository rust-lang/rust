//@ run-pass

#![allow(dead_code)]
#![allow(unused_variables)]
// Test that the `wf` checker properly handles bound regions in object
// types. Compiling this code used to trigger an ICE.


pub struct Context<'tcx> {
    vec: &'tcx Vec<isize>
}

pub type Cmd<'a> = &'a isize;

pub type DecodeInlinedItem<'a> =
    Box<dyn for<'tcx> FnMut(Cmd, &Context<'tcx>) -> Result<&'tcx isize, ()> + 'a>;

fn foo(d: DecodeInlinedItem) {
}

fn main() { }
