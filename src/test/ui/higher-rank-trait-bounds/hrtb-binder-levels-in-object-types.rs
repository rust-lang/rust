// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that we handle binder levels in object types correctly.
// Initially, the reference to `'tcx` in the object type
// `&Typer<'tcx>` was getting an incorrect binder level, yielding
// weird compilation ICEs and so forth.

// pretty-expanded FIXME #23616

trait Typer<'tcx> {
    fn method(&self, data: &'tcx isize) -> &'tcx isize { data }
}

struct Tcx<'tcx> {
    fields: &'tcx isize
}

impl<'tcx> Typer<'tcx> for Tcx<'tcx> {
}

fn g<'tcx>(typer: &dyn Typer<'tcx>) {
}

fn check_static_type<'x>(tcx: &Tcx<'x>) {
    g(tcx)
}

fn main() { }
