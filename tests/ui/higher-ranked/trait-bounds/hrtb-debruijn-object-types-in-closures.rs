// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

trait Typer<'tcx> {
    fn method(&self, data: &'tcx isize) -> &'tcx isize { data }
    fn dummy(&self) { }
}

fn g<F>(_: F) where F: FnOnce(&dyn Typer) {}

fn h() {
    g(|typer| typer.dummy())
}

fn main() { }
