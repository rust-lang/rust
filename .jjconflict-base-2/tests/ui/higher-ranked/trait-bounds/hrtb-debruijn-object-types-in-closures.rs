//@ run-pass
#![allow(dead_code)]

trait Typer<'tcx> {
    fn method(&self, data: &'tcx isize) -> &'tcx isize { data }
    fn dummy(&self) { }
}

fn g<F>(_: F) where F: FnOnce(&dyn Typer) {}

fn h() {
    g(|typer| typer.dummy())
}

fn main() { }
