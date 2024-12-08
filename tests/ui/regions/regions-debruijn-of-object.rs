//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_camel_case_types)]


struct ctxt<'tcx> {
    x: &'tcx i32
}

trait AstConv<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ctxt<'tcx>;
}

fn foo(conv: &dyn AstConv) { }

fn bar<'tcx>(conv: &dyn AstConv<'tcx>) {
    foo(conv)
}

fn main() { }
