// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we handle binder levels in object types correctly.
// Initially, the reference to `'tcx` in the object type
// `&Typer<'tcx>` was getting an incorrect binder level, yielding
// weird compilation ICEs and so forth.

trait Typer<'tcx> {
    fn method(&self, data: &'tcx int) -> &'tcx int { data }
}

struct Tcx<'tcx> {
    fields: &'tcx int
}

impl<'tcx> Typer<'tcx> for Tcx<'tcx> {
}

fn g<'tcx>(typer: &Typer<'tcx>) {
}

fn check_static_type<'x>(tcx: &Tcx<'x>) {
    g(tcx)
}

fn main() { }
