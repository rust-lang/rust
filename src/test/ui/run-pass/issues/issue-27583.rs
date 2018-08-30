// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #27583. Unclear how useful this will be
// going forward, since the issue in question was EXTREMELY sensitive
// to compiler internals (like the precise numbering of nodes), but
// what the hey.

#![allow(warnings)]

use std::cell::Cell;
use std::marker::PhantomData;

pub trait Delegate<'tcx> { }

pub struct InferCtxt<'a, 'tcx: 'a> {
    x: PhantomData<&'a Cell<&'tcx ()>>
}

pub struct MemCategorizationContext<'t, 'a: 't, 'tcx : 'a> {
    x: &'t InferCtxt<'a, 'tcx>,
}

pub struct ExprUseVisitor<'d, 't, 'a: 't, 'tcx:'a+'d> {
    typer: &'t InferCtxt<'a, 'tcx>,
    mc: MemCategorizationContext<'t, 'a, 'tcx>,
    delegate: &'d mut (Delegate<'tcx>+'d),
}

impl<'d,'t,'a,'tcx> ExprUseVisitor<'d,'t,'a,'tcx> {
    pub fn new(delegate: &'d mut Delegate<'tcx>,
               typer: &'t InferCtxt<'a, 'tcx>)
               -> ExprUseVisitor<'d,'t,'a,'tcx>
    {
        ExprUseVisitor {
            typer: typer,
            mc: MemCategorizationContext::new(typer),
            delegate: delegate,
        }
    }
}

impl<'t, 'a,'tcx> MemCategorizationContext<'t, 'a, 'tcx> {
    pub fn new(typer: &'t InferCtxt<'a, 'tcx>) -> MemCategorizationContext<'t, 'a, 'tcx> {
        MemCategorizationContext { x: typer }
    }
}

fn main() { }
