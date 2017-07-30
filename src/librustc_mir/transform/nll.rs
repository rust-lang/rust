// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TypeFoldable;
use rustc::ty::subst::Substs;
use rustc::ty::{Ty, TyCtxt, ClosureSubsts};
use rustc::mir::{Mir, Location, Rvalue, BasicBlock, Statement, StatementKind};
use rustc::mir::visit::{MutVisitor, Lookup};
use rustc::mir::transform::{MirPass, MirSource};
use rustc::infer::{self, InferCtxt};
use syntax_pos::Span;

#[allow(dead_code)]
struct NLLVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: InferCtxt<'a, 'gcx, 'tcx>,
    source: &'a Mir<'tcx>
}

impl<'a, 'gcx, 'tcx> NLLVisitor<'a, 'gcx, 'tcx> {
    pub fn new(infcx: InferCtxt<'a, 'gcx, 'tcx>, source: &'a Mir<'tcx>) -> Self {
        NLLVisitor {
            infcx: infcx,
            source: source,
        }
    }

    fn renumber_regions<T>(&self, value: &T, span: Span) -> T where T: TypeFoldable<'tcx> {
        self.infcx.tcx.fold_regions(value, &mut false, |_region, _depth| {
            self.infcx.next_region_var(infer::MiscVariable(span))
        })
    }
}

fn span_from_location<'tcx>(source: &Mir<'tcx>, location: Location) -> Span {
    source[location.block].statements[location.statement_index].source_info.span
}

impl<'a, 'gcx, 'tcx> MutVisitor<'tcx> for NLLVisitor<'a, 'gcx, 'tcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, lookup: Lookup) {
        let old_ty = *ty;
        let span = match lookup {
            Lookup::Loc(location) => span_from_location(self.source, location),
            Lookup::Src(source_info) => source_info.span,
        };
        *ty = self.renumber_regions(&old_ty, span);
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>, location: Location) {
        *substs = self.renumber_regions(&{*substs}, span_from_location(self.source, location));
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        match *rvalue {
            Rvalue::Ref(ref mut r, _, _) => {
                let old_r = *r;
                *r = self.renumber_regions(&old_r, span_from_location(self.source, location));
            }
            Rvalue::Use(..) |
            Rvalue::Repeat(..) |
            Rvalue::Len(..) |
            Rvalue::Cast(..) |
            Rvalue::BinaryOp(..) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::UnaryOp(..) |
            Rvalue::Discriminant(..) |
            Rvalue::NullaryOp(..) |
            Rvalue::Aggregate(..) => {
                // These variants don't contain regions.
            }
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_closure_substs(&mut self,
                            substs: &mut ClosureSubsts<'tcx>,
                            location: Location) {
        *substs = self.renumber_regions(substs, span_from_location(self.source, location));
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        if let StatementKind::EndRegion(_) = statement.kind {
            statement.kind = StatementKind::Nop;
        }
        self.super_statement(block, statement, location);
    }
}

// MIR Pass for non-lexical lifetimes
pub struct NLL;

impl MirPass for NLL {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource,
                          mir: &mut Mir<'tcx>) {
        if !tcx.sess.opts.debugging_opts.nll {
            return;
        }

        tcx.infer_ctxt().enter(|infcx| {
            // Clone mir so we can mutate it without disturbing the rest
            // of the compiler
            let mut renumbered_mir = mir.clone();

            // Note that we're using the passed-in mir for the visitor. This is
            // so we can lookup locations during traversal without worrying about
            // maintaing both a mutable and immutable reference to the same object
            let mut visitor = NLLVisitor::new(infcx, &mir);
            visitor.visit_mir(&mut renumbered_mir);
        })
    }
}