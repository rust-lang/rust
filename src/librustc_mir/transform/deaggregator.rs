// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc_data_structures::indexed_vec::Idx;
use rustc::ty::VariantKind;

pub struct Deaggregator;

impl Pass for Deaggregator {}

impl<'tcx> MirPass<'tcx> for Deaggregator {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    source: MirSource, mir: &mut Mir<'tcx>) {
        let node_id = source.item_id();
        let node_path = tcx.item_path_str(tcx.map.local_def_id(node_id));
        debug!("running on: {:?}", node_path);
        // we only run when mir_opt_level > 1
        match tcx.sess.opts.debugging_opts.mir_opt_level {
            Some(0) |
            Some(1) |
            None => { return; },
            _ => {}
        };
        if let MirSource::Fn(_) = source {} else { return; }

        let mut curr: usize = 0;
        for bb in mir.basic_blocks_mut() {
            while let Some(idx) = get_aggregate_statement(curr, &bb.statements) {
                // do the replacement
                debug!("removing statement {:?}", idx);
                let src_info = bb.statements[idx].source_info;
                let mut suffix_stmts = bb.statements.split_off(idx);
                let orig_stmt = suffix_stmts.remove(0);
                let StatementKind::Assign(ref lhs, ref rhs) = orig_stmt.kind;
                if let &Rvalue::Aggregate(ref agg_kind, ref operands) = rhs {
                    if let &AggregateKind::Adt(adt_def, variant, substs) = agg_kind {
                        let n = bb.statements.len();
                        bb.statements.reserve(n + operands.len() + suffix_stmts.len());
                        for (i, op) in operands.iter().enumerate() {
                            let ref variant_def = adt_def.variants[variant];
                            let ty = variant_def.fields[variant].ty(tcx, substs);
                            let rhs = Rvalue::Use(op.clone());

                            // since we don't handle enums, we don't need a cast
                            let lhs_cast = lhs.clone();

                            // if we handled enums:
                            // let lhs_cast = if adt_def.variants.len() > 1 {
                            //     Lvalue::Projection(Box::new(LvalueProjection {
                            //         base: ai.lhs.clone(),
                            //         elem: ProjectionElem::Downcast(ai.adt_def, ai.variant),
                            //     }))
                            // } else {
                            //     lhs_cast
                            // };

                            let lhs_proj = Lvalue::Projection(Box::new(LvalueProjection {
                                base: lhs_cast,
                                elem: ProjectionElem::Field(Field::new(i), ty),
                            }));
                            let new_statement = Statement {
                                source_info: src_info,
                                kind: StatementKind::Assign(lhs_proj, rhs),
                            };
                            debug!("inserting: {:?} @ {:?}", new_statement, idx + i);
                            bb.statements.push(new_statement);
                        }
                        curr = bb.statements.len();
                        bb.statements.extend(suffix_stmts);
                    }
                }
            }
        }
    }
}

fn get_aggregate_statement<'a, 'tcx, 'b>(curr: usize,
                                         statements: &Vec<Statement<'tcx>>)
                                         -> Option<usize> {
    for i in curr..statements.len() {
        let ref statement = statements[i];
        let StatementKind::Assign(_, ref rhs) = statement.kind;
        if let &Rvalue::Aggregate(ref kind, ref operands) = rhs {
            if let &AggregateKind::Adt(adt_def, variant, _) = kind {
                if operands.len() > 0 { // don't deaggregate ()
                    if adt_def.variants.len() > 1 {
                        // only deaggrate structs for now
                        continue;
                    }
                    debug!("getting variant {:?}", variant);
                    debug!("for adt_def {:?}", adt_def);
                    let variant_def = &adt_def.variants[variant];
                    if variant_def.kind == VariantKind::Struct {
                        return Some(i);
                    }
                }
            }
        }
    };
    None
}
