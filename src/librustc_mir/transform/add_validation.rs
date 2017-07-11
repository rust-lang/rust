// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass adds validation calls (AcquireValid, ReleaseValid) where appropriate.
//! It has to be run really early, before transformations like inlining, because
//! introducing these calls *adds* UB -- so, conceptually, this pass is actually part
//! of MIR building, and only after this pass we think of the program has having the
//! normal MIR semantics.

use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource};

pub struct AddValidation;

impl MirPass for AddValidation {
    fn run_pass<'a, 'tcx>(&self,
                          _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          _: MirSource,
                          mir: &mut Mir<'tcx>) {
        // Add an AcquireValid at the beginning of the start block
        if mir.arg_count > 0 {
            let acquire_stmt = Statement {
                source_info: SourceInfo {
                    scope: ARGUMENT_VISIBILITY_SCOPE,
                    span: mir.span,
                },
                kind: StatementKind::Validate(ValidationOp::Acquire,
                    // Skip return value, go over all the arguments
                    mir.local_decls.iter_enumerated().skip(1).take(mir.arg_count)
                    .map(|(local, local_decl)| (local_decl.ty, Lvalue::Local(local))).collect())
            };
            mir.basic_blocks_mut()[START_BLOCK].statements.insert(0, acquire_stmt);
        }
    }
}
