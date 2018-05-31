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

use rustc::ty::{self, TyCtxt, RegionKind};
use rustc::hir;
use rustc::mir::*;
use rustc::middle::region;
use transform::{MirPass, MirSource};

pub struct AddValidation;

/// Determine the "context" of the place: Mutability and region.
fn place_context<'a, 'tcx, D>(
    place: &Place<'tcx>,
    local_decls: &D,
    tcx: TyCtxt<'a, 'tcx, 'tcx>
) -> (Option<region::Scope>, hir::Mutability)
    where D: HasLocalDecls<'tcx>
{
    use rustc::mir::Place::*;

    match *place {
        Local { .. } => (None, hir::MutMutable),
        Static(_) => (None, hir::MutImmutable),
        Projection(ref proj) => {
            match proj.elem {
                ProjectionElem::Deref => {
                    // Computing the inside the recursion makes this quadratic.
                    // We don't expect deep paths though.
                    let ty = proj.base.ty(local_decls, tcx).to_ty(tcx);
                    // A Deref projection may restrict the context, this depends on the type
                    // being deref'd.
                    let context = match ty.sty {
                        ty::TyRef(re, _, mutbl) => {
                            let re = match re {
                                &RegionKind::ReScope(ce) => Some(ce),
                                &RegionKind::ReErased =>
                                    bug!("AddValidation pass must be run before erasing lifetimes"),
                                _ => None
                            };
                            (re, mutbl)
                        }
                        ty::TyRawPtr(_) =>
                            // There is no guarantee behind even a mutable raw pointer,
                            // no write locks are acquired there, so we also don't want to
                            // release any.
                            (None, hir::MutImmutable),
                        ty::TyAdt(adt, _) if adt.is_box() => (None, hir::MutMutable),
                        _ => bug!("Deref on a non-pointer type {:?}", ty),
                    };
                    // "Intersect" this restriction with proj.base.
                    if let (Some(_), hir::MutImmutable) = context {
                        // This is already as restricted as it gets, no need to even recurse
                        context
                    } else {
                        let base_context = place_context(&proj.base, local_decls, tcx);
                        // The region of the outermost Deref is always most restrictive.
                        let re = context.0.or(base_context.0);
                        let mutbl = context.1.and(base_context.1);
                        (re, mutbl)
                    }

                }
                _ => place_context(&proj.base, local_decls, tcx),
            }
        }
    }
}

/// Check if this function contains an unsafe block or is an unsafe function.
fn fn_contains_unsafe<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource) -> bool {
    use rustc::hir::intravisit::{self, Visitor, FnKind};
    use rustc::hir::map::blocks::FnLikeNode;
    use rustc::hir::map::Node;

    /// Decide if this is an unsafe block
    fn block_is_unsafe(block: &hir::Block) -> bool {
        use rustc::hir::BlockCheckMode::*;

        match block.rules {
            UnsafeBlock(_) | PushUnsafeBlock(_) => true,
            // For PopUnsafeBlock, we don't actually know -- but we will always also check all
            // parent blocks, so we can safely declare the PopUnsafeBlock to not be unsafe.
            DefaultBlock | PopUnsafeBlock(_) => false,
        }
    }

    /// Decide if this FnLike is a closure
    fn fn_is_closure<'a>(fn_like: FnLikeNode<'a>) -> bool {
        match fn_like.kind() {
            FnKind::Closure(_) => true,
            FnKind::Method(..) | FnKind::ItemFn(..) => false,
        }
    }

    let node_id = tcx.hir.as_local_node_id(src.def_id).unwrap();
    let fn_like = match tcx.hir.body_owner_kind(node_id) {
        hir::BodyOwnerKind::Fn => {
            match FnLikeNode::from_node(tcx.hir.get(node_id)) {
                Some(fn_like) => fn_like,
                None => return false, // e.g. struct ctor shims -- such auto-generated code cannot
                                      // contain unsafe.
            }
        },
        _ => return false, // only functions can have unsafe
    };

    // Test if the function is marked unsafe.
    if fn_like.unsafety() == hir::Unsafety::Unsafe {
        return true;
    }

    // For closures, we need to walk up the parents and see if we are inside an unsafe fn or
    // unsafe block.
    if fn_is_closure(fn_like) {
        let mut cur = fn_like.id();
        loop {
            // Go further upwards.
            cur = tcx.hir.get_parent_node(cur);
            let node = tcx.hir.get(cur);
            // Check if this is an unsafe function
            if let Some(fn_like) = FnLikeNode::from_node(node) {
                if !fn_is_closure(fn_like) {
                    if fn_like.unsafety() == hir::Unsafety::Unsafe {
                        return true;
                    }
                }
            }
            // Check if this is an unsafe block, or an item
            match node {
                Node::NodeExpr(&hir::Expr { node: hir::ExprBlock(ref block, _), ..}) => {
                    if block_is_unsafe(&*block) {
                        // Found an unsafe block, we can bail out here.
                        return true;
                    }
                }
                Node::NodeItem(..) => {
                    // No walking up beyond items.  This makes sure the loop always terminates.
                    break;
                }
                _ => {},
            }
        }
    }

    // Visit the entire body of the function and check for unsafe blocks in there
    struct FindUnsafe {
        found_unsafe: bool,
    }
    let mut finder = FindUnsafe { found_unsafe: false };
    // Run the visitor on the NodeId we got.  Seems like there is no uniform way to do that.
    finder.visit_body(tcx.hir.body(fn_like.body()));

    impl<'tcx> Visitor<'tcx> for FindUnsafe {
        fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
            intravisit::NestedVisitorMap::None
        }

        fn visit_block(&mut self, b: &'tcx hir::Block) {
            if self.found_unsafe { return; } // short-circuit

            if block_is_unsafe(b) {
                // We found an unsafe block.  We can stop searching.
                self.found_unsafe = true;
            } else {
                // No unsafe block here, go on searching.
                intravisit::walk_block(self, b);
            }
        }
    }

    finder.found_unsafe
}

impl MirPass for AddValidation {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          src: MirSource,
                          mir: &mut Mir<'tcx>)
    {
        let emit_validate = tcx.sess.opts.debugging_opts.mir_emit_validate;
        if emit_validate == 0 {
            return;
        }
        let restricted_validation = emit_validate == 1 && fn_contains_unsafe(tcx, src);
        let local_decls = mir.local_decls.clone(); // FIXME: Find a way to get rid of this clone.

        // Convert a place to a validation operand.
        let place_to_operand = |place: Place<'tcx>| -> ValidationOperand<'tcx, Place<'tcx>> {
            let (re, mutbl) = place_context(&place, &local_decls, tcx);
            let ty = place.ty(&local_decls, tcx).to_ty(tcx);
            ValidationOperand { place, ty, re, mutbl }
        };

        // Emit an Acquire at the beginning of the given block.  If we are in restricted emission
        // mode (mir_emit_validate=1), also emit a Release immediately after the Acquire.
        let emit_acquire = |block: &mut BasicBlockData<'tcx>, source_info, operands: Vec<_>| {
            if operands.len() == 0 {
                return; // Nothing to do
            }
            // Emit the release first, to avoid cloning if we do not emit it
            if restricted_validation {
                let release_stmt = Statement {
                    source_info,
                    kind: StatementKind::Validate(ValidationOp::Release, operands.clone()),
                };
                block.statements.insert(0, release_stmt);
            }
            // Now, the acquire
            let acquire_stmt = Statement {
                source_info,
                kind: StatementKind::Validate(ValidationOp::Acquire, operands),
            };
            block.statements.insert(0, acquire_stmt);
        };

        // PART 1
        // Add an AcquireValid at the beginning of the start block.
        {
            let source_info = SourceInfo {
                scope: OUTERMOST_SOURCE_SCOPE,
                span: mir.span, // FIXME: Consider using just the span covering the function
                                // argument declaration.
            };
            // Gather all arguments, skip return value.
            let operands = mir.local_decls.iter_enumerated().skip(1).take(mir.arg_count)
                    .map(|(local, _)| place_to_operand(Place::Local(local))).collect();
            emit_acquire(&mut mir.basic_blocks_mut()[START_BLOCK], source_info, operands);
        }

        // PART 2
        // Add ReleaseValid/AcquireValid around function call terminators.  We don't use a visitor
        // because we need to access the block that a Call jumps to.
        let mut returns : Vec<(SourceInfo, Place<'tcx>, BasicBlock)> = Vec::new();
        for block_data in mir.basic_blocks_mut() {
            match block_data.terminator {
                Some(Terminator { kind: TerminatorKind::Call { ref args, ref destination, .. },
                                  source_info }) => {
                    // Before the call: Release all arguments *and* the return value.
                    // The callee may write into the return value!  Note that this relies
                    // on "release of uninitialized" to be a NOP.
                    if !restricted_validation {
                        let release_stmt = Statement {
                            source_info,
                            kind: StatementKind::Validate(ValidationOp::Release,
                                destination.iter().map(|dest| place_to_operand(dest.0.clone()))
                                .chain(
                                    args.iter().filter_map(|op| {
                                        match op {
                                            &Operand::Copy(ref place) |
                                            &Operand::Move(ref place) =>
                                                Some(place_to_operand(place.clone())),
                                            &Operand::Constant(..) => { None },
                                        }
                                    })
                                ).collect())
                        };
                        block_data.statements.push(release_stmt);
                    }
                    // Remember the return destination for later
                    if let &Some(ref destination) = destination {
                        returns.push((source_info, destination.0.clone(), destination.1));
                    }
                }
                Some(Terminator { kind: TerminatorKind::Drop { location: ref place, .. },
                                  source_info }) |
                Some(Terminator { kind: TerminatorKind::DropAndReplace { location: ref place, .. },
                                  source_info }) => {
                    // Before the call: Release all arguments
                    if !restricted_validation {
                        let release_stmt = Statement {
                            source_info,
                            kind: StatementKind::Validate(ValidationOp::Release,
                                    vec![place_to_operand(place.clone())]),
                        };
                        block_data.statements.push(release_stmt);
                    }
                    // drop doesn't return anything, so we need no acquire.
                }
                _ => {
                    // Not a block ending in a Call -> ignore.
                }
            }
        }
        // Now we go over the returns we collected to acquire the return values.
        for (source_info, dest_place, dest_block) in returns {
            emit_acquire(
                &mut mir.basic_blocks_mut()[dest_block],
                source_info,
                vec![place_to_operand(dest_place)]
            );
        }

        if restricted_validation {
            // No part 3 for us.
            return;
        }

        // PART 3
        // Add ReleaseValid/AcquireValid around Ref and Cast.  Again an iterator does not seem very
        // suited as we need to add new statements before and after each Ref.
        for block_data in mir.basic_blocks_mut() {
            // We want to insert statements around Ref commands as we iterate.  To this end, we
            // iterate backwards using indices.
            for i in (0..block_data.statements.len()).rev() {
                match block_data.statements[i].kind {
                    // When the borrow of this ref expires, we need to recover validation.
                    StatementKind::Assign(_, Rvalue::Ref(_, _, _)) => {
                        // Due to a lack of NLL; we can't capture anything directly here.
                        // Instead, we have to re-match and clone there.
                        let (dest_place, re, src_place) = match block_data.statements[i].kind {
                            StatementKind::Assign(ref dest_place,
                                                  Rvalue::Ref(re, _, ref src_place)) => {
                                (dest_place.clone(), re, src_place.clone())
                            },
                            _ => bug!("We already matched this."),
                        };
                        // So this is a ref, and we got all the data we wanted.
                        // Do an acquire of the result -- but only what it points to, so add a Deref
                        // projection.
                        let acquire_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(ValidationOp::Acquire,
                                    vec![place_to_operand(dest_place.deref())]),
                        };
                        block_data.statements.insert(i+1, acquire_stmt);

                        // The source is released until the region of the borrow ends.
                        let op = match re {
                            &RegionKind::ReScope(ce) => ValidationOp::Suspend(ce),
                            &RegionKind::ReErased =>
                                bug!("AddValidation pass must be run before erasing lifetimes"),
                            _ => ValidationOp::Release,
                        };
                        let release_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(op, vec![place_to_operand(src_place)]),
                        };
                        block_data.statements.insert(i, release_stmt);
                    }
                    // Casts can change what validation does (e.g. unsizing)
                    StatementKind::Assign(_, Rvalue::Cast(kind, Operand::Copy(_), _)) |
                    StatementKind::Assign(_, Rvalue::Cast(kind, Operand::Move(_), _))
                        if kind != CastKind::Misc =>
                    {
                        // Due to a lack of NLL; we can't capture anything directly here.
                        // Instead, we have to re-match and clone there.
                        let (dest_place, src_place) = match block_data.statements[i].kind {
                            StatementKind::Assign(ref dest_place,
                                    Rvalue::Cast(_, Operand::Copy(ref src_place), _)) |
                            StatementKind::Assign(ref dest_place,
                                    Rvalue::Cast(_, Operand::Move(ref src_place), _)) =>
                            {
                                (dest_place.clone(), src_place.clone())
                            },
                            _ => bug!("We already matched this."),
                        };

                        // Acquire of the result
                        let acquire_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(ValidationOp::Acquire,
                                    vec![place_to_operand(dest_place)]),
                        };
                        block_data.statements.insert(i+1, acquire_stmt);

                        // Release of the input
                        let release_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(ValidationOp::Release,
                                                            vec![place_to_operand(src_place)]),
                        };
                        block_data.statements.insert(i, release_stmt);
                    }
                    _ => {},
                }
            }
        }
    }
}
