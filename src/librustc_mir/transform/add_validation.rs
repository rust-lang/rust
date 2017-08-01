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

use syntax_pos::Span;
use syntax::ast::NodeId;
use rustc::ty::{self, TyCtxt, RegionKind};
use rustc::hir;
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource};
use rustc::middle::region::CodeExtent;

pub struct AddValidation;

/// Determine the "context" of the lval: Mutability and region.
fn lval_context<'a, 'tcx, D>(
    lval: &Lvalue<'tcx>,
    local_decls: &D,
    tcx: TyCtxt<'a, 'tcx, 'tcx>
) -> (Option<CodeExtent>, hir::Mutability)
    where D: HasLocalDecls<'tcx>
{
    use rustc::mir::Lvalue::*;

    match *lval {
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
                        ty::TyRef(re, tam) => {
                            let re = match re {
                                &RegionKind::ReScope(ce) => Some(ce),
                                &RegionKind::ReErased =>
                                    bug!("AddValidation pass must be run before erasing lifetimes"),
                                _ => None
                            };
                            (re, tam.mutbl)
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
                        let base_context = lval_context(&proj.base, local_decls, tcx);
                        // The region of the outermost Deref is always most restrictive.
                        let re = context.0.or(base_context.0);
                        let mutbl = context.1.and(base_context.1);
                        (re, mutbl)
                    }

                }
                _ => lval_context(&proj.base, local_decls, tcx),
            }
        }
    }
}

/// Check if this function contains an unsafe block or is an unsafe function.
fn fn_contains_unsafe<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource) -> bool {
    use rustc::hir::intravisit::{self, Visitor};
    use rustc::hir::map::Node;

    fn block_is_unsafe(block: &hir::Block) -> bool {
        use rustc::hir::BlockCheckMode::*;

        match block.rules {
            UnsafeBlock(_) | PushUnsafeBlock(_) => true,
            // For PopUnsafeBlock, we don't actually know -- but we will always also check all
            // parent blocks, so we can safely declare the PopUnsafeBlock to not be unsafe.
            DefaultBlock | PopUnsafeBlock(_) => false,
        }
    }

    let fn_node_id = match src {
        MirSource::Fn(node_id) => node_id,
        _ => return false, // only functions can have unsafe
    };

    struct FindUnsafe<'b, 'tcx> where 'tcx : 'b {
        map: &'b hir::map::Map<'tcx>,
        found_unsafe: bool,
    }
    let mut finder = FindUnsafe { map: &tcx.hir, found_unsafe: false };
    // Run the visitor on the NodeId we got.  Seems like there is no uniform way to do that.
    match tcx.hir.find(fn_node_id) {
        Some(Node::NodeItem(item)) => finder.visit_item(item),
        Some(Node::NodeImplItem(item)) => finder.visit_impl_item(item),
        Some(Node::NodeExpr(item)) => {
            // This is a closure.
            // We also have to walk up the parents and check that there is no unsafe block
            // there.
            let mut cur = fn_node_id;
            loop {
                // Go further upwards.
                let parent = tcx.hir.get_parent_node(cur);
                if cur == parent {
                    break;
                }
                cur = parent;
                // Check if this is a a block
                match tcx.hir.find(cur) {
                    Some(Node::NodeExpr(&hir::Expr { node: hir::ExprBlock(ref block), ..})) => {
                        if block_is_unsafe(&*block) {
                            // Found an unsafe block, we can bail out here.
                            return true;
                        }
                    }
                    _ => {},
                }
            }
            // Finally, visit the closure itself.
            finder.visit_expr(item);
        }
        Some(Node::NodeStructCtor(_)) => {
            // Auto-generated tuple struct ctor.  Cannot contain unsafe code.
            return false;
        },
        Some(_) | None =>
            bug!("Expected function, method or closure, found {}",
                 tcx.hir.node_to_string(fn_node_id)),
    };

    impl<'b, 'tcx> Visitor<'tcx> for FindUnsafe<'b, 'tcx> {
        fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
            intravisit::NestedVisitorMap::OnlyBodies(self.map)
        }

        fn visit_fn(&mut self, fk: intravisit::FnKind<'tcx>, fd: &'tcx hir::FnDecl,
                    b: hir::BodyId, s: Span, id: NodeId)
        {
            assert!(!self.found_unsafe, "We should never see a fn when we already saw unsafe");
            let is_unsafe = match fk {
                intravisit::FnKind::ItemFn(_, _, unsafety, ..) => unsafety == hir::Unsafety::Unsafe,
                intravisit::FnKind::Method(_, sig, ..) => sig.unsafety == hir::Unsafety::Unsafe,
                intravisit::FnKind::Closure(_) => false,
            };
            if is_unsafe {
                // This is unsafe, and we are done.
                self.found_unsafe = true;
            } else {
                // Go on searching.
                intravisit::walk_fn(self, fk, fd, b, s, id)
            }
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

        // Convert an lvalue to a validation operand.
        let lval_to_operand = |lval: Lvalue<'tcx>| -> ValidationOperand<'tcx, Lvalue<'tcx>> {
            let (re, mutbl) = lval_context(&lval, &local_decls, tcx);
            let ty = lval.ty(&local_decls, tcx).to_ty(tcx);
            ValidationOperand { lval, ty, re, mutbl }
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
                scope: ARGUMENT_VISIBILITY_SCOPE,
                span: mir.span, // FIXME: Consider using just the span covering the function
                                // argument declaration.
            };
            // Gather all arguments, skip return value.
            let operands = mir.local_decls.iter_enumerated().skip(1).take(mir.arg_count)
                    .map(|(local, _)| lval_to_operand(Lvalue::Local(local))).collect();
            emit_acquire(&mut mir.basic_blocks_mut()[START_BLOCK], source_info, operands);
        }

        // PART 2
        // Add ReleaseValid/AcquireValid around function call terminators.  We don't use a visitor
        // because we need to access the block that a Call jumps to.
        let mut returns : Vec<(SourceInfo, Lvalue<'tcx>, BasicBlock)> = Vec::new();
        for block_data in mir.basic_blocks_mut() {
            match block_data.terminator {
                Some(Terminator { kind: TerminatorKind::Call { ref args, ref destination, .. },
                                  source_info }) => {
                    // Before the call: Release all arguments
                    if !restricted_validation {
                        let release_stmt = Statement {
                            source_info,
                            kind: StatementKind::Validate(ValidationOp::Release,
                                args.iter().filter_map(|op| {
                                    match op {
                                        &Operand::Consume(ref lval) =>
                                            Some(lval_to_operand(lval.clone())),
                                        &Operand::Constant(..) => { None },
                                    }
                                }).collect())
                        };
                        block_data.statements.push(release_stmt);
                    }
                    // Remember the return destination for later
                    if let &Some(ref destination) = destination {
                        returns.push((source_info, destination.0.clone(), destination.1));
                    }
                }
                Some(Terminator { kind: TerminatorKind::Drop { location: ref lval, .. },
                                  source_info }) |
                Some(Terminator { kind: TerminatorKind::DropAndReplace { location: ref lval, .. },
                                  source_info }) => {
                    // Before the call: Release all arguments
                    if !restricted_validation {
                        let release_stmt = Statement {
                            source_info,
                            kind: StatementKind::Validate(ValidationOp::Release,
                                    vec![lval_to_operand(lval.clone())]),
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
        for (source_info, dest_lval, dest_block) in returns {
            emit_acquire(
                &mut mir.basic_blocks_mut()[dest_block],
                source_info,
                vec![lval_to_operand(dest_lval)]
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
                        let (dest_lval, re, src_lval) = match block_data.statements[i].kind {
                            StatementKind::Assign(ref dest_lval,
                                                  Rvalue::Ref(re, _, ref src_lval)) => {
                                (dest_lval.clone(), re, src_lval.clone())
                            },
                            _ => bug!("We already matched this."),
                        };
                        // So this is a ref, and we got all the data we wanted.
                        // Do an acquire of the result -- but only what it points to, so add a Deref
                        // projection.
                        let dest_lval = Projection { base: dest_lval, elem: ProjectionElem::Deref };
                        let dest_lval = Lvalue::Projection(Box::new(dest_lval));
                        let acquire_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(ValidationOp::Acquire,
                                    vec![lval_to_operand(dest_lval)]),
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
                            kind: StatementKind::Validate(op, vec![lval_to_operand(src_lval)]),
                        };
                        block_data.statements.insert(i, release_stmt);
                    }
                    // Casts can change what validation does (e.g. unsizing)
                    StatementKind::Assign(_, Rvalue::Cast(kind, Operand::Consume(_), _))
                        if kind != CastKind::Misc =>
                    {
                        // Due to a lack of NLL; we can't capture anything directly here.
                        // Instead, we have to re-match and clone there.
                        let (dest_lval, src_lval) = match block_data.statements[i].kind {
                            StatementKind::Assign(ref dest_lval,
                                    Rvalue::Cast(_, Operand::Consume(ref src_lval), _)) =>
                            {
                                (dest_lval.clone(), src_lval.clone())
                            },
                            _ => bug!("We already matched this."),
                        };

                        // Acquire of the result
                        let acquire_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(ValidationOp::Acquire,
                                    vec![lval_to_operand(dest_lval)]),
                        };
                        block_data.statements.insert(i+1, acquire_stmt);

                        // Release of the input
                        let release_stmt = Statement {
                            source_info: block_data.statements[i].source_info,
                            kind: StatementKind::Validate(ValidationOp::Release,
                                                            vec![lval_to_operand(src_lval)]),
                        };
                        block_data.statements.insert(i, release_stmt);
                    }
                    _ => {},
                }
            }
        }
    }
}
