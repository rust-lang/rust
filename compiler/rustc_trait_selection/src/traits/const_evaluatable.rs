//! Checking that constant values used in types can be successfully evaluated.
//!
//! For concrete constants, this is fairly simple as we can just try and evaluate it.
//!
//! When dealing with polymorphic constants, for example `std::mem::size_of::<T>() - 1`,
//! this is not as easy.
//!
//! In this case we try to build an abstract representation of this constant using
//! `mir_abstract_const` which can then be checked for structural equality with other
//! generic constants mentioned in the `caller_bounds` of the current environment.
use rustc_hir::def::DefKind;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::abstract_const::{Node, NodeId};
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::mir::{self, Rvalue, StatementKind, TerminatorKind};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, TyCtxt, TypeFoldable};
use rustc_session::lint;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::Span;

pub fn is_const_evaluatable<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    def: ty::WithOptConstParam<DefId>,
    substs: SubstsRef<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), ErrorHandled> {
    debug!("is_const_evaluatable({:?}, {:?})", def, substs);
    if infcx.tcx.features().const_evaluatable_checked {
        if let Some(ct) = AbstractConst::new(infcx.tcx, def, substs) {
            for pred in param_env.caller_bounds() {
                match pred.skip_binders() {
                    ty::PredicateAtom::ConstEvaluatable(b_def, b_substs) => {
                        debug!("is_const_evaluatable: caller_bound={:?}, {:?}", b_def, b_substs);
                        if b_def == def && b_substs == substs {
                            debug!("is_const_evaluatable: caller_bound ~~> ok");
                            return Ok(());
                        } else if AbstractConst::new(infcx.tcx, b_def, b_substs)
                            .map_or(false, |b_ct| try_unify(infcx.tcx, ct, b_ct))
                        {
                            debug!("is_const_evaluatable: abstract_const ~~> ok");
                            return Ok(());
                        }
                    }
                    _ => {} // don't care
                }
            }
        }
    }

    let future_compat_lint = || {
        if let Some(local_def_id) = def.did.as_local() {
            infcx.tcx.struct_span_lint_hir(
                lint::builtin::CONST_EVALUATABLE_UNCHECKED,
                infcx.tcx.hir().local_def_id_to_hir_id(local_def_id),
                span,
                |err| {
                    err.build("cannot use constants which depend on generic parameters in types")
                        .emit();
                },
            );
        }
    };

    // FIXME: We should only try to evaluate a given constant here if it is fully concrete
    // as we don't want to allow things like `[u8; std::mem::size_of::<*mut T>()]`.
    //
    // We previously did not check this, so we only emit a future compat warning if
    // const evaluation succeeds and the given constant is still polymorphic for now
    // and hopefully soon change this to an error.
    //
    // See #74595 for more details about this.
    let concrete = infcx.const_eval_resolve(param_env, def, substs, None, Some(span));

    if concrete.is_ok() && substs.has_param_types_or_consts() {
        match infcx.tcx.def_kind(def.did) {
            DefKind::AnonConst => {
                let mir_body = if let Some(def) = def.as_const_arg() {
                    infcx.tcx.optimized_mir_of_const_arg(def)
                } else {
                    infcx.tcx.optimized_mir(def.did)
                };

                if mir_body.is_polymorphic {
                    future_compat_lint();
                }
            }
            _ => future_compat_lint(),
        }
    }

    debug!(?concrete, "is_const_evaluatable");
    concrete.map(drop)
}

/// A tree representing an anonymous constant.
///
/// This is only able to represent a subset of `MIR`,
/// and should not leak any information about desugarings.
#[derive(Clone, Copy)]
pub struct AbstractConst<'tcx> {
    // FIXME: Consider adding something like `IndexSlice`
    // and use this here.
    inner: &'tcx [Node<'tcx>],
    substs: SubstsRef<'tcx>,
}

impl AbstractConst<'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        def: ty::WithOptConstParam<DefId>,
        substs: SubstsRef<'tcx>,
    ) -> Option<AbstractConst<'tcx>> {
        let inner = match (def.did.as_local(), def.const_param_did) {
            (Some(did), Some(param_did)) => {
                tcx.mir_abstract_const_of_const_arg((did, param_did))?
            }
            _ => tcx.mir_abstract_const(def.did)?,
        };

        Some(AbstractConst { inner, substs })
    }

    #[inline]
    pub fn subtree(self, node: NodeId) -> AbstractConst<'tcx> {
        AbstractConst { inner: &self.inner[..=node.index()], substs: self.substs }
    }

    #[inline]
    pub fn root(self) -> Node<'tcx> {
        self.inner.last().copied().unwrap()
    }
}

struct AbstractConstBuilder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    /// The current WIP node tree.
    nodes: IndexVec<NodeId, Node<'tcx>>,
    locals: IndexVec<mir::Local, NodeId>,
    /// We only allow field accesses if they access
    /// the result of a checked operation.
    checked_op_locals: BitSet<mir::Local>,
}

impl<'a, 'tcx> AbstractConstBuilder<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, body: &'a mir::Body<'tcx>) -> Option<AbstractConstBuilder<'a, 'tcx>> {
        // We only allow consts without control flow, so
        // we check for cycles here which simplifies the
        // rest of this implementation.
        if body.is_cfg_cyclic() {
            return None;
        }

        // We don't have to look at concrete constants, as we
        // can just evaluate them.
        if !body.is_polymorphic {
            return None;
        }

        Some(AbstractConstBuilder {
            tcx,
            body,
            nodes: IndexVec::new(),
            locals: IndexVec::from_elem(NodeId::MAX, &body.local_decls),
            checked_op_locals: BitSet::new_empty(body.local_decls.len()),
        })
    }
    fn operand_to_node(&mut self, op: &mir::Operand<'tcx>) -> Option<NodeId> {
        debug!("operand_to_node: op={:?}", op);
        const ZERO_FIELD: mir::Field = mir::Field::from_usize(0);
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                // Do not allow any projections.
                //
                // One exception are field accesses on the result of checked operations,
                // which are required to support things like `1 + 2`.
                if let Some(p) = p.as_local() {
                    debug_assert!(!self.checked_op_locals.contains(p));
                    Some(self.locals[p])
                } else if let &[mir::ProjectionElem::Field(ZERO_FIELD, _)] = p.projection.as_ref() {
                    // Only allow field accesses if the given local
                    // contains the result of a checked operation.
                    if self.checked_op_locals.contains(p.local) {
                        Some(self.locals[p.local])
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            mir::Operand::Constant(ct) => Some(self.nodes.push(Node::Leaf(ct.literal))),
        }
    }

    /// We do not allow all binary operations in abstract consts, so filter disallowed ones.
    fn check_binop(op: mir::BinOp) -> bool {
        use mir::BinOp::*;
        match op {
            Add | Sub | Mul | Div | Rem | BitXor | BitAnd | BitOr | Shl | Shr | Eq | Lt | Le
            | Ne | Ge | Gt => true,
            Offset => false,
        }
    }

    /// While we currently allow all unary operations, we still want to explicitly guard against
    /// future changes here.
    fn check_unop(op: mir::UnOp) -> bool {
        use mir::UnOp::*;
        match op {
            Not | Neg => true,
        }
    }

    fn build_statement(&mut self, stmt: &mir::Statement<'tcx>) -> Option<()> {
        debug!("AbstractConstBuilder: stmt={:?}", stmt);
        match stmt.kind {
            StatementKind::Assign(box (ref place, ref rvalue)) => {
                let local = place.as_local()?;
                match *rvalue {
                    Rvalue::Use(ref operand) => {
                        self.locals[local] = self.operand_to_node(operand)?;
                        Some(())
                    }
                    Rvalue::BinaryOp(op, ref lhs, ref rhs) if Self::check_binop(op) => {
                        let lhs = self.operand_to_node(lhs)?;
                        let rhs = self.operand_to_node(rhs)?;
                        self.locals[local] = self.nodes.push(Node::Binop(op, lhs, rhs));
                        if op.is_checkable() {
                            bug!("unexpected unchecked checkable binary operation");
                        } else {
                            Some(())
                        }
                    }
                    Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) if Self::check_binop(op) => {
                        let lhs = self.operand_to_node(lhs)?;
                        let rhs = self.operand_to_node(rhs)?;
                        self.locals[local] = self.nodes.push(Node::Binop(op, lhs, rhs));
                        self.checked_op_locals.insert(local);
                        Some(())
                    }
                    Rvalue::UnaryOp(op, ref operand) if Self::check_unop(op) => {
                        let operand = self.operand_to_node(operand)?;
                        self.locals[local] = self.nodes.push(Node::UnaryOp(op, operand));
                        Some(())
                    }
                    _ => None,
                }
            }
            // These are not actually relevant for us here, so we can ignore them.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => Some(()),
            _ => None,
        }
    }

    /// Possible return values:
    ///
    /// - `None`: unsupported terminator, stop building
    /// - `Some(None)`: supported terminator, finish building
    /// - `Some(Some(block))`: support terminator, build `block` next
    fn build_terminator(
        &mut self,
        terminator: &mir::Terminator<'tcx>,
    ) -> Option<Option<mir::BasicBlock>> {
        debug!("AbstractConstBuilder: terminator={:?}", terminator);
        match terminator.kind {
            TerminatorKind::Goto { target } => Some(Some(target)),
            TerminatorKind::Return => Some(None),
            TerminatorKind::Call {
                ref func,
                ref args,
                destination: Some((ref place, target)),
                // We do not care about `cleanup` here. Any branch which
                // uses `cleanup` will fail const-eval and they therefore
                // do not matter when checking for const evaluatability.
                //
                // Do note that even if `panic::catch_unwind` is made const,
                // we still do not have to care about this, as we do not look
                // into functions.
                cleanup: _,
                // Do not allow overloaded operators for now,
                // we probably do want to allow this in the future.
                //
                // This is currently fairly irrelevant as it requires `const Trait`s.
                from_hir_call: true,
                fn_span: _,
            } => {
                let local = place.as_local()?;
                let func = self.operand_to_node(func)?;
                let args = self.tcx.arena.alloc_from_iter(
                    args.iter()
                        .map(|arg| self.operand_to_node(arg))
                        .collect::<Option<Vec<NodeId>>>()?,
                );
                self.locals[local] = self.nodes.push(Node::FunctionCall(func, args));
                Some(Some(target))
            }
            // We only allow asserts for checked operations.
            //
            // These asserts seem to all have the form `!_local.0` so
            // we only allow exactly that.
            TerminatorKind::Assert { ref cond, expected: false, target, .. } => {
                let p = match cond {
                    mir::Operand::Copy(p) | mir::Operand::Move(p) => p,
                    mir::Operand::Constant(_) => bug!("unexpected assert"),
                };

                const ONE_FIELD: mir::Field = mir::Field::from_usize(1);
                debug!("proj: {:?}", p.projection);
                if let &[mir::ProjectionElem::Field(ONE_FIELD, _)] = p.projection.as_ref() {
                    // Only allow asserts checking the result of a checked operation.
                    if self.checked_op_locals.contains(p.local) {
                        return Some(Some(target));
                    }
                }

                None
            }
            _ => None,
        }
    }

    /// Builds the abstract const by walking the mir from start to finish
    /// and bailing out when encountering an unsupported operation.
    fn build(mut self) -> Option<&'tcx [Node<'tcx>]> {
        let mut block = &self.body.basic_blocks()[mir::START_BLOCK];
        // We checked for a cyclic cfg above, so this should terminate.
        loop {
            debug!("AbstractConstBuilder: block={:?}", block);
            for stmt in block.statements.iter() {
                self.build_statement(stmt)?;
            }

            if let Some(next) = self.build_terminator(block.terminator())? {
                block = &self.body.basic_blocks()[next];
            } else {
                return Some(self.tcx.arena.alloc_from_iter(self.nodes));
            }
        }
    }
}

/// Builds an abstract const, do not use this directly, but use `AbstractConst::new` instead.
pub(super) fn mir_abstract_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> Option<&'tcx [Node<'tcx>]> {
    if tcx.features().const_evaluatable_checked {
        match tcx.def_kind(def.did) {
            // FIXME(const_evaluatable_checked): We currently only do this for anonymous constants,
            // meaning that we do not look into associated constants. I(@lcnr) am not yet sure whether
            // we want to look into them or treat them as opaque projections.
            //
            // Right now we do neither of that and simply always fail to unify them.
            DefKind::AnonConst => (),
            _ => return None,
        }
        let body = tcx.mir_const(def).borrow();
        AbstractConstBuilder::new(tcx, &body)?.build()
    } else {
        None
    }
}

pub(super) fn try_unify_abstract_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    ((a, a_substs), (b, b_substs)): (
        (ty::WithOptConstParam<DefId>, SubstsRef<'tcx>),
        (ty::WithOptConstParam<DefId>, SubstsRef<'tcx>),
    ),
) -> bool {
    if let Some(a) = AbstractConst::new(tcx, a, a_substs) {
        if let Some(b) = AbstractConst::new(tcx, b, b_substs) {
            return try_unify(tcx, a, b);
        }
    }

    false
}

/// Tries to unify two abstract constants using structural equality.
pub(super) fn try_unify<'tcx>(
    tcx: TyCtxt<'tcx>,
    a: AbstractConst<'tcx>,
    b: AbstractConst<'tcx>,
) -> bool {
    match (a.root(), b.root()) {
        (Node::Leaf(a_ct), Node::Leaf(b_ct)) => {
            let a_ct = a_ct.subst(tcx, a.substs);
            let b_ct = b_ct.subst(tcx, b.substs);
            match (a_ct.val, b_ct.val) {
                // We can just unify errors with everything to reduce the amount of
                // emitted errors here.
                (ty::ConstKind::Error(_), _) | (_, ty::ConstKind::Error(_)) => true,
                (ty::ConstKind::Param(a_param), ty::ConstKind::Param(b_param)) => {
                    a_param == b_param
                }
                (ty::ConstKind::Value(a_val), ty::ConstKind::Value(b_val)) => a_val == b_val,
                // If we have `fn a<const N: usize>() -> [u8; N + 1]` and `fn b<const M: usize>() -> [u8; 1 + M]`
                // we do not want to use `assert_eq!(a(), b())` to infer that `N` and `M` have to be `1`. This
                // means that we only allow inference variables if they are equal.
                (ty::ConstKind::Infer(a_val), ty::ConstKind::Infer(b_val)) => a_val == b_val,
                // FIXME(const_evaluatable_checked): We may want to either actually try
                // to evaluate `a_ct` and `b_ct` if they are are fully concrete or something like
                // this, for now we just return false here.
                _ => false,
            }
        }
        (Node::Binop(a_op, al, ar), Node::Binop(b_op, bl, br)) if a_op == b_op => {
            try_unify(tcx, a.subtree(al), b.subtree(bl))
                && try_unify(tcx, a.subtree(ar), b.subtree(br))
        }
        (Node::UnaryOp(a_op, av), Node::UnaryOp(b_op, bv)) if a_op == b_op => {
            try_unify(tcx, a.subtree(av), b.subtree(bv))
        }
        (Node::FunctionCall(a_f, a_args), Node::FunctionCall(b_f, b_args))
            if a_args.len() == b_args.len() =>
        {
            try_unify(tcx, a.subtree(a_f), b.subtree(b_f))
                && a_args
                    .iter()
                    .zip(b_args)
                    .all(|(&an, &bn)| try_unify(tcx, a.subtree(an), b.subtree(bn)))
        }
        _ => false,
    }
}
