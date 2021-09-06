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
use rustc_data_structures::sync::Lrc;
use rustc_errors::ErrorReported;
use rustc_hir::def::DefKind;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir;
use rustc_middle::mir::abstract_const::{Node, NodeId, NotConstEvaluatable};
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::thir;
use rustc_middle::ty::subst::{Subst, SubstsRef};
use rustc_middle::ty::{self, TyCtxt, TypeFoldable};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

use std::cmp;
use std::iter;
use std::ops::ControlFlow;

/// Check if a given constant can be evaluated.
pub fn is_const_evaluatable<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    uv: ty::Unevaluated<'tcx, ()>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), NotConstEvaluatable> {
    debug!("is_const_evaluatable({:?})", uv);
    if infcx.tcx.features().generic_const_exprs {
        let tcx = infcx.tcx;
        match AbstractConst::new(tcx, uv)? {
            // We are looking at a generic abstract constant.
            Some(ct) => {
                for pred in param_env.caller_bounds() {
                    match pred.kind().skip_binder() {
                        ty::PredicateKind::ConstEvaluatable(uv) => {
                            if let Some(b_ct) = AbstractConst::new(tcx, uv)? {
                                // Try to unify with each subtree in the AbstractConst to allow for
                                // `N + 1` being const evaluatable even if theres only a `ConstEvaluatable`
                                // predicate for `(N + 1) * 2`
                                let result =
                                    walk_abstract_const(tcx, b_ct, |b_ct| {
                                        match try_unify(tcx, ct, b_ct) {
                                            true => ControlFlow::BREAK,
                                            false => ControlFlow::CONTINUE,
                                        }
                                    });

                                if let ControlFlow::Break(()) = result {
                                    debug!("is_const_evaluatable: abstract_const ~~> ok");
                                    return Ok(());
                                }
                            }
                        }
                        _ => {} // don't care
                    }
                }

                // We were unable to unify the abstract constant with
                // a constant found in the caller bounds, there are
                // now three possible cases here.
                #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
                enum FailureKind {
                    /// The abstract const still references an inference
                    /// variable, in this case we return `TooGeneric`.
                    MentionsInfer,
                    /// The abstract const references a generic parameter,
                    /// this means that we emit an error here.
                    MentionsParam,
                    /// The substs are concrete enough that we can simply
                    /// try and evaluate the given constant.
                    Concrete,
                }
                let mut failure_kind = FailureKind::Concrete;
                walk_abstract_const::<!, _>(tcx, ct, |node| match node.root() {
                    Node::Leaf(leaf) => {
                        let leaf = leaf.subst(tcx, ct.substs);
                        if leaf.has_infer_types_or_consts() {
                            failure_kind = FailureKind::MentionsInfer;
                        } else if leaf.definitely_has_param_types_or_consts(tcx) {
                            failure_kind = cmp::min(failure_kind, FailureKind::MentionsParam);
                        }

                        ControlFlow::CONTINUE
                    }
                    Node::Cast(_, ty) => {
                        let ty = ty.subst(tcx, ct.substs);
                        if ty.has_infer_types_or_consts() {
                            failure_kind = FailureKind::MentionsInfer;
                        } else if ty.definitely_has_param_types_or_consts(tcx) {
                            failure_kind = cmp::min(failure_kind, FailureKind::MentionsParam);
                        }

                        ControlFlow::CONTINUE
                    }
                    Node::Binop(_, _, _)
                    | Node::UnaryOp(_, _)
                    | Node::FunctionCall(_, _) => ControlFlow::CONTINUE,
                });

                match failure_kind {
                    FailureKind::MentionsInfer => {
                        return Err(NotConstEvaluatable::MentionsInfer);
                    }
                    FailureKind::MentionsParam => {
                        return Err(NotConstEvaluatable::MentionsParam);
                    }
                    FailureKind::Concrete => {
                        // Dealt with below by the same code which handles this
                        // without the feature gate.
                    }
                }
            }
            None => {
                // If we are dealing with a concrete constant, we can
                // reuse the old code path and try to evaluate
                // the constant.
            }
        }
    }

    let future_compat_lint = || {
        if let Some(local_def_id) = uv.def.did.as_local() {
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
    let concrete = infcx.const_eval_resolve(param_env, uv.expand(), Some(span));

    if concrete.is_ok() && uv.substs(infcx.tcx).definitely_has_param_types_or_consts(infcx.tcx) {
        match infcx.tcx.def_kind(uv.def.did) {
            DefKind::AnonConst => {
                let mir_body = infcx.tcx.mir_for_ctfe_opt_const_arg(uv.def);

                if mir_body.is_polymorphic {
                    future_compat_lint();
                }
            }
            _ => future_compat_lint(),
        }
    }

    debug!(?concrete, "is_const_evaluatable");
    match concrete {
        Err(ErrorHandled::TooGeneric) => Err(match uv.has_infer_types_or_consts() {
            true => NotConstEvaluatable::MentionsInfer,
            false => NotConstEvaluatable::MentionsParam,
        }),
        Err(ErrorHandled::Linted) => {
            infcx.tcx.sess.delay_span_bug(span, "constant in type had error reported as lint");
            Err(NotConstEvaluatable::Error(ErrorReported))
        }
        Err(ErrorHandled::Reported(e)) => Err(NotConstEvaluatable::Error(e)),
        Ok(_) => Ok(()),
    }
}

/// A tree representing an anonymous constant.
///
/// This is only able to represent a subset of `MIR`,
/// and should not leak any information about desugarings.
#[derive(Debug, Clone, Copy)]
pub struct AbstractConst<'tcx> {
    // FIXME: Consider adding something like `IndexSlice`
    // and use this here.
    pub inner: &'tcx [Node<'tcx>],
    pub substs: SubstsRef<'tcx>,
}

impl<'tcx> AbstractConst<'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        uv: ty::Unevaluated<'tcx, ()>,
    ) -> Result<Option<AbstractConst<'tcx>>, ErrorReported> {
        let inner = tcx.mir_abstract_const_opt_const_arg(uv.def)?;
        debug!("AbstractConst::new({:?}) = {:?}", uv, inner);
        Ok(inner.map(|inner| AbstractConst { inner, substs: uv.substs(tcx) }))
    }

    pub fn from_const(
        tcx: TyCtxt<'tcx>,
        ct: &ty::Const<'tcx>,
    ) -> Result<Option<AbstractConst<'tcx>>, ErrorReported> {
        match ct.val {
            ty::ConstKind::Unevaluated(uv) => AbstractConst::new(tcx, uv.shrink()),
            ty::ConstKind::Error(_) => Err(ErrorReported),
            _ => Ok(None),
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkNode<'tcx> {
    node: Node<'tcx>,
    span: Span,
    used: bool,
}

struct AbstractConstBuilder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body_id: thir::ExprId,
    body: Lrc<&'a thir::Thir<'tcx>>,
    /// The current WIP node tree.
    ///
    /// We require all nodes to be used in the final abstract const,
    /// so we store this here. Note that we also consider nodes as used
    /// if they are mentioned in an assert, so some used nodes are never
    /// actually reachable by walking the [`AbstractConst`].
    nodes: IndexVec<NodeId, WorkNode<'tcx>>,
}

impl<'a, 'tcx> AbstractConstBuilder<'a, 'tcx> {
    fn root_span(&self) -> Span {
        self.body.exprs[self.body_id].span
    }

    fn error(&mut self, span: Option<Span>, msg: &str) -> Result<!, ErrorReported> {
        self.tcx
            .sess
            .struct_span_err(self.root_span(), "overly complex generic constant")
            .span_label(span.unwrap_or(self.root_span()), msg)
            .help("consider moving this anonymous constant into a `const` function")
            .emit();

        Err(ErrorReported)
    }

    fn new(
        tcx: TyCtxt<'tcx>,
        (body, body_id): (&'a thir::Thir<'tcx>, thir::ExprId),
    ) -> Result<Option<AbstractConstBuilder<'a, 'tcx>>, ErrorReported> {
        let builder =
            AbstractConstBuilder { tcx, body_id, body: Lrc::new(body), nodes: IndexVec::new() };

        // FIXME non-constants should return Ok(None)

        Ok(Some(builder))
    }

    fn add_node(&mut self, node: Node<'tcx>, span: Span) -> NodeId {
        // Mark used nodes.
        match node {
            Node::Leaf(_) => (),
            Node::Binop(_, lhs, rhs) => {
                self.nodes[lhs].used = true;
                self.nodes[rhs].used = true;
            }
            Node::UnaryOp(_, input) => {
                self.nodes[input].used = true;
            }
            Node::FunctionCall(func, nodes) => {
                self.nodes[func].used = true;
                nodes.iter().for_each(|&n| self.nodes[n].used = true);
            }
            Node::Cast(operand, _) => {
                self.nodes[operand].used = true;
            }
        }

        // Nodes start as unused.
        self.nodes.push(WorkNode { node, span, used: false })
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

    /// Builds the abstract const by walking the thir and bailing out when
    /// encountering an unspported operation.
    fn build(mut self) -> Result<&'tcx [Node<'tcx>], ErrorReported> {
        debug!("Abstractconstbuilder::build: body={:?}", &*self.body);
        let last = self.recurse_build(self.body_id)?;
        self.nodes[last].used = true;

        for n in self.nodes.iter() {
            if let Node::Leaf(ty::Const { val: ty::ConstKind::Unevaluated(ct), ty: _ }) = n.node {
                // `AbstractConst`s should not contain any promoteds as they require references which
                // are not allowed.
                assert_eq!(ct.promoted, None);
            }
        }

        // FIXME I dont even think we can get unused nodes anymore with thir abstract const
        if let Some(&unused) = self.nodes.iter().find(|n| !n.used) {
            self.error(Some(unused.span), "dead code")?;
        }

        Ok(self.tcx.arena.alloc_from_iter(self.nodes.into_iter().map(|n| n.node)))
    }

    fn recurse_build(&mut self, node: thir::ExprId) -> Result<NodeId, ErrorReported> {
        use thir::ExprKind;
        let node = &self.body.clone().exprs[node];
        debug!("recurse_build: node={:?}", node);
        Ok(match &node.kind {
            // I dont know if handling of these 3 is correct
            &ExprKind::Scope { value, .. } => self.recurse_build(value)?,
            &ExprKind::PlaceTypeAscription { source, .. } |
            &ExprKind::ValueTypeAscription { source, .. } => self.recurse_build(source)?,

            // subtle: associated consts are literals this arm handles 
            // `<T as Trait>::ASSOC` as well as `12` 
            &ExprKind::Literal { literal, .. }
            | &ExprKind::StaticRef { literal, .. } => self.add_node(Node::Leaf(literal), node.span),

            // FIXME(generic_const_exprs) handle `from_hir_call` field
            ExprKind::Call { fun, args,  .. } => {
                let fun = self.recurse_build(*fun)?;

                let mut new_args = Vec::<NodeId>::with_capacity(args.len());
                for &id in args.iter() {
                    new_args.push(self.recurse_build(id)?);
                }
                let new_args = self.tcx.arena.alloc_slice(&new_args);
                self.add_node(Node::FunctionCall(fun, new_args), node.span)
            },
            &ExprKind::Binary { op, lhs, rhs } if Self::check_binop(op) => {
                let lhs = self.recurse_build(lhs)?;
                let rhs = self.recurse_build(rhs)?;
                self.add_node(Node::Binop(op, lhs, rhs), node.span)
            }
            &ExprKind::Unary { op, arg } if Self::check_unop(op) => {
                let arg = self.recurse_build(arg)?;
                self.add_node(Node::UnaryOp(op, arg), node.span)
            },
            // this is necessary so that the following compiles:
            //
            // ```
            // fn foo<const N: usize>(a: [(); N + 1]) {
            //     bar::<{ N + 1 }>();
            // }
            // ```
            ExprKind::Block { body: thir::Block { stmts: box [], expr: Some(e), .. }} => self.recurse_build(*e)?,
            // ExprKind::Use happens when a `hir::ExprKind::Cast` is a 
            // "coercion cast" i.e. using a coercion or is a no-op.
            // this is important so that `N as usize as usize` doesnt unify with `N as usize`
            &ExprKind::Use { source} 
            | &ExprKind::Cast { source } => {
                let arg = self.recurse_build(source)?;
                self.add_node(Node::Cast(arg, node.ty), node.span)
            },
            // never can arise even without panic/fail to terminate
            &ExprKind::NeverToAny { source } => todo!(),

            // FIXME(generic_const_exprs) we want to support these
            ExprKind::AddressOf { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::Deref { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Array { .. }
            | ExprKind::Block { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Index { .. }
            | ExprKind::Field { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::Adt(_) => return self.error(
                    Some(node.span), 
                    "unsupported operation in generic constant, this may be supported in the future",
                ).map(|never| never),

            ExprKind::Match { .. }
            | ExprKind::VarRef { .. } //
            | ExprKind::UpvarRef { .. } // we dont permit let stmts so...
            | ExprKind::Closure { .. }
            | ExprKind::Let { .. } // let expressions imply control flow
            | ExprKind::Loop { .. }
            | ExprKind::Assign { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Unary { .. } //
            | ExprKind::Binary { .. } // we handle valid unary/binary ops above 
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::If { .. }
            | ExprKind::Pointer { .. } // dont know if this is correct
            | ExprKind::ThreadLocalRef(_)
            | ExprKind::LlvmInlineAsm { .. }
            | ExprKind::Return { .. }
            | ExprKind::Box { .. } // allocations not allowed in constants
            | ExprKind::AssignOp { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::Yield { .. } => return self.error(Some(node.span), "unsupported operation in generic constant").map(|never| never),
        })
    }
}

/// Builds an abstract const, do not use this directly, but use `AbstractConst::new` instead.
pub(super) fn mir_abstract_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> Result<Option<&'tcx [mir::abstract_const::Node<'tcx>]>, ErrorReported> {
    if tcx.features().generic_const_exprs {
        match tcx.def_kind(def.did) {
            // FIXME(generic_const_exprs): We currently only do this for anonymous constants,
            // meaning that we do not look into associated constants. I(@lcnr) am not yet sure whether
            // we want to look into them or treat them as opaque projections.
            //
            // Right now we do neither of that and simply always fail to unify them.
            DefKind::AnonConst => (),
            _ => return Ok(None),
        }
        debug!("mir_abstract_const: {:?}", def);
        let body = tcx.thir_body(def);

        if body.0.borrow().exprs.is_empty() {
            // type error in constant, there is no thir
            return Err(ErrorReported);
        }

        AbstractConstBuilder::new(tcx, (&*body.0.borrow(), body.1))?
            .map(AbstractConstBuilder::build)
            .transpose()
    } else {
        Ok(None)
    }
}

pub(super) fn try_unify_abstract_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    (a, b): (ty::Unevaluated<'tcx, ()>, ty::Unevaluated<'tcx, ()>),
) -> bool {
    (|| {
        if let Some(a) = AbstractConst::new(tcx, a)? {
            if let Some(b) = AbstractConst::new(tcx, b)? {
                return Ok(try_unify(tcx, a, b));
            }
        }

        Ok(false)
    })()
    .unwrap_or_else(|ErrorReported| true)
    // FIXME(generic_const_exprs): We should instead have this
    // method return the resulting `ty::Const` and return `ConstKind::Error`
    // on `ErrorReported`.
}

pub fn walk_abstract_const<'tcx, R, F>(
    tcx: TyCtxt<'tcx>,
    ct: AbstractConst<'tcx>,
    mut f: F,
) -> ControlFlow<R>
where
    F: FnMut(AbstractConst<'tcx>) -> ControlFlow<R>,
{
    fn recurse<'tcx, R>(
        tcx: TyCtxt<'tcx>,
        ct: AbstractConst<'tcx>,
        f: &mut dyn FnMut(AbstractConst<'tcx>) -> ControlFlow<R>,
    ) -> ControlFlow<R> {
        f(ct)?;
        let root = ct.root();
        match root {
            Node::Leaf(_) => ControlFlow::CONTINUE,
            Node::Binop(_, l, r) => {
                recurse(tcx, ct.subtree(l), f)?;
                recurse(tcx, ct.subtree(r), f)
            }
            Node::UnaryOp(_, v) => recurse(tcx, ct.subtree(v), f),
            Node::FunctionCall(func, args) => {
                recurse(tcx, ct.subtree(func), f)?;
                args.iter().try_for_each(|&arg| recurse(tcx, ct.subtree(arg), f))
            }
            Node::Cast(operand, _) => recurse(tcx, ct.subtree(operand), f),
        }
    }

    recurse(tcx, ct, &mut f)
}

/// Tries to unify two abstract constants using structural equality.
pub(super) fn try_unify<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut a: AbstractConst<'tcx>,
    mut b: AbstractConst<'tcx>,
) -> bool {
    // We substitute generics repeatedly to allow AbstractConsts to unify where a
    // ConstKind::Unevalated could be turned into an AbstractConst that would unify e.g.
    // Param(N) should unify with Param(T), substs: [Unevaluated("T2", [Unevaluated("T3", [Param(N)])])]
    while let Node::Leaf(a_ct) = a.root() {
        let a_ct = a_ct.subst(tcx, a.substs);
        match AbstractConst::from_const(tcx, a_ct) {
            Ok(Some(a_act)) => a = a_act,
            Ok(None) => break,
            Err(_) => return true,
        }
    }
    while let Node::Leaf(b_ct) = b.root() {
        let b_ct = b_ct.subst(tcx, b.substs);
        match AbstractConst::from_const(tcx, b_ct) {
            Ok(Some(b_act)) => b = b_act,
            Ok(None) => break,
            Err(_) => return true,
        }
    }

    match (a.root(), b.root()) {
        (Node::Leaf(a_ct), Node::Leaf(b_ct)) => {
            let a_ct = a_ct.subst(tcx, a.substs);
            let b_ct = b_ct.subst(tcx, b.substs);
            if a_ct.ty != b_ct.ty {
                return false;
            }

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
                // We expand generic anonymous constants at the start of this function, so this
                // branch should only be taking when dealing with associated constants, at
                // which point directly comparing them seems like the desired behavior.
                //
                // FIXME(generic_const_exprs): This isn't actually the case.
                // We also take this branch for concrete anonymous constants and
                // expand generic anonymous constants with concrete substs.
                (ty::ConstKind::Unevaluated(a_uv), ty::ConstKind::Unevaluated(b_uv)) => {
                    a_uv == b_uv
                }
                // FIXME(generic_const_exprs): We may want to either actually try
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
                && iter::zip(a_args, b_args)
                    .all(|(&an, &bn)| try_unify(tcx, a.subtree(an), b.subtree(bn)))
        }
        (Node::Cast(a_operand, a_ty), Node::Cast(b_operand, b_ty))
            if (a_ty == b_ty) =>
        {
            try_unify(tcx, a.subtree(a_operand), b.subtree(b_operand))
        }
        // use this over `_ => false` to make adding variants to `Node` less error prone
        (Node::Cast(..), _) 
        | (Node::FunctionCall(..), _) 
        | (Node::UnaryOp(..), _) 
        | (Node::Binop(..), _) 
        | (Node::Leaf(..), _) => false,
    }
}
