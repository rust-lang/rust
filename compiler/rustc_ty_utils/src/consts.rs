use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::ty::abstract_const::{CastKind, Node, NodeId};
use rustc_middle::ty::{self, TyCtxt, TypeVisitable};
use rustc_middle::{mir, thir};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;

use std::iter;

use crate::errors::{GenericConstantTooComplex, GenericConstantTooComplexSub};

/// Destructures array, ADT or tuple constants into the constants
/// of their fields.
pub(crate) fn destructure_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    const_: ty::Const<'tcx>,
) -> ty::DestructuredConst<'tcx> {
    let ty::ConstKind::Value(valtree) = const_.kind() else {
        bug!("cannot destructure constant {:?}", const_)
    };

    let branches = match valtree {
        ty::ValTree::Branch(b) => b,
        _ => bug!("cannot destructure constant {:?}", const_),
    };

    let (fields, variant) = match const_.ty().kind() {
        ty::Array(inner_ty, _) | ty::Slice(inner_ty) => {
            // construct the consts for the elements of the array/slice
            let field_consts = branches
                .iter()
                .map(|b| tcx.mk_const(ty::ConstS { kind: ty::ConstKind::Value(*b), ty: *inner_ty }))
                .collect::<Vec<_>>();
            debug!(?field_consts);

            (field_consts, None)
        }
        ty::Adt(def, _) if def.variants().is_empty() => bug!("unreachable"),
        ty::Adt(def, substs) => {
            let (variant_idx, branches) = if def.is_enum() {
                let (head, rest) = branches.split_first().unwrap();
                (VariantIdx::from_u32(head.unwrap_leaf().try_to_u32().unwrap()), rest)
            } else {
                (VariantIdx::from_u32(0), branches)
            };
            let fields = &def.variant(variant_idx).fields;
            let mut field_consts = Vec::with_capacity(fields.len());

            for (field, field_valtree) in iter::zip(fields, branches) {
                let field_ty = field.ty(tcx, substs);
                let field_const = tcx.mk_const(ty::ConstS {
                    kind: ty::ConstKind::Value(*field_valtree),
                    ty: field_ty,
                });
                field_consts.push(field_const);
            }
            debug!(?field_consts);

            (field_consts, Some(variant_idx))
        }
        ty::Tuple(elem_tys) => {
            let fields = iter::zip(*elem_tys, branches)
                .map(|(elem_ty, elem_valtree)| {
                    tcx.mk_const(ty::ConstS {
                        kind: ty::ConstKind::Value(*elem_valtree),
                        ty: elem_ty,
                    })
                })
                .collect::<Vec<_>>();

            (fields, None)
        }
        _ => bug!("cannot destructure constant {:?}", const_),
    };

    let fields = tcx.arena.alloc_from_iter(fields.into_iter());

    ty::DestructuredConst { variant, fields }
}

pub struct AbstractConstBuilder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body_id: thir::ExprId,
    body: &'a thir::Thir<'tcx>,
    /// The current WIP node tree.
    nodes: IndexVec<NodeId, Node<'tcx>>,
}

impl<'a, 'tcx> AbstractConstBuilder<'a, 'tcx> {
    fn root_span(&self) -> Span {
        self.body.exprs[self.body_id].span
    }

    fn error(&mut self, sub: GenericConstantTooComplexSub) -> Result<!, ErrorGuaranteed> {
        let reported = self.tcx.sess.emit_err(GenericConstantTooComplex {
            span: self.root_span(),
            maybe_supported: None,
            sub,
        });

        Err(reported)
    }

    fn maybe_supported_error(
        &mut self,
        sub: GenericConstantTooComplexSub,
    ) -> Result<!, ErrorGuaranteed> {
        let reported = self.tcx.sess.emit_err(GenericConstantTooComplex {
            span: self.root_span(),
            maybe_supported: Some(()),
            sub,
        });

        Err(reported)
    }

    #[instrument(skip(tcx, body, body_id), level = "debug")]
    pub fn new(
        tcx: TyCtxt<'tcx>,
        (body, body_id): (&'a thir::Thir<'tcx>, thir::ExprId),
    ) -> Result<Option<AbstractConstBuilder<'a, 'tcx>>, ErrorGuaranteed> {
        let builder = AbstractConstBuilder { tcx, body_id, body, nodes: IndexVec::new() };

        struct IsThirPolymorphic<'a, 'tcx> {
            is_poly: bool,
            thir: &'a thir::Thir<'tcx>,
        }

        use crate::rustc_middle::thir::visit::Visitor;
        use thir::visit;

        impl<'a, 'tcx> IsThirPolymorphic<'a, 'tcx> {
            fn expr_is_poly(&mut self, expr: &thir::Expr<'tcx>) -> bool {
                if expr.ty.has_param_types_or_consts() {
                    return true;
                }

                match expr.kind {
                    thir::ExprKind::NamedConst { substs, .. } => substs.has_param_types_or_consts(),
                    thir::ExprKind::ConstParam { .. } => true,
                    thir::ExprKind::Repeat { value, count } => {
                        self.visit_expr(&self.thir()[value]);
                        count.has_param_types_or_consts()
                    }
                    _ => false,
                }
            }

            fn pat_is_poly(&mut self, pat: &thir::Pat<'tcx>) -> bool {
                if pat.ty.has_param_types_or_consts() {
                    return true;
                }

                match pat.kind.as_ref() {
                    thir::PatKind::Constant { value } => value.has_param_types_or_consts(),
                    thir::PatKind::Range(thir::PatRange { lo, hi, .. }) => {
                        lo.has_param_types_or_consts() || hi.has_param_types_or_consts()
                    }
                    _ => false,
                }
            }
        }

        impl<'a, 'tcx> visit::Visitor<'a, 'tcx> for IsThirPolymorphic<'a, 'tcx> {
            fn thir(&self) -> &'a thir::Thir<'tcx> {
                &self.thir
            }

            #[instrument(skip(self), level = "debug")]
            fn visit_expr(&mut self, expr: &thir::Expr<'tcx>) {
                self.is_poly |= self.expr_is_poly(expr);
                if !self.is_poly {
                    visit::walk_expr(self, expr)
                }
            }

            #[instrument(skip(self), level = "debug")]
            fn visit_pat(&mut self, pat: &thir::Pat<'tcx>) {
                self.is_poly |= self.pat_is_poly(pat);
                if !self.is_poly {
                    visit::walk_pat(self, pat);
                }
            }
        }

        let mut is_poly_vis = IsThirPolymorphic { is_poly: false, thir: body };
        visit::walk_expr(&mut is_poly_vis, &body[body_id]);
        debug!("AbstractConstBuilder: is_poly={}", is_poly_vis.is_poly);
        if !is_poly_vis.is_poly {
            return Ok(None);
        }

        Ok(Some(builder))
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
    /// encountering an unsupported operation.
    pub fn build(mut self) -> Result<&'tcx [Node<'tcx>], ErrorGuaranteed> {
        debug!("AbstractConstBuilder::build: body={:?}", &*self.body);
        self.recurse_build(self.body_id)?;

        for n in self.nodes.iter() {
            if let Node::Leaf(ct) = n {
                if let ty::ConstKind::Unevaluated(ct) = ct.kind() {
                    // `AbstractConst`s should not contain any promoteds as they require references which
                    // are not allowed.
                    assert_eq!(ct.promoted, None);
                    assert_eq!(ct, self.tcx.erase_regions(ct));
                }
            }
        }

        Ok(self.tcx.arena.alloc_from_iter(self.nodes.into_iter()))
    }

    fn recurse_build(&mut self, node: thir::ExprId) -> Result<NodeId, ErrorGuaranteed> {
        use thir::ExprKind;
        let node = &self.body.exprs[node];
        Ok(match &node.kind {
            // I dont know if handling of these 3 is correct
            &ExprKind::Scope { value, .. } => self.recurse_build(value)?,
            &ExprKind::PlaceTypeAscription { source, .. }
            | &ExprKind::ValueTypeAscription { source, .. } => self.recurse_build(source)?,
            &ExprKind::Literal { lit, neg } => {
                let sp = node.span;
                let constant = match self.tcx.at(sp).lit_to_const(LitToConstInput {
                    lit: &lit.node,
                    ty: node.ty,
                    neg,
                }) {
                    Ok(c) => c,
                    Err(LitToConstError::Reported) => self.tcx.const_error(node.ty),
                    Err(LitToConstError::TypeError) => {
                        bug!("encountered type error in lit_to_const")
                    }
                };

                self.nodes.push(Node::Leaf(constant))
            }
            &ExprKind::NonHirLiteral { lit, user_ty: _ } => {
                let val = ty::ValTree::from_scalar_int(lit);
                self.nodes.push(Node::Leaf(ty::Const::from_value(self.tcx, val, node.ty)))
            }
            &ExprKind::ZstLiteral { user_ty: _ } => {
                let val = ty::ValTree::zst();
                self.nodes.push(Node::Leaf(ty::Const::from_value(self.tcx, val, node.ty)))
            }
            &ExprKind::NamedConst { def_id, substs, user_ty: _ } => {
                let uneval = ty::Unevaluated::new(ty::WithOptConstParam::unknown(def_id), substs);

                let constant = self
                    .tcx
                    .mk_const(ty::ConstS { kind: ty::ConstKind::Unevaluated(uneval), ty: node.ty });

                self.nodes.push(Node::Leaf(constant))
            }

            ExprKind::ConstParam { param, .. } => {
                let const_param = self
                    .tcx
                    .mk_const(ty::ConstS { kind: ty::ConstKind::Param(*param), ty: node.ty });
                self.nodes.push(Node::Leaf(const_param))
            }

            ExprKind::Call { fun, args, .. } => {
                let fun = self.recurse_build(*fun)?;

                let mut new_args = Vec::<NodeId>::with_capacity(args.len());
                for &id in args.iter() {
                    new_args.push(self.recurse_build(id)?);
                }
                let new_args = self.tcx.arena.alloc_slice(&new_args);
                self.nodes.push(Node::FunctionCall(fun, new_args))
            }
            &ExprKind::Binary { op, lhs, rhs } if Self::check_binop(op) => {
                let lhs = self.recurse_build(lhs)?;
                let rhs = self.recurse_build(rhs)?;
                self.nodes.push(Node::Binop(op, lhs, rhs))
            }
            &ExprKind::Unary { op, arg } if Self::check_unop(op) => {
                let arg = self.recurse_build(arg)?;
                self.nodes.push(Node::UnaryOp(op, arg))
            }
            // This is necessary so that the following compiles:
            //
            // ```
            // fn foo<const N: usize>(a: [(); N + 1]) {
            //     bar::<{ N + 1 }>();
            // }
            // ```
            ExprKind::Block { block } => {
                if let thir::Block { stmts: box [], expr: Some(e), .. } = &self.body.blocks[*block]
                {
                    self.recurse_build(*e)?
                } else {
                    self.maybe_supported_error(GenericConstantTooComplexSub::BlockNotSupported(
                        node.span,
                    ))?
                }
            }
            // `ExprKind::Use` happens when a `hir::ExprKind::Cast` is a
            // "coercion cast" i.e. using a coercion or is a no-op.
            // This is important so that `N as usize as usize` doesnt unify with `N as usize`. (untested)
            &ExprKind::Use { source } => {
                let arg = self.recurse_build(source)?;
                self.nodes.push(Node::Cast(CastKind::Use, arg, node.ty))
            }
            &ExprKind::Cast { source } => {
                let arg = self.recurse_build(source)?;
                self.nodes.push(Node::Cast(CastKind::As, arg, node.ty))
            }
            ExprKind::Borrow { arg, .. } => {
                let arg_node = &self.body.exprs[*arg];

                // Skip reborrows for now until we allow Deref/Borrow/AddressOf
                // expressions.
                // FIXME(generic_const_exprs): Verify/explain why this is sound
                if let ExprKind::Deref { arg } = arg_node.kind {
                    self.recurse_build(arg)?
                } else {
                    self.maybe_supported_error(GenericConstantTooComplexSub::BorrowNotSupported(
                        node.span,
                    ))?
                }
            }
            // FIXME(generic_const_exprs): We may want to support these.
            ExprKind::AddressOf { .. } | ExprKind::Deref { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::AddressAndDerefNotSupported(node.span),
            )?,
            ExprKind::Repeat { .. } | ExprKind::Array { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::ArrayNotSupported(node.span),
            )?,
            ExprKind::NeverToAny { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::NeverToAnyNotSupported(node.span),
            )?,
            ExprKind::Tuple { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::TupleNotSupported(node.span),
            )?,
            ExprKind::Index { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::IndexNotSupported(node.span),
            )?,
            ExprKind::Field { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::FieldNotSupported(node.span),
            )?,
            ExprKind::ConstBlock { .. } => self.maybe_supported_error(
                GenericConstantTooComplexSub::ConstBlockNotSupported(node.span),
            )?,
            ExprKind::Adt(_) => self
                .maybe_supported_error(GenericConstantTooComplexSub::AdtNotSupported(node.span))?,
            // dont know if this is correct
            ExprKind::Pointer { .. } => {
                self.error(GenericConstantTooComplexSub::PointerNotSupported(node.span))?
            }
            ExprKind::Yield { .. } => {
                self.error(GenericConstantTooComplexSub::YieldNotSupported(node.span))?
            }
            ExprKind::Continue { .. } | ExprKind::Break { .. } | ExprKind::Loop { .. } => {
                self.error(GenericConstantTooComplexSub::LoopNotSupported(node.span))?
            }
            ExprKind::Box { .. } => {
                self.error(GenericConstantTooComplexSub::BoxNotSupported(node.span))?
            }

            ExprKind::Unary { .. } => unreachable!(),
            // we handle valid unary/binary ops above
            ExprKind::Binary { .. } => {
                self.error(GenericConstantTooComplexSub::BinaryNotSupported(node.span))?
            }
            ExprKind::LogicalOp { .. } => {
                self.error(GenericConstantTooComplexSub::LogicalOpNotSupported(node.span))?
            }
            ExprKind::Assign { .. } | ExprKind::AssignOp { .. } => {
                self.error(GenericConstantTooComplexSub::AssignNotSupported(node.span))?
            }
            ExprKind::Closure { .. } | ExprKind::Return { .. } => {
                self.error(GenericConstantTooComplexSub::ClosureAndReturnNotSupported(node.span))?
            }
            // let expressions imply control flow
            ExprKind::Match { .. } | ExprKind::If { .. } | ExprKind::Let { .. } => {
                self.error(GenericConstantTooComplexSub::ControlFlowNotSupported(node.span))?
            }
            ExprKind::InlineAsm { .. } => {
                self.error(GenericConstantTooComplexSub::InlineAsmNotSupported(node.span))?
            }

            // we dont permit let stmts so `VarRef` and `UpvarRef` cant happen
            ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::ThreadLocalRef(_) => {
                self.error(GenericConstantTooComplexSub::OperationNotSupported(node.span))?
            }
        })
    }
}

/// Builds an abstract const, do not use this directly, but use `AbstractConst::new` instead.
pub fn thir_abstract_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> Result<Option<&'tcx [Node<'tcx>]>, ErrorGuaranteed> {
    if tcx.features().generic_const_exprs {
        match tcx.def_kind(def.did) {
            // FIXME(generic_const_exprs): We currently only do this for anonymous constants,
            // meaning that we do not look into associated constants. I(@lcnr) am not yet sure whether
            // we want to look into them or treat them as opaque projections.
            //
            // Right now we do neither of that and simply always fail to unify them.
            DefKind::AnonConst | DefKind::InlineConst => (),
            _ => return Ok(None),
        }

        let body = tcx.thir_body(def)?;

        AbstractConstBuilder::new(tcx, (&*body.0.borrow(), body.1))?
            .map(AbstractConstBuilder::build)
            .transpose()
    } else {
        Ok(None)
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        destructure_const,
        thir_abstract_const: |tcx, def_id| {
            let def_id = def_id.expect_local();
            if let Some(def) = ty::WithOptConstParam::try_lookup(def_id, tcx) {
                tcx.thir_abstract_const_of_const_arg(def)
            } else {
                thir_abstract_const(tcx, ty::WithOptConstParam::unknown(def_id))
            }
        },
        thir_abstract_const_of_const_arg: |tcx, (did, param_did)| {
            thir_abstract_const(
                tcx,
                ty::WithOptConstParam { did, const_param_did: Some(param_did) },
            )
        },
        ..*providers
    };
}
