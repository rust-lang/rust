//! # Categorization
//!
//! The job of the categorization module is to analyze an expression to
//! determine what kind of memory is used in evaluating it (for example,
//! where dereferences occur and what kind of pointer is dereferenced;
//! whether the memory is mutable, etc.).
//!
//! Categorization effectively transforms all of our expressions into
//! expressions of the following forms (the actual enum has many more
//! possibilities, naturally, but they are all variants of these base
//! forms):
//!
//!     E = rvalue    // some computed rvalue
//!       | x         // address of a local variable or argument
//!       | *E        // deref of a ptr
//!       | E.comp    // access to an interior component
//!
//! Imagine a routine ToAddr(Expr) that evaluates an expression and returns an
//! address where the result is to be found. If Expr is a place, then this
//! is the address of the place. If `Expr` is an rvalue, this is the address of
//! some temporary spot in memory where the result is stored.
//!
//! Now, `cat_expr()` classifies the expression `Expr` and the address `A = ToAddr(Expr)`
//! as follows:
//!
//! - `cat`: what kind of expression was this? This is a subset of the
//!   full expression forms which only includes those that we care about
//!   for the purpose of the analysis.
//! - `mutbl`: mutability of the address `A`.
//! - `ty`: the type of data found at the address `A`.
//!
//! The resulting categorization tree differs somewhat from the expressions
//! themselves. For example, auto-derefs are explicit. Also, an index a[b] is
//! decomposed into two operations: a dereference to reach the array data and
//! then an index to jump forward to the relevant item.
//!
//! ## By-reference upvars
//!
//! One part of the codegen which may be non-obvious is that we translate
//! closure upvars into the dereference of a borrowed pointer; this more closely
//! resembles the runtime codegen. So, for example, if we had:
//!
//!     let mut x = 3;
//!     let y = 5;
//!     let inc = || x += y;
//!
//! Then when we categorize `x` (*within* the closure) we would yield a
//! result of `*x'`, effectively, where `x'` is a `Categorization::Upvar` reference
//! tied to `x`. The type of `x'` will be a borrowed pointer.

use rustc::ty::adjustment;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, Ty, TyCtxt};

use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::PatKind;
use rustc_infer::infer::InferCtxt;
use rustc_span::Span;
use rustc_trait_selection::infer::InferCtxtExt;

#[derive(Clone, Debug)]
pub enum PlaceBase {
    /// A temporary variable
    Rvalue,
    /// A named `static` item
    StaticItem,
    /// A named local variable
    Local(hir::HirId),
    /// An upvar referenced by closure env
    Upvar(ty::UpvarId),
}

#[derive(Clone, Debug)]
pub enum Projection<'tcx> {
    /// A dereference of a pointer, reference or `Box<T>` of the given type
    Deref(Ty<'tcx>),
    /// An index or a field
    Other,
}

/// A `Place` represents how a value is located in memory.
///
/// This is an HIR version of `mir::Place`
#[derive(Clone, Debug)]
pub struct Place<'tcx> {
    /// `HirId` of the expression or pattern producing this value.
    pub hir_id: hir::HirId,
    /// The `Span` of the expression or pattern producing this value.
    pub span: Span,
    /// The type of the `Place`
    pub ty: Ty<'tcx>,
    /// The "outermost" place that holds this value.
    pub base: PlaceBase,
    /// How this place is derived from the base place.
    pub projections: Vec<Projection<'tcx>>,
}

impl<'tcx> Place<'tcx> {
    /// Returns an iterator of the types that have to be dereferenced to access
    /// the `Place`.
    ///
    /// The types are in the reverse order that they are applied. So if
    /// `x: &*const u32` and the `Place` is `**x`, then the types returned are
    ///`*const u32` then `&*const u32`.
    crate fn deref_tys(&self) -> impl Iterator<Item = Ty<'tcx>> + '_ {
        self.projections.iter().rev().filter_map(|proj| {
            if let Projection::Deref(deref_ty) = *proj { Some(deref_ty) } else { None }
        })
    }
}

crate trait HirNode {
    fn hir_id(&self) -> hir::HirId;
    fn span(&self) -> Span;
}

impl HirNode for hir::Expr<'_> {
    fn hir_id(&self) -> hir::HirId {
        self.hir_id
    }
    fn span(&self) -> Span {
        self.span
    }
}

impl HirNode for hir::Pat<'_> {
    fn hir_id(&self) -> hir::HirId {
        self.hir_id
    }
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Clone)]
crate struct MemCategorizationContext<'a, 'tcx> {
    crate tables: &'a ty::TypeckTables<'tcx>,
    infcx: &'a InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_owner: DefId,
    upvars: Option<&'tcx FxIndexMap<hir::HirId, hir::Upvar>>,
}

crate type McResult<T> = Result<T, ()>;

impl<'a, 'tcx> MemCategorizationContext<'a, 'tcx> {
    /// Creates a `MemCategorizationContext`.
    crate fn new(
        infcx: &'a InferCtxt<'a, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_owner: DefId,
        tables: &'a ty::TypeckTables<'tcx>,
    ) -> MemCategorizationContext<'a, 'tcx> {
        MemCategorizationContext {
            tables,
            infcx,
            param_env,
            body_owner,
            upvars: infcx.tcx.upvars(body_owner),
        }
    }

    crate fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    crate fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.infcx.type_is_copy_modulo_regions(self.param_env, ty, span)
    }

    fn resolve_vars_if_possible<T>(&self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.infcx.resolve_vars_if_possible(value)
    }

    fn is_tainted_by_errors(&self) -> bool {
        self.infcx.is_tainted_by_errors()
    }

    fn resolve_type_vars_or_error(
        &self,
        id: hir::HirId,
        ty: Option<Ty<'tcx>>,
    ) -> McResult<Ty<'tcx>> {
        match ty {
            Some(ty) => {
                let ty = self.resolve_vars_if_possible(&ty);
                if ty.references_error() || ty.is_ty_var() {
                    debug!("resolve_type_vars_or_error: error from {:?}", ty);
                    Err(())
                } else {
                    Ok(ty)
                }
            }
            // FIXME
            None if self.is_tainted_by_errors() => Err(()),
            None => {
                bug!(
                    "no type for node {}: {} in mem_categorization",
                    id,
                    self.tcx().hir().node_to_string(id)
                );
            }
        }
    }

    crate fn node_ty(&self, hir_id: hir::HirId) -> McResult<Ty<'tcx>> {
        self.resolve_type_vars_or_error(hir_id, self.tables.node_type_opt(hir_id))
    }

    fn expr_ty(&self, expr: &hir::Expr<'_>) -> McResult<Ty<'tcx>> {
        self.resolve_type_vars_or_error(expr.hir_id, self.tables.expr_ty_opt(expr))
    }

    crate fn expr_ty_adjusted(&self, expr: &hir::Expr<'_>) -> McResult<Ty<'tcx>> {
        self.resolve_type_vars_or_error(expr.hir_id, self.tables.expr_ty_adjusted_opt(expr))
    }

    /// Returns the type of value that this pattern matches against.
    /// Some non-obvious cases:
    ///
    /// - a `ref x` binding matches against a value of type `T` and gives
    ///   `x` the type `&T`; we return `T`.
    /// - a pattern with implicit derefs (thanks to default binding
    ///   modes #42640) may look like `Some(x)` but in fact have
    ///   implicit deref patterns attached (e.g., it is really
    ///   `&Some(x)`). In that case, we return the "outermost" type
    ///   (e.g., `&Option<T>).
    crate fn pat_ty_adjusted(&self, pat: &hir::Pat<'_>) -> McResult<Ty<'tcx>> {
        // Check for implicit `&` types wrapping the pattern; note
        // that these are never attached to binding patterns, so
        // actually this is somewhat "disjoint" from the code below
        // that aims to account for `ref x`.
        if let Some(vec) = self.tables.pat_adjustments().get(pat.hir_id) {
            if let Some(first_ty) = vec.first() {
                debug!("pat_ty(pat={:?}) found adjusted ty `{:?}`", pat, first_ty);
                return Ok(first_ty);
            }
        }

        self.pat_ty_unadjusted(pat)
    }

    /// Like `pat_ty`, but ignores implicit `&` patterns.
    fn pat_ty_unadjusted(&self, pat: &hir::Pat<'_>) -> McResult<Ty<'tcx>> {
        let base_ty = self.node_ty(pat.hir_id)?;
        debug!("pat_ty(pat={:?}) base_ty={:?}", pat, base_ty);

        // This code detects whether we are looking at a `ref x`,
        // and if so, figures out what the type *being borrowed* is.
        let ret_ty = match pat.kind {
            PatKind::Binding(..) => {
                let bm =
                    *self.tables.pat_binding_modes().get(pat.hir_id).expect("missing binding mode");

                if let ty::BindByReference(_) = bm {
                    // a bind-by-ref means that the base_ty will be the type of the ident itself,
                    // but what we want here is the type of the underlying value being borrowed.
                    // So peel off one-level, turning the &T into T.
                    match base_ty.builtin_deref(false) {
                        Some(t) => t.ty,
                        None => {
                            debug!("By-ref binding of non-derefable type {:?}", base_ty);
                            return Err(());
                        }
                    }
                } else {
                    base_ty
                }
            }
            _ => base_ty,
        };
        debug!("pat_ty(pat={:?}) ret_ty={:?}", pat, ret_ty);

        Ok(ret_ty)
    }

    crate fn cat_expr(&self, expr: &hir::Expr<'_>) -> McResult<Place<'tcx>> {
        // This recursion helper avoids going through *too many*
        // adjustments, since *only* non-overloaded deref recurses.
        fn helper<'a, 'tcx>(
            mc: &MemCategorizationContext<'a, 'tcx>,
            expr: &hir::Expr<'_>,
            adjustments: &[adjustment::Adjustment<'tcx>],
        ) -> McResult<Place<'tcx>> {
            match adjustments.split_last() {
                None => mc.cat_expr_unadjusted(expr),
                Some((adjustment, previous)) => {
                    mc.cat_expr_adjusted_with(expr, || helper(mc, expr, previous), adjustment)
                }
            }
        }

        helper(self, expr, self.tables.expr_adjustments(expr))
    }

    crate fn cat_expr_adjusted(
        &self,
        expr: &hir::Expr<'_>,
        previous: Place<'tcx>,
        adjustment: &adjustment::Adjustment<'tcx>,
    ) -> McResult<Place<'tcx>> {
        self.cat_expr_adjusted_with(expr, || Ok(previous), adjustment)
    }

    fn cat_expr_adjusted_with<F>(
        &self,
        expr: &hir::Expr<'_>,
        previous: F,
        adjustment: &adjustment::Adjustment<'tcx>,
    ) -> McResult<Place<'tcx>>
    where
        F: FnOnce() -> McResult<Place<'tcx>>,
    {
        debug!("cat_expr_adjusted_with({:?}): {:?}", adjustment, expr);
        let target = self.resolve_vars_if_possible(&adjustment.target);
        match adjustment.kind {
            adjustment::Adjust::Deref(overloaded) => {
                // Equivalent to *expr or something similar.
                let base = if let Some(deref) = overloaded {
                    let ref_ty = self
                        .tcx()
                        .mk_ref(deref.region, ty::TypeAndMut { ty: target, mutbl: deref.mutbl });
                    self.cat_rvalue(expr.hir_id, expr.span, ref_ty)
                } else {
                    previous()?
                };
                self.cat_deref(expr, base)
            }

            adjustment::Adjust::NeverToAny
            | adjustment::Adjust::Pointer(_)
            | adjustment::Adjust::Borrow(_) => {
                // Result is an rvalue.
                Ok(self.cat_rvalue(expr.hir_id, expr.span, target))
            }
        }
    }

    crate fn cat_expr_unadjusted(&self, expr: &hir::Expr<'_>) -> McResult<Place<'tcx>> {
        debug!("cat_expr: id={} expr={:?}", expr.hir_id, expr);

        let expr_ty = self.expr_ty(expr)?;
        match expr.kind {
            hir::ExprKind::Unary(hir::UnOp::UnDeref, ref e_base) => {
                if self.tables.is_method_call(expr) {
                    self.cat_overloaded_place(expr, e_base)
                } else {
                    let base = self.cat_expr(&e_base)?;
                    self.cat_deref(expr, base)
                }
            }

            hir::ExprKind::Field(ref base, _) => {
                let base = self.cat_expr(&base)?;
                debug!("cat_expr(cat_field): id={} expr={:?} base={:?}", expr.hir_id, expr, base);
                Ok(self.cat_projection(expr, base, expr_ty))
            }

            hir::ExprKind::Index(ref base, _) => {
                if self.tables.is_method_call(expr) {
                    // If this is an index implemented by a method call, then it
                    // will include an implicit deref of the result.
                    // The call to index() returns a `&T` value, which
                    // is an rvalue. That is what we will be
                    // dereferencing.
                    self.cat_overloaded_place(expr, base)
                } else {
                    let base = self.cat_expr(&base)?;
                    Ok(self.cat_projection(expr, base, expr_ty))
                }
            }

            hir::ExprKind::Path(ref qpath) => {
                let res = self.tables.qpath_res(qpath, expr.hir_id);
                self.cat_res(expr.hir_id, expr.span, expr_ty, res)
            }

            hir::ExprKind::Type(ref e, _) => self.cat_expr(&e),

            hir::ExprKind::AddrOf(..)
            | hir::ExprKind::Call(..)
            | hir::ExprKind::Assign(..)
            | hir::ExprKind::AssignOp(..)
            | hir::ExprKind::Closure(..)
            | hir::ExprKind::Ret(..)
            | hir::ExprKind::Unary(..)
            | hir::ExprKind::Yield(..)
            | hir::ExprKind::MethodCall(..)
            | hir::ExprKind::Cast(..)
            | hir::ExprKind::DropTemps(..)
            | hir::ExprKind::Array(..)
            | hir::ExprKind::Tup(..)
            | hir::ExprKind::Binary(..)
            | hir::ExprKind::Block(..)
            | hir::ExprKind::Loop(..)
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Lit(..)
            | hir::ExprKind::Break(..)
            | hir::ExprKind::Continue(..)
            | hir::ExprKind::Struct(..)
            | hir::ExprKind::Repeat(..)
            | hir::ExprKind::InlineAsm(..)
            | hir::ExprKind::Box(..)
            | hir::ExprKind::Err => Ok(self.cat_rvalue(expr.hir_id, expr.span, expr_ty)),
        }
    }

    crate fn cat_res(
        &self,
        hir_id: hir::HirId,
        span: Span,
        expr_ty: Ty<'tcx>,
        res: Res,
    ) -> McResult<Place<'tcx>> {
        debug!("cat_res: id={:?} expr={:?} def={:?}", hir_id, expr_ty, res);

        match res {
            Res::Def(DefKind::Ctor(..), _)
            | Res::Def(DefKind::Const, _)
            | Res::Def(DefKind::ConstParam, _)
            | Res::Def(DefKind::AssocConst, _)
            | Res::Def(DefKind::Fn, _)
            | Res::Def(DefKind::AssocFn, _)
            | Res::SelfCtor(..) => Ok(self.cat_rvalue(hir_id, span, expr_ty)),

            Res::Def(DefKind::Static, _) => Ok(Place {
                hir_id,
                span,
                ty: expr_ty,
                base: PlaceBase::StaticItem,
                projections: Vec::new(),
            }),

            Res::Local(var_id) => {
                if self.upvars.map_or(false, |upvars| upvars.contains_key(&var_id)) {
                    self.cat_upvar(hir_id, span, var_id)
                } else {
                    Ok(Place {
                        hir_id,
                        span,
                        ty: expr_ty,
                        base: PlaceBase::Local(var_id),
                        projections: Vec::new(),
                    })
                }
            }

            def => span_bug!(span, "unexpected definition in memory categorization: {:?}", def),
        }
    }

    /// Categorize an upvar.
    ///
    /// Note: the actual upvar access contains invisible derefs of closure
    /// environment and upvar reference as appropriate. Only regionck cares
    /// about these dereferences, so we let it compute them as needed.
    fn cat_upvar(
        &self,
        hir_id: hir::HirId,
        span: Span,
        var_id: hir::HirId,
    ) -> McResult<Place<'tcx>> {
        let closure_expr_def_id = self.body_owner;

        let upvar_id = ty::UpvarId {
            var_path: ty::UpvarPath { hir_id: var_id },
            closure_expr_id: closure_expr_def_id.to_local(),
        };
        let var_ty = self.node_ty(var_id)?;

        let ret = Place {
            hir_id,
            span,
            ty: var_ty,
            base: PlaceBase::Upvar(upvar_id),
            projections: Vec::new(),
        };

        debug!("cat_upvar ret={:?}", ret);
        Ok(ret)
    }

    crate fn cat_rvalue(&self, hir_id: hir::HirId, span: Span, expr_ty: Ty<'tcx>) -> Place<'tcx> {
        debug!("cat_rvalue hir_id={:?}, expr_ty={:?}, span={:?}", hir_id, expr_ty, span);
        let ret =
            Place { hir_id, span, base: PlaceBase::Rvalue, projections: Vec::new(), ty: expr_ty };
        debug!("cat_rvalue ret={:?}", ret);
        ret
    }

    crate fn cat_projection<N: HirNode>(
        &self,
        node: &N,
        base_place: Place<'tcx>,
        ty: Ty<'tcx>,
    ) -> Place<'tcx> {
        let mut projections = base_place.projections;
        projections.push(Projection::Other);
        let ret = Place {
            hir_id: node.hir_id(),
            span: node.span(),
            ty,
            base: base_place.base,
            projections,
        };
        debug!("cat_field ret {:?}", ret);
        ret
    }

    fn cat_overloaded_place(
        &self,
        expr: &hir::Expr<'_>,
        base: &hir::Expr<'_>,
    ) -> McResult<Place<'tcx>> {
        debug!("cat_overloaded_place(expr={:?}, base={:?})", expr, base);

        // Reconstruct the output assuming it's a reference with the
        // same region and mutability as the receiver. This holds for
        // `Deref(Mut)::Deref(_mut)` and `Index(Mut)::index(_mut)`.
        let place_ty = self.expr_ty(expr)?;
        let base_ty = self.expr_ty_adjusted(base)?;

        let (region, mutbl) = match base_ty.kind {
            ty::Ref(region, _, mutbl) => (region, mutbl),
            _ => span_bug!(expr.span, "cat_overloaded_place: base is not a reference"),
        };
        let ref_ty = self.tcx().mk_ref(region, ty::TypeAndMut { ty: place_ty, mutbl });

        let base = self.cat_rvalue(expr.hir_id, expr.span, ref_ty);
        self.cat_deref(expr, base)
    }

    fn cat_deref(&self, node: &impl HirNode, base_place: Place<'tcx>) -> McResult<Place<'tcx>> {
        debug!("cat_deref: base_place={:?}", base_place);

        let base_ty = base_place.ty;
        let deref_ty = match base_ty.builtin_deref(true) {
            Some(mt) => mt.ty,
            None => {
                debug!("explicit deref of non-derefable type: {:?}", base_ty);
                return Err(());
            }
        };
        let mut projections = base_place.projections;
        projections.push(Projection::Deref(base_ty));

        let ret = Place {
            hir_id: node.hir_id(),
            span: node.span(),
            ty: deref_ty,
            base: base_place.base,
            projections,
        };
        debug!("cat_deref ret {:?}", ret);
        Ok(ret)
    }

    crate fn cat_pattern<F>(
        &self,
        place: Place<'tcx>,
        pat: &hir::Pat<'_>,
        mut op: F,
    ) -> McResult<()>
    where
        F: FnMut(&Place<'tcx>, &hir::Pat<'_>),
    {
        self.cat_pattern_(place, pat, &mut op)
    }

    // FIXME(#19596) This is a workaround, but there should be a better way to do this
    fn cat_pattern_<F>(
        &self,
        mut place: Place<'tcx>,
        pat: &hir::Pat<'_>,
        op: &mut F,
    ) -> McResult<()>
    where
        F: FnMut(&Place<'tcx>, &hir::Pat<'_>),
    {
        // Here, `place` is the `Place` being matched and pat is the pattern it
        // is being matched against.
        //
        // In general, the way that this works is that we walk down the pattern,
        // constructing a `Place` that represents the path that will be taken
        // to reach the value being matched.

        debug!("cat_pattern(pat={:?}, place={:?})", pat, place);

        // If (pattern) adjustments are active for this pattern, adjust the `Place` correspondingly.
        // `Place`s are constructed differently from patterns. For example, in
        //
        // ```
        // match foo {
        //     &&Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // the pattern `&&Some(x,)` is represented as `Ref { Ref { TupleStruct }}`. To build the
        // corresponding `Place` we start with the `Place` for `foo`, and then, by traversing the
        // pattern, try to answer the question: given the address of `foo`, how is `x` reached?
        //
        // `&&Some(x,)` `place_foo`
        //  `&Some(x,)` `deref { place_foo}`
        //   `Some(x,)` `deref { deref { place_foo }}`
        //        (x,)` `field0 { deref { deref { place_foo }}}` <- resulting place
        //
        // The above example has no adjustments. If the code were instead the (after adjustments,
        // equivalent) version
        //
        // ```
        // match foo {
        //     Some(x, ) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // Then we see that to get the same result, we must start with
        // `deref { deref { place_foo }}` instead of `place_foo` since the pattern is now `Some(x,)`
        // and not `&&Some(x,)`, even though its assigned type is that of `&&Some(x,)`.
        for _ in 0..self.tables.pat_adjustments().get(pat.hir_id).map(|v| v.len()).unwrap_or(0) {
            debug!("cat_pattern: applying adjustment to place={:?}", place);
            place = self.cat_deref(pat, place)?;
        }
        let place = place; // lose mutability
        debug!("cat_pattern: applied adjustment derefs to get place={:?}", place);

        // Invoke the callback, but only now, after the `place` has adjusted.
        //
        // To see that this makes sense, consider `match &Some(3) { Some(x) => { ... }}`. In that
        // case, the initial `place` will be that for `&Some(3)` and the pattern is `Some(x)`. We
        // don't want to call `op` with these incompatible values. As written, what happens instead
        // is that `op` is called with the adjusted place (that for `*&Some(3)`) and the pattern
        // `Some(x)` (which matches). Recursing once more, `*&Some(3)` and the pattern `Some(x)`
        // result in the place `Downcast<Some>(*&Some(3)).0` associated to `x` and invoke `op` with
        // that (where the `ref` on `x` is implied).
        op(&place, pat);

        match pat.kind {
            PatKind::TupleStruct(_, ref subpats, _) | PatKind::Tuple(ref subpats, _) => {
                // S(p1, ..., pN) or (p1, ..., pN)
                for subpat in subpats.iter() {
                    let subpat_ty = self.pat_ty_adjusted(&subpat)?;
                    let sub_place = self.cat_projection(pat, place.clone(), subpat_ty);
                    self.cat_pattern_(sub_place, &subpat, op)?;
                }
            }

            PatKind::Struct(_, field_pats, _) => {
                // S { f1: p1, ..., fN: pN }
                for fp in field_pats {
                    let field_ty = self.pat_ty_adjusted(&fp.pat)?;
                    let field_place = self.cat_projection(pat, place.clone(), field_ty);
                    self.cat_pattern_(field_place, &fp.pat, op)?;
                }
            }

            PatKind::Or(pats) => {
                for pat in pats {
                    self.cat_pattern_(place.clone(), &pat, op)?;
                }
            }

            PatKind::Binding(.., Some(ref subpat)) => {
                self.cat_pattern_(place, &subpat, op)?;
            }

            PatKind::Box(ref subpat) | PatKind::Ref(ref subpat, _) => {
                // box p1, &p1, &mut p1.  we can ignore the mutability of
                // PatKind::Ref since that information is already contained
                // in the type.
                let subplace = self.cat_deref(pat, place)?;
                self.cat_pattern_(subplace, &subpat, op)?;
            }

            PatKind::Slice(before, ref slice, after) => {
                let element_ty = match place.ty.builtin_index() {
                    Some(ty) => ty,
                    None => {
                        debug!("explicit index of non-indexable type {:?}", place);
                        return Err(());
                    }
                };
                let elt_place = self.cat_projection(pat, place.clone(), element_ty);
                for before_pat in before {
                    self.cat_pattern_(elt_place.clone(), &before_pat, op)?;
                }
                if let Some(ref slice_pat) = *slice {
                    let slice_pat_ty = self.pat_ty_adjusted(&slice_pat)?;
                    let slice_place = self.cat_projection(pat, place, slice_pat_ty);
                    self.cat_pattern_(slice_place, &slice_pat, op)?;
                }
                for after_pat in after {
                    self.cat_pattern_(elt_place.clone(), &after_pat, op)?;
                }
            }

            PatKind::Path(_)
            | PatKind::Binding(.., None)
            | PatKind::Lit(..)
            | PatKind::Range(..)
            | PatKind::Wild => {
                // always ok
            }
        }

        Ok(())
    }
}
