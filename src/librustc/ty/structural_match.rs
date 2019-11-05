use crate::hir;
use rustc::infer::InferCtxt;
use rustc::traits::{self, ConstPatternStructural, TraitEngine};
use rustc::traits::ObligationCause;
use rustc::hir::def::{DefKind, Res};
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::lint;

use rustc_data_structures::fx::{FxHashSet};

use syntax_pos::Span;

use crate::ty::{self, AdtDef, ToPredicate, Ty, TyCtxt};
use crate::ty::fold::{TypeFoldable, TypeVisitor};

#[derive(Debug)]
pub enum NonStructuralMatchTy<'tcx> {
    Adt(&'tcx AdtDef),
    Param,
}

pub fn report_structural_match_violation(tcx: TyCtxt<'tcx>,
                                         non_sm_ty: NonStructuralMatchTy<'tcx>,
                                         id: hir::HirId,
                                         span: Span,
                                         warn_instead_of_hard_error: bool) {
    let adt_def = match non_sm_ty {
        ty::NonStructuralMatchTy::Adt(adt_def) => adt_def,
        ty::NonStructuralMatchTy::Param =>
            bug!("use of constant whose type is a parameter inside a pattern"),
    };
    let path = tcx.def_path_str(adt_def.did);
    let msg = format!("to use a constant of type `{}` in a pattern, \
                       `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                      path, path);


    if warn_instead_of_hard_error {
        tcx.lint_hir(lint::builtin::INDIRECT_STRUCTURAL_MATCH, id, span, &msg);
    } else {
        // span_fatal avoids ICE from resolution of non-existent method (rare case).
        tcx.sess.span_fatal(span, &msg);
    }
}

/// This method traverses the structure of `ty`, trying to find an
/// instance of an ADT (i.e. struct or enum) that was declared without
/// the `#[structural_match]` attribute, or a generic type parameter
/// (which cannot be determined to be `structural_match`).
///
/// The "structure of a type" includes all components that would be
/// considered when doing a pattern match on a constant of that
/// type.
///
///  * This means this method descends into fields of structs/enums,
///    and also descends into the inner type `T` of `&T` and `&mut T`
///
///  * The traversal doesn't dereference unsafe pointers (`*const T`,
///    `*mut T`), and it does not visit the type arguments of an
///    instantiated generic like `PhantomData<T>`.
///
/// The reason we do this search is Rust currently require all ADTs
/// reachable from a constant's type to be annotated with
/// `#[structural_match]`, an attribute which essentially says that
/// the implementation of `PartialEq::eq` behaves *equivalently* to a
/// comparison against the unfolded structure.
///
/// For more background on why Rust has this requirement, and issues
/// that arose when the requirement was not enforced completely, see
/// Rust RFC 1445, rust-lang/rust#61188, and rust-lang/rust#62307.
pub fn search_type_for_structural_match_violation<'tcx>(
    id: hir::HirId,
    span: Span,
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<NonStructuralMatchTy<'tcx>> {
    // FIXME: consider passing in an `infcx` from the outside.
    tcx.infer_ctxt().enter(|infcx| {
        let mut search = SearchTy { id, span, infcx, found: None, seen: FxHashSet::default() };
        ty.visit_with(&mut search);
        search.found
    })
}

pub fn search_const_rhs_for_structural_match_violation<'tcx>(
    infcx: &InferCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    const_def_id: DefId,
    pattern_id: hir::HirId,
    pattern_span: Span,
) {
    let def_id = const_def_id;
    let id = pattern_id;
    let span = pattern_span;

    // Traverses right-hand side of const definition, looking for:
    //
    // 1. literals constructing ADTs that do not implement `Structural`
    //    (rust-lang/rust#62614), and
    //
    // 2. non-scalar types that do not implement `PartialEq` (which would
    //    cause codegen to ICE).

    debug!("walk_const_value_looking_for_nonstructural_adt \
            def_id: {:?} id: {:?} span: {:?}", def_id, id, span);

    assert!(def_id.is_local());

    let const_hir_id: hir::HirId = infcx.tcx.hir().local_def_id_to_hir_id(def_id.to_local());
    debug!("walk_const_value_looking_for_nonstructural_adt const_hir_id: {:?}", const_hir_id);
    let body_id = infcx.tcx.hir().body_owned_by(const_hir_id);
    debug!("walk_const_value_looking_for_nonstructural_adt body_id: {:?}", body_id);
    let body_tables = infcx.tcx.body_tables(body_id);
    let body = infcx.tcx.hir().body(body_id);
    let mut v = SearchHirExpr {
        infcx, param_env, body_tables, id, span,
        structural_fulfillment_cx: traits::FulfillmentContext::new(),
        partialeq_fulfillment_cx: traits::FulfillmentContext::new(),
    };
    v.visit_body(body);
    if let Err(errs) = v.structural_fulfillment_cx.select_all_or_error(&v.infcx) {
        v.infcx.report_fulfillment_errors(&errs, None, false);
    }
    if let Err(errs) = v.partialeq_fulfillment_cx.select_all_or_error(&v.infcx) {
        for err in errs {
            let traits::FulfillmentError { obligation, code: _, points_at_arg_span: _ } = err;
            if let ty::Predicate::Trait(pred) = obligation.predicate {
                let ty = pred.skip_binder().self_ty();
                infcx.tcx.sess.span_fatal(
                    span,
                    &format!("to use a constant of type `{}` in a pattern, \
                              `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                             ty, ty));
            } else {
                bug!("only should have trait predicates");
            }
        }
    }
}

fn register_structural_bound(fulfillment_cx: &mut traits::FulfillmentContext<'tcx>,
                             id: hir::HirId,
                             span: Span,
                             infcx: &InferCtxt<'_, 'tcx>,
                             adt_ty: Ty<'tcx>)
{
    let cause = ObligationCause::new(span, id, ConstPatternStructural);
    // require `#[derive(PartialEq)]`
    let structural_peq_def_id = infcx.tcx.lang_items().structural_peq_trait().unwrap();
    fulfillment_cx.register_bound(
        infcx, ty::ParamEnv::empty(), adt_ty, structural_peq_def_id, cause);
    // for now, require `#[derive(Eq)]`. (Doing so is a hack to work around
    // the type `for<'a> fn(&'a ())` failing to implement `Eq` itself.)
    let cause = ObligationCause::new(span, id, ConstPatternStructural);
    let structural_teq_def_id = infcx.tcx.lang_items().structural_teq_trait().unwrap();
    fulfillment_cx.register_bound(
        infcx, ty::ParamEnv::empty(), adt_ty, structural_teq_def_id, cause);
}

/// This method returns true if and only if `adt_ty` itself has been marked as
/// eligible for structural-match: namely, if it implements both
/// `StructuralPartialEq` and `StructuralEq` (which are respectively injected by
/// `#[derive(PartialEq)]` and `#[derive(Eq)]`).
///
/// Note that this does *not* recursively check if the substructure of `adt_ty`
/// implements the traits.
pub fn type_marked_structural(id: hir::HirId,
                              span: Span,
                              infcx: &InferCtxt<'_, 'tcx>,
                              adt_ty: Ty<'tcx>)
                              -> bool
{
    let mut fulfillment_cx = traits::FulfillmentContext::new();

    register_structural_bound(&mut fulfillment_cx, id, span, infcx, adt_ty);

    // We deliberately skip *reporting* fulfillment errors (via
    // `report_fulfillment_errors`), for two reasons:
    //
    // 1. The error messages would mention `std::marker::StructuralPartialEq`
    //    (a trait which is solely meant as an implementation detail
    //    for now), and
    //
    // 2. We are sometimes doing future-incompatibility lints for
    //    now, so we do not want unconditional errors here.
    fulfillment_cx.select_all_or_error(infcx).is_ok()
}

/// This implements the traversal over the structure of a given type to try to
/// find instances of ADTs (specifically structs or enums) that do not implement
/// the structural-match traits (`StructuralPartialEq` and `StructuralEq`).
struct SearchTy<'a, 'tcx> {
    id: hir::HirId,
    span: Span,

    infcx: InferCtxt<'a, 'tcx>,

    /// Records first ADT that does not implement a structural-match trait.
    found: Option<NonStructuralMatchTy<'tcx>>,

    /// Tracks ADTs previously encountered during search, so that
    /// we will not recur on them again.
    seen: FxHashSet<hir::def_id::DefId>,
}

impl SearchTy<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn type_marked_structural(&self, adt_ty: Ty<'tcx>) -> bool {
        type_marked_structural(self.id, self.span, &self.infcx, adt_ty)
    }
}

impl<'a, 'tcx> TypeVisitor<'tcx> for SearchTy<'a, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        debug!("SearchTy visiting ty: {:?}", ty);

        let (adt_def, substs) = match ty.kind {
            ty::Adt(adt_def, substs) => (adt_def, substs),
            ty::Param(_) => {
                self.found = Some(NonStructuralMatchTy::Param);
                return true; // Stop visiting.
            }
            ty::RawPtr(..) => {
                // structural-match ignores substructure of
                // `*const _`/`*mut _`, so skip `super_visit_with`.
                //
                // For example, if you have:
                // ```
                // struct NonStructural;
                // #[derive(PartialEq, Eq)]
                // struct T(*const NonStructural);
                // const C: T = T(std::ptr::null());
                // ```
                //
                // Even though `NonStructural` does not implement `PartialEq`,
                // structural equality on `T` does not recur into the raw
                // pointer. Therefore, one can still use `C` in a pattern.

                // (But still tell caller to continue search.)
                return false;
            }
            ty::FnDef(..) | ty::FnPtr(..) => {
                // types of formals and return in `fn(_) -> _` are also irrelevant;
                // so we do not recur into them via `super_visit_with`
                //
                // (But still tell caller to continue search.)
                return false;
            }
            ty::Array(_, n) if {
                n.try_eval_usize(self.tcx(), ty::ParamEnv::reveal_all()) == Some(0)
            } => {
                // rust-lang/rust#62336: ignore type of contents
                // for empty array.
                return false;
            }
            _ => {
                ty.super_visit_with(self);
                return false;
            }
        };

        if !self.seen.insert(adt_def.did) {
            debug!("SearchTy already seen adt_def: {:?}", adt_def);
            // let caller continue its search
            return false;
        }

        if !self.type_marked_structural(ty) {
            debug!("SearchTy found ty: {:?}", ty);
            self.found = Some(NonStructuralMatchTy::Adt(&adt_def));
            return true; // Halt visiting!
        }

        // structural-match does not care about the
        // instantiation of the generics in an ADT (it
        // instead looks directly at its fields outside
        // this match), so we skip super_visit_with.
        //
        // (Must not recur on substs for `PhantomData<T>` cf
        // rust-lang/rust#55028 and rust-lang/rust#55837; but also
        // want to skip substs when only uses of generic are
        // behind unsafe pointers `*const T`/`*mut T`.)

        // even though we skip super_visit_with, we must recur on
        // fields of ADT.
        let tcx = self.tcx();
        for field_ty in adt_def.all_fields().map(|field| field.ty(tcx, substs)) {
            if field_ty.visit_with(self) {
                // found an ADT without structural-match; halt visiting!
                assert!(self.found.is_some());
                return true;
            }
        }

        // Even though we do not want to recur on substs, we do
        // want our caller to continue its own search.
        false
    }
}

struct SearchHirExpr<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_tables: &'a ty::TypeckTables<'tcx>,
    structural_fulfillment_cx: traits::FulfillmentContext<'tcx>,
    partialeq_fulfillment_cx: traits::FulfillmentContext<'tcx>,
    id: hir::HirId,
    span: Span,
}

impl<'a, 'tcx> SearchHirExpr<'a, 'tcx> {
    fn recur_into_const_rhs(&self, const_def_id: DefId) {
        assert!(const_def_id.is_local());
        search_const_rhs_for_structural_match_violation(
            &self.infcx, self.param_env, const_def_id, self.id, self.span);
    }

    fn search_for_structural_match_violation(&self, adt_ty: Ty<'tcx>) {
        if let Some(witness) =
            search_type_for_structural_match_violation(self.id, self.span, self.infcx.tcx, adt_ty)
        {
            // the logic about when to shift from value-based to type-based
            // reasoning is new and open to redesign, so only warn about
            // violations there for now.
            let warn_instead_of_hard_error = true;
            report_structural_match_violation(
                self.infcx.tcx, witness, self.id, self.span, warn_instead_of_hard_error);
        }
    }

    fn register_structural_bound(&mut self, adt_ty: Ty<'tcx>) {
        register_structural_bound(
            &mut self.structural_fulfillment_cx, self.id, self.span, &self.infcx, adt_ty);
    }

    fn register_partial_eq_bound(&mut self, ty: Ty<'tcx>) {
        let cause = ObligationCause::new(self.span, self.id, ConstPatternStructural);
        let partial_eq_def_id = self.infcx.tcx.lang_items().eq_trait().unwrap();

        // Note: Cannot use register_bound here, because it requires (but does
        // not check) that the given trait has no type parameters apart from
        // `Self`, but `PartialEq` has a type parameter that defaults to `Self`.
        let trait_ref = ty::TraitRef {
            def_id: partial_eq_def_id,
            substs: self.infcx.tcx.mk_substs_trait(ty, &[ty.into()]),
        };
        let obligation = traits::Obligation {
            cause,
            recursion_depth: 0,
            param_env: self.param_env,
            predicate: trait_ref.to_predicate(),
        };
        self.partialeq_fulfillment_cx.register_predicate_obligation(&self.infcx, obligation);
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for SearchHirExpr<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v>
    {
        NestedVisitorMap::None
    }
    fn visit_expr(&mut self, ex: &'v hir::Expr) {
        // When we inspect the expression on a RHS of a const, we may be able to
        // prove that it can participate in a match pattern, without worrying
        // about its actual type being `PartialEq`.
        //
        // In an ideal world we would not bother with side-stepping the
        // `PartialEq` and just unconditionally check that the types of a
        // constant in a match pattern are all `PartialEq`. However, one big
        // exception is types with `for <'a> fn(...)` in them; today these are
        // not `PartialEq` due to technical limitations.
        //
        // So we cut the gordian knot here: the expression analysis can allow
        // the omission of the `PartialEq` check.
        let mut impose_partialeq_bound = true;

        let ty = self.body_tables.expr_ty(ex);
        match &ex.kind {
            hir::ExprKind::Struct(..) => {
                // register structural requirement ...
                self.register_structural_bound(ty);
                // ... and continue expression-based traversal
                intravisit::walk_expr(self, ex)
            }

            hir::ExprKind::Path(qpath) => {
                let res = self.body_tables.qpath_res(qpath, ex.hir_id);
                match res {
                    Res::Def(DefKind::Const, def_id) |
                    Res::Def(DefKind::AssocConst, def_id) => {
                        let substs = self.body_tables.node_substs(ex.hir_id);
                        if let Some(instance) = ty::Instance::resolve(
                            self.infcx.tcx, self.param_env, def_id, substs)
                        {
                            let const_def_id = instance.def_id();
                            if const_def_id.is_local() {
                                self.recur_into_const_rhs(const_def_id);
                            } else {
                                // abstraction barrer for non-local definition;
                                // traverse `typeof(expr)` instead.
                                debug!("SearchHirExpr switch to type analysis \
                                        for expr: {:?} ty: {:?}", ex, ty);
                                self.search_for_structural_match_violation(ty);
                                impose_partialeq_bound = false;
                            }
                        } else {
                            self.infcx.tcx.sess.delay_span_bug(self.span, &format!(
                                "SearchHirExpr didn't resolve def_id: {:?}", def_id));
                        }
                    }
                    Res::Def(DefKind::Ctor(..), _def_id) => {
                        self.register_structural_bound(ty);
                    }
                    _ => {
                        debug!("SearchHirExpr ExprKind::Path res: {:?} \
                                traverse type instead", res);
                    }
                }
            }

            hir::ExprKind::Box(..) |
            hir::ExprKind::Array(..) |
            hir::ExprKind::Repeat(..) |
            hir::ExprKind::Tup(..) |
            hir::ExprKind::Type(..) |
            hir::ExprKind::DropTemps(..) |
            hir::ExprKind::AddrOf(..) => {
                // continue expression-based traversal
                intravisit::walk_expr(self, ex)
            }

            hir::ExprKind::Block(block, _opt_label) => {
                // skip the statements, focus solely on the return expression
                if let Some(ex) = &block.expr {
                    intravisit::walk_expr(self, ex)
                }
            }
            hir::ExprKind::Match(_input, arms, _match_source) => {
                // skip the input, focus solely on the arm bodies
                for a in arms.iter() {
                    intravisit::walk_expr(self, &a.body)
                }
            }

            hir::ExprKind::Index(base, _index) => {
                // skip the index, focus solely on the base content.
                // (alternative would be to do type-based analysis, which would
                // be even more conservative).
                intravisit::walk_expr(self, base);
            }

            hir::ExprKind::Loop(..) |
            hir::ExprKind::Call(..) |
            hir::ExprKind::MethodCall(..) |
            hir::ExprKind::Binary(..) |
            hir::ExprKind::Unary(..) |
            hir::ExprKind::Cast(..) |
            hir::ExprKind::Closure(..) |
            hir::ExprKind::Assign(..) |
            hir::ExprKind::AssignOp(..) |
            hir::ExprKind::Field(..) |
            hir::ExprKind::Break(..) |
            hir::ExprKind::Continue(..) |
            hir::ExprKind::Ret(..) |
            hir::ExprKind::Yield(..) |
            hir::ExprKind::InlineAsm(..) |
            hir::ExprKind::Lit(..) |
            hir::ExprKind::Err => {
                // abstraction barrier for non-trivial expression; traverse
                // `typeof(expr)` instead of expression itself.
                debug!("SearchHirExpr switch to type analysis for expr: {:?} ty: {:?}", ex, ty);
                self.search_for_structural_match_violation(ty);
                impose_partialeq_bound = false;
            }
        }

        if impose_partialeq_bound && !ty.is_scalar() {
            debug!("SearchHirExpr registering PartialEq bound for non-scalar ty: {:?}", ty);
            self.register_partial_eq_bound(ty);
        }
    }
}
