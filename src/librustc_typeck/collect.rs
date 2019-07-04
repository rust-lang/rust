//! "Collection" is the process of determining the type and other external
//! details of each item in Rust. Collection is specifically concerned
//! with *inter-procedural* things -- for example, for a function
//! definition, collection will figure out the type and signature of the
//! function, but it will not visit the *body* of the function in any way,
//! nor examine type annotations on local variables (that's the job of
//! type *checking*).
//!
//! Collecting is ultimately defined by a bundle of queries that
//! inquire after various facts about the items in the crate (e.g.,
//! `type_of`, `generics_of`, `predicates_of`, etc). See the `provide` function
//! for the full set.
//!
//! At present, however, we do run collection across all items in the
//! crate as a kind of pass. This should eventually be factored away.

use crate::astconv::{AstConv, Bounds, SizedByDefault};
use crate::constrained_generic_params as cgp;
use crate::check::intrinsic::intrisic_operation_unsafety;
use crate::lint;
use crate::middle::resolve_lifetime as rl;
use crate::middle::weak_lang_items;
use rustc::mir::mono::Linkage;
use rustc::ty::query::Providers;
use rustc::ty::subst::{Subst, InternalSubsts};
use rustc::ty::util::Discr;
use rustc::ty::util::IntTypeExt;
use rustc::ty::subst::UnpackedKind;
use rustc::ty::{self, AdtKind, DefIdTree, ToPolyTraitRef, Ty, TyCtxt, Const};
use rustc::ty::{ReprOptions, ToPredicate};
use rustc::util::captures::Captures;
use rustc::util::nodemap::FxHashMap;
use rustc_target::spec::abi;

use syntax::ast;
use syntax::ast::{Ident, MetaItemKind};
use syntax::attr::{InlineAttr, OptimizeAttr, list_contains_name, mark_used};
use syntax::source_map::Spanned;
use syntax::feature_gate;
use syntax::symbol::{InternedString, kw, Symbol, sym};
use syntax_pos::{Span, DUMMY_SP};

use rustc::hir::def::{CtorKind, Res, DefKind};
use rustc::hir::Node;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc::hir::GenericParamKind;
use rustc::hir::{self, CodegenFnAttrFlags, CodegenFnAttrs, Unsafety};

use errors::{Applicability, DiagnosticId};

use std::iter;

struct OnlySelfBounds(bool);

///////////////////////////////////////////////////////////////////////////
// Main entry point

fn collect_mod_item_types(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CollectItemTypesVisitor { tcx }.as_deep_visitor()
    );
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        type_of,
        generics_of,
        predicates_of,
        predicates_defined_on,
        explicit_predicates_of,
        super_predicates_of,
        type_param_predicates,
        trait_def,
        adt_def,
        fn_sig,
        impl_trait_ref,
        impl_polarity,
        is_foreign_item,
        static_mutability,
        codegen_fn_attrs,
        collect_mod_item_types,
        ..*providers
    };
}

///////////////////////////////////////////////////////////////////////////

/// Context specific to some particular item. This is what implements
/// `AstConv`. It has information about the predicates that are defined
/// on the trait. Unfortunately, this predicate information is
/// available in various different forms at various points in the
/// process. So we can't just store a pointer to e.g., the AST or the
/// parsed ty form, we have to be more flexible. To this end, the
/// `ItemCtxt` is parameterized by a `DefId` that it uses to satisfy
/// `get_type_parameter_bounds` requests, drawing the information from
/// the AST (`hir::Generics`), recursively.
pub struct ItemCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
    item_def_id: DefId,
}

///////////////////////////////////////////////////////////////////////////

struct CollectItemTypesVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl Visitor<'tcx> for CollectItemTypesVisitor<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_item(&mut self, item: &'tcx hir::Item) {
        convert_item(self.tcx, item.hir_id);
        intravisit::walk_item(self, item);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
        for param in &generics.params {
            match param.kind {
                hir::GenericParamKind::Lifetime { .. } => {}
                hir::GenericParamKind::Type {
                    default: Some(_), ..
                } => {
                    let def_id = self.tcx.hir().local_def_id_from_hir_id(param.hir_id);
                    self.tcx.type_of(def_id);
                }
                hir::GenericParamKind::Type { .. } => {}
                hir::GenericParamKind::Const { .. } => {
                    let def_id = self.tcx.hir().local_def_id_from_hir_id(param.hir_id);
                    self.tcx.type_of(def_id);
                }
            }
        }
        intravisit::walk_generics(self, generics);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprKind::Closure(..) = expr.node {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(expr.hir_id);
            self.tcx.generics_of(def_id);
            self.tcx.type_of(def_id);
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        convert_trait_item(self.tcx, trait_item.hir_id);
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        convert_impl_item(self.tcx, impl_item.hir_id);
        intravisit::walk_impl_item(self, impl_item);
    }
}

///////////////////////////////////////////////////////////////////////////
// Utility types and common code for the above passes.

impl ItemCtxt<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, item_def_id: DefId) -> ItemCtxt<'tcx> {
        ItemCtxt { tcx, item_def_id }
    }

    pub fn to_ty(&self, ast_ty: &'tcx hir::Ty) -> Ty<'tcx> {
        AstConv::ast_ty_to_ty(self, ast_ty)
    }
}

impl AstConv<'tcx> for ItemCtxt<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn get_type_parameter_bounds(&self, span: Span, def_id: DefId)
                                 -> &'tcx ty::GenericPredicates<'tcx> {
        self.tcx
            .at(span)
            .type_param_predicates((self.item_def_id, def_id))
    }

    fn re_infer(
        &self,
        _: Option<&ty::GenericParamDef>,
        _: Span,
    ) -> Option<ty::Region<'tcx>> {
        None
    }

    fn ty_infer(&self, _: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx> {
        self.tcx().sess.struct_span_err_with_code(
            span,
            "the type placeholder `_` is not allowed within types on item signatures",
            DiagnosticId::Error("E0121".into()),
        ).span_label(span, "not allowed in type signatures")
         .emit();

        self.tcx().types.err
    }

    fn ct_infer(
        &self,
        _: Ty<'tcx>,
        _: Option<&ty::GenericParamDef>,
        span: Span,
    ) -> &'tcx Const<'tcx> {
        self.tcx().sess.struct_span_err_with_code(
            span,
            "the const placeholder `_` is not allowed within types on item signatures",
            DiagnosticId::Error("E0121".into()),
        ).span_label(span, "not allowed in type signatures")
         .emit();

        self.tcx().consts.err
    }

    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        if let Some(trait_ref) = poly_trait_ref.no_bound_vars() {
            self.tcx().mk_projection(item_def_id, trait_ref.substs)
        } else {
            // no late-bound regions, we can just ignore the binder
            span_err!(
                self.tcx().sess,
                span,
                E0212,
                "cannot extract an associated type from a higher-ranked trait bound \
                 in this context"
            );
            self.tcx().types.err
        }
    }

    fn normalize_ty(&self, _span: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        // types in item signatures are not normalized, to avoid undue
        // dependencies.
        ty
    }

    fn set_tainted_by_errors(&self) {
        // no obvious place to track this, so just let it go
    }

    fn record_ty(&self, _hir_id: hir::HirId, _ty: Ty<'tcx>, _span: Span) {
        // no place to record types from signatures?
    }
}

fn type_param_predicates(
    tcx: TyCtxt<'_>,
    (item_def_id, def_id): (DefId, DefId),
) -> &ty::GenericPredicates<'_> {
    use rustc::hir::*;

    // In the AST, bounds can derive from two places. Either
    // written inline like `<T : Foo>` or in a where clause like
    // `where T : Foo`.

    let param_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let param_owner = tcx.hir().ty_param_owner(param_id);
    let param_owner_def_id = tcx.hir().local_def_id_from_hir_id(param_owner);
    let generics = tcx.generics_of(param_owner_def_id);
    let index = generics.param_def_id_to_index[&def_id];
    let ty = tcx.mk_ty_param(index, tcx.hir().ty_param_name(param_id).as_interned_str());

    // Don't look for bounds where the type parameter isn't in scope.
    let parent = if item_def_id == param_owner_def_id {
        None
    } else {
        tcx.generics_of(item_def_id).parent
    };

    let result = parent.map_or(&tcx.common.empty_predicates, |parent| {
        let icx = ItemCtxt::new(tcx, parent);
        icx.get_type_parameter_bounds(DUMMY_SP, def_id)
    });
    let mut extend = None;

    let item_hir_id = tcx.hir().as_local_hir_id(item_def_id).unwrap();
    let ast_generics = match tcx.hir().get(item_hir_id) {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.node {
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::Impl(_, _, _, ref generics, ..)
                | ItemKind::Ty(_, ref generics)
                | ItemKind::Existential(ExistTy {
                    ref generics,
                    impl_trait_fn: None,
                    ..
                })
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics) => generics,
                ItemKind::Trait(_, _, ref generics, ..) => {
                    // Implied `Self: Trait` and supertrait bounds.
                    if param_id == item_hir_id {
                        let identity_trait_ref = ty::TraitRef::identity(tcx, item_def_id);
                        extend = Some((identity_trait_ref.to_predicate(), item.span));
                    }
                    generics
                }
                _ => return result,
            }
        }

        Node::ForeignItem(item) => match item.node {
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            _ => return result,
        },

        _ => return result,
    };

    let icx = ItemCtxt::new(tcx, item_def_id);
    let mut result = (*result).clone();
    result.predicates.extend(extend.into_iter());
    result.predicates
          .extend(icx.type_parameter_bounds_in_generics(ast_generics, param_id, ty,
                  OnlySelfBounds(true)));
    tcx.arena.alloc(result)
}

impl ItemCtxt<'tcx> {
    /// Finds bounds from `hir::Generics`. This requires scanning through the
    /// AST. We do this to avoid having to convert *all* the bounds, which
    /// would create artificial cycles. Instead we can only convert the
    /// bounds for a type parameter `X` if `X::Foo` is used.
    fn type_parameter_bounds_in_generics(
        &self,
        ast_generics: &'tcx hir::Generics,
        param_id: hir::HirId,
        ty: Ty<'tcx>,
        only_self_bounds: OnlySelfBounds,
    ) -> Vec<(ty::Predicate<'tcx>, Span)> {
        let from_ty_params = ast_generics
            .params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Type { .. } if param.hir_id == param_id => Some(&param.bounds),
                _ => None,
            })
            .flat_map(|bounds| bounds.iter())
            .flat_map(|b| predicates_from_bound(self, ty, b));

        let from_where_clauses = ast_generics
            .where_clause
            .predicates
            .iter()
            .filter_map(|wp| match *wp {
                hir::WherePredicate::BoundPredicate(ref bp) => Some(bp),
                _ => None,
            })
            .flat_map(|bp| {
                let bt = if is_param(self.tcx, &bp.bounded_ty, param_id) {
                    Some(ty)
                } else if !only_self_bounds.0 {
                    Some(self.to_ty(&bp.bounded_ty))
                } else {
                    None
                };
                bp.bounds.iter().filter_map(move |b| bt.map(|bt| (bt, b)))
            })
            .flat_map(|(bt, b)| predicates_from_bound(self, bt, b));

        from_ty_params.chain(from_where_clauses).collect()
    }
}

/// Tests whether this is the AST for a reference to the type
/// parameter with ID `param_id`. We use this so as to avoid running
/// `ast_ty_to_ty`, because we want to avoid triggering an all-out
/// conversion of the type to avoid inducing unnecessary cycles.
fn is_param(tcx: TyCtxt<'_>, ast_ty: &hir::Ty, param_id: hir::HirId) -> bool {
    if let hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) = ast_ty.node {
        match path.res {
            Res::SelfTy(Some(def_id), None) | Res::Def(DefKind::TyParam, def_id) => {
                def_id == tcx.hir().local_def_id_from_hir_id(param_id)
            }
            _ => false,
        }
    } else {
        false
    }
}

fn convert_item(tcx: TyCtxt<'_>, item_id: hir::HirId) {
    let it = tcx.hir().expect_item(item_id);
    debug!("convert: item {} with id {}", it.ident, it.hir_id);
    let def_id = tcx.hir().local_def_id_from_hir_id(item_id);
    match it.node {
        // These don't define types.
        hir::ItemKind::ExternCrate(_)
        | hir::ItemKind::Use(..)
        | hir::ItemKind::Mod(_)
        | hir::ItemKind::GlobalAsm(_) => {}
        hir::ItemKind::ForeignMod(ref foreign_mod) => {
            for item in &foreign_mod.items {
                let def_id = tcx.hir().local_def_id_from_hir_id(item.hir_id);
                tcx.generics_of(def_id);
                tcx.type_of(def_id);
                tcx.predicates_of(def_id);
                if let hir::ForeignItemKind::Fn(..) = item.node {
                    tcx.fn_sig(def_id);
                }
            }
        }
        hir::ItemKind::Enum(ref enum_definition, _) => {
            tcx.generics_of(def_id);
            tcx.type_of(def_id);
            tcx.predicates_of(def_id);
            convert_enum_variant_types(tcx, def_id, &enum_definition.variants);
        }
        hir::ItemKind::Impl(..) => {
            tcx.generics_of(def_id);
            tcx.type_of(def_id);
            tcx.impl_trait_ref(def_id);
            tcx.predicates_of(def_id);
        }
        hir::ItemKind::Trait(..) => {
            tcx.generics_of(def_id);
            tcx.trait_def(def_id);
            tcx.at(it.span).super_predicates_of(def_id);
            tcx.predicates_of(def_id);
        }
        hir::ItemKind::TraitAlias(..) => {
            tcx.generics_of(def_id);
            tcx.at(it.span).super_predicates_of(def_id);
            tcx.predicates_of(def_id);
        }
        hir::ItemKind::Struct(ref struct_def, _) | hir::ItemKind::Union(ref struct_def, _) => {
            tcx.generics_of(def_id);
            tcx.type_of(def_id);
            tcx.predicates_of(def_id);

            for f in struct_def.fields() {
                let def_id = tcx.hir().local_def_id_from_hir_id(f.hir_id);
                tcx.generics_of(def_id);
                tcx.type_of(def_id);
                tcx.predicates_of(def_id);
            }

            if let Some(ctor_hir_id) = struct_def.ctor_hir_id() {
                convert_variant_ctor(tcx, ctor_hir_id);
            }
        }

        // Desugared from `impl Trait`, so visited by the function's return type.
        hir::ItemKind::Existential(hir::ExistTy {
            impl_trait_fn: Some(_),
            ..
        }) => {}

        hir::ItemKind::Existential(..)
        | hir::ItemKind::Ty(..)
        | hir::ItemKind::Static(..)
        | hir::ItemKind::Const(..)
        | hir::ItemKind::Fn(..) => {
            tcx.generics_of(def_id);
            tcx.type_of(def_id);
            tcx.predicates_of(def_id);
            if let hir::ItemKind::Fn(..) = it.node {
                tcx.fn_sig(def_id);
            }
        }
    }
}

fn convert_trait_item(tcx: TyCtxt<'_>, trait_item_id: hir::HirId) {
    let trait_item = tcx.hir().expect_trait_item(trait_item_id);
    let def_id = tcx.hir().local_def_id_from_hir_id(trait_item.hir_id);
    tcx.generics_of(def_id);

    match trait_item.node {
        hir::TraitItemKind::Const(..)
        | hir::TraitItemKind::Type(_, Some(_))
        | hir::TraitItemKind::Method(..) => {
            tcx.type_of(def_id);
            if let hir::TraitItemKind::Method(..) = trait_item.node {
                tcx.fn_sig(def_id);
            }
        }

        hir::TraitItemKind::Type(_, None) => {}
    };

    tcx.predicates_of(def_id);
}

fn convert_impl_item(tcx: TyCtxt<'_>, impl_item_id: hir::HirId) {
    let def_id = tcx.hir().local_def_id_from_hir_id(impl_item_id);
    tcx.generics_of(def_id);
    tcx.type_of(def_id);
    tcx.predicates_of(def_id);
    if let hir::ImplItemKind::Method(..) = tcx.hir().expect_impl_item(impl_item_id).node {
        tcx.fn_sig(def_id);
    }
}

fn convert_variant_ctor(tcx: TyCtxt<'_>, ctor_id: hir::HirId) {
    let def_id = tcx.hir().local_def_id_from_hir_id(ctor_id);
    tcx.generics_of(def_id);
    tcx.type_of(def_id);
    tcx.predicates_of(def_id);
}

fn convert_enum_variant_types<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, variants: &[hir::Variant]) {
    let def = tcx.adt_def(def_id);
    let repr_type = def.repr.discr_type();
    let initial = repr_type.initial_discriminant(tcx);
    let mut prev_discr = None::<Discr<'tcx>>;

    // fill the discriminant values and field types
    for variant in variants {
        let wrapped_discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
        prev_discr = Some(
            if let Some(ref e) = variant.node.disr_expr {
                let expr_did = tcx.hir().local_def_id_from_hir_id(e.hir_id);
                def.eval_explicit_discr(tcx, expr_did)
            } else if let Some(discr) = repr_type.disr_incr(tcx, prev_discr) {
                Some(discr)
            } else {
                struct_span_err!(
                    tcx.sess,
                    variant.span,
                    E0370,
                    "enum discriminant overflowed"
                ).span_label(
                    variant.span,
                    format!("overflowed on value after {}", prev_discr.unwrap()),
                ).note(&format!(
                    "explicitly set `{} = {}` if that is desired outcome",
                    variant.node.ident, wrapped_discr
                ))
                .emit();
                None
            }.unwrap_or(wrapped_discr),
        );

        for f in variant.node.data.fields() {
            let def_id = tcx.hir().local_def_id_from_hir_id(f.hir_id);
            tcx.generics_of(def_id);
            tcx.type_of(def_id);
            tcx.predicates_of(def_id);
        }

        // Convert the ctor, if any. This also registers the variant as
        // an item.
        if let Some(ctor_hir_id) = variant.node.data.ctor_hir_id() {
            convert_variant_ctor(tcx, ctor_hir_id);
        }
    }
}

fn convert_variant(
    tcx: TyCtxt<'_>,
    variant_did: Option<DefId>,
    ctor_did: Option<DefId>,
    ident: Ident,
    discr: ty::VariantDiscr,
    def: &hir::VariantData,
    adt_kind: ty::AdtKind,
    parent_did: DefId,
) -> ty::VariantDef {
    let mut seen_fields: FxHashMap<ast::Ident, Span> = Default::default();
    let hir_id = tcx.hir().as_local_hir_id(variant_did.unwrap_or(parent_did)).unwrap();
    let fields = def
        .fields()
        .iter()
        .map(|f| {
            let fid = tcx.hir().local_def_id_from_hir_id(f.hir_id);
            let dup_span = seen_fields.get(&f.ident.modern()).cloned();
            if let Some(prev_span) = dup_span {
                struct_span_err!(
                    tcx.sess,
                    f.span,
                    E0124,
                    "field `{}` is already declared",
                    f.ident
                ).span_label(f.span, "field already declared")
                 .span_label(prev_span, format!("`{}` first declared here", f.ident))
                 .emit();
            } else {
                seen_fields.insert(f.ident.modern(), f.span);
            }

            ty::FieldDef {
                did: fid,
                ident: f.ident,
                vis: ty::Visibility::from_hir(&f.vis, hir_id, tcx),
            }
        })
        .collect();
    let recovered = match def {
        hir::VariantData::Struct(_, r) => *r,
        _ => false,
    };
    ty::VariantDef::new(
        tcx,
        ident,
        variant_did,
        ctor_did,
        discr,
        fields,
        CtorKind::from_hir(def),
        adt_kind,
        parent_did,
        recovered,
    )
}

fn adt_def(tcx: TyCtxt<'_>, def_id: DefId) -> &ty::AdtDef {
    use rustc::hir::*;

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let item = match tcx.hir().get(hir_id) {
        Node::Item(item) => item,
        _ => bug!(),
    };

    let repr = ReprOptions::new(tcx, def_id);
    let (kind, variants) = match item.node {
        ItemKind::Enum(ref def, _) => {
            let mut distance_from_explicit = 0;
            let variants = def.variants
                .iter()
                .map(|v| {
                    let variant_did = Some(tcx.hir().local_def_id_from_hir_id(v.node.id));
                    let ctor_did = v.node.data.ctor_hir_id()
                        .map(|hir_id| tcx.hir().local_def_id_from_hir_id(hir_id));

                    let discr = if let Some(ref e) = v.node.disr_expr {
                        distance_from_explicit = 0;
                        ty::VariantDiscr::Explicit(tcx.hir().local_def_id_from_hir_id(e.hir_id))
                    } else {
                        ty::VariantDiscr::Relative(distance_from_explicit)
                    };
                    distance_from_explicit += 1;

                    convert_variant(tcx, variant_did, ctor_did, v.node.ident, discr,
                                    &v.node.data, AdtKind::Enum, def_id)
                })
                .collect();

            (AdtKind::Enum, variants)
        }
        ItemKind::Struct(ref def, _) => {
            let variant_did = None;
            let ctor_did = def.ctor_hir_id()
                .map(|hir_id| tcx.hir().local_def_id_from_hir_id(hir_id));

            let variants = std::iter::once(convert_variant(
                tcx, variant_did, ctor_did, item.ident, ty::VariantDiscr::Relative(0), def,
                AdtKind::Struct, def_id,
            )).collect();

            (AdtKind::Struct, variants)
        }
        ItemKind::Union(ref def, _) => {
            let variant_did = None;
            let ctor_did = def.ctor_hir_id()
                .map(|hir_id| tcx.hir().local_def_id_from_hir_id(hir_id));

            let variants = std::iter::once(convert_variant(
                tcx, variant_did, ctor_did, item.ident, ty::VariantDiscr::Relative(0), def,
                AdtKind::Union, def_id,
            )).collect();

            (AdtKind::Union, variants)
        },
        _ => bug!(),
    };
    tcx.alloc_adt_def(def_id, kind, variants, repr)
}

/// Ensures that the super-predicates of the trait with a `DefId`
/// of `trait_def_id` are converted and stored. This also ensures that
/// the transitive super-predicates are converted.
fn super_predicates_of(
    tcx: TyCtxt<'_>,
    trait_def_id: DefId,
) -> &ty::GenericPredicates<'_> {
    debug!("super_predicates(trait_def_id={:?})", trait_def_id);
    let trait_hir_id = tcx.hir().as_local_hir_id(trait_def_id).unwrap();

    let item = match tcx.hir().get(trait_hir_id) {
        Node::Item(item) => item,
        _ => bug!("trait_node_id {} is not an item", trait_hir_id),
    };

    let (generics, bounds) = match item.node {
        hir::ItemKind::Trait(.., ref generics, ref supertraits, _) => (generics, supertraits),
        hir::ItemKind::TraitAlias(ref generics, ref supertraits) => (generics, supertraits),
        _ => span_bug!(item.span, "super_predicates invoked on non-trait"),
    };

    let icx = ItemCtxt::new(tcx, trait_def_id);

    // Convert the bounds that follow the colon, e.g., `Bar + Zed` in `trait Foo: Bar + Zed`.
    let self_param_ty = tcx.mk_self_type();
    let superbounds1 = AstConv::compute_bounds(&icx, self_param_ty, bounds, SizedByDefault::No,
        item.span);

    let superbounds1 = superbounds1.predicates(tcx, self_param_ty);

    // Convert any explicit superbounds in the where-clause,
    // e.g., `trait Foo where Self: Bar`.
    // In the case of trait aliases, however, we include all bounds in the where-clause,
    // so e.g., `trait Foo = where u32: PartialEq<Self>` would include `u32: PartialEq<Self>`
    // as one of its "superpredicates".
    let is_trait_alias = tcx.is_trait_alias(trait_def_id);
    let superbounds2 = icx.type_parameter_bounds_in_generics(
        generics, item.hir_id, self_param_ty, OnlySelfBounds(!is_trait_alias));

    // Combine the two lists to form the complete set of superbounds:
    let superbounds: Vec<_> = superbounds1.into_iter().chain(superbounds2).collect();

    // Now require that immediate supertraits are converted,
    // which will, in turn, reach indirect supertraits.
    for &(pred, span) in &superbounds {
        debug!("superbound: {:?}", pred);
        if let ty::Predicate::Trait(bound) = pred {
            tcx.at(span).super_predicates_of(bound.def_id());
        }
    }

    tcx.arena.alloc(ty::GenericPredicates {
        parent: None,
        predicates: superbounds,
    })
}

fn trait_def(tcx: TyCtxt<'_>, def_id: DefId) -> &ty::TraitDef {
    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let item = tcx.hir().expect_item(hir_id);

    let (is_auto, unsafety) = match item.node {
        hir::ItemKind::Trait(is_auto, unsafety, ..) => (is_auto == hir::IsAuto::Yes, unsafety),
        hir::ItemKind::TraitAlias(..) => (false, hir::Unsafety::Normal),
        _ => span_bug!(item.span, "trait_def_of_item invoked on non-trait"),
    };

    let paren_sugar = tcx.has_attr(def_id, sym::rustc_paren_sugar);
    if paren_sugar && !tcx.features().unboxed_closures {
        let mut err = tcx.sess.struct_span_err(
            item.span,
            "the `#[rustc_paren_sugar]` attribute is a temporary means of controlling \
             which traits can use parenthetical notation",
        );
        help!(
            &mut err,
            "add `#![feature(unboxed_closures)]` to \
             the crate attributes to use it"
        );
        err.emit();
    }

    let is_marker = tcx.has_attr(def_id, sym::marker);
    let def_path_hash = tcx.def_path_hash(def_id);
    let def = ty::TraitDef::new(def_id, unsafety, paren_sugar, is_auto, is_marker, def_path_hash);
    tcx.arena.alloc(def)
}

fn has_late_bound_regions<'tcx>(tcx: TyCtxt<'tcx>, node: Node<'tcx>) -> Option<Span> {
    struct LateBoundRegionsDetector<'tcx> {
        tcx: TyCtxt<'tcx>,
        outer_index: ty::DebruijnIndex,
        has_late_bound_regions: Option<Span>,
    }

    impl Visitor<'tcx> for LateBoundRegionsDetector<'tcx> {
        fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
            NestedVisitorMap::None
        }

        fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
            if self.has_late_bound_regions.is_some() {
                return;
            }
            match ty.node {
                hir::TyKind::BareFn(..) => {
                    self.outer_index.shift_in(1);
                    intravisit::walk_ty(self, ty);
                    self.outer_index.shift_out(1);
                }
                _ => intravisit::walk_ty(self, ty),
            }
        }

        fn visit_poly_trait_ref(
            &mut self,
            tr: &'tcx hir::PolyTraitRef,
            m: hir::TraitBoundModifier,
        ) {
            if self.has_late_bound_regions.is_some() {
                return;
            }
            self.outer_index.shift_in(1);
            intravisit::walk_poly_trait_ref(self, tr, m);
            self.outer_index.shift_out(1);
        }

        fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) {
            if self.has_late_bound_regions.is_some() {
                return;
            }

            match self.tcx.named_region(lt.hir_id) {
                Some(rl::Region::Static) | Some(rl::Region::EarlyBound(..)) => {}
                Some(rl::Region::LateBound(debruijn, _, _))
                | Some(rl::Region::LateBoundAnon(debruijn, _)) if debruijn < self.outer_index => {}
                Some(rl::Region::LateBound(..))
                | Some(rl::Region::LateBoundAnon(..))
                | Some(rl::Region::Free(..))
                | None => {
                    self.has_late_bound_regions = Some(lt.span);
                }
            }
        }
    }

    fn has_late_bound_regions<'tcx>(
        tcx: TyCtxt<'tcx>,
        generics: &'tcx hir::Generics,
        decl: &'tcx hir::FnDecl,
    ) -> Option<Span> {
        let mut visitor = LateBoundRegionsDetector {
            tcx,
            outer_index: ty::INNERMOST,
            has_late_bound_regions: None,
        };
        for param in &generics.params {
            if let GenericParamKind::Lifetime { .. } = param.kind {
                if tcx.is_late_bound(param.hir_id) {
                    return Some(param.span);
                }
            }
        }
        visitor.visit_fn_decl(decl);
        visitor.has_late_bound_regions
    }

    match node {
        Node::TraitItem(item) => match item.node {
            hir::TraitItemKind::Method(ref sig, _) => {
                has_late_bound_regions(tcx, &item.generics, &sig.decl)
            }
            _ => None,
        },
        Node::ImplItem(item) => match item.node {
            hir::ImplItemKind::Method(ref sig, _) => {
                has_late_bound_regions(tcx, &item.generics, &sig.decl)
            }
            _ => None,
        },
        Node::ForeignItem(item) => match item.node {
            hir::ForeignItemKind::Fn(ref fn_decl, _, ref generics) => {
                has_late_bound_regions(tcx, generics, fn_decl)
            }
            _ => None,
        },
        Node::Item(item) => match item.node {
            hir::ItemKind::Fn(ref fn_decl, .., ref generics, _) => {
                has_late_bound_regions(tcx, generics, fn_decl)
            }
            _ => None,
        },
        _ => None,
    }
}

fn generics_of(tcx: TyCtxt<'_>, def_id: DefId) -> &ty::Generics {
    use rustc::hir::*;

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

    let node = tcx.hir().get(hir_id);
    let parent_def_id = match node {
        Node::ImplItem(_) | Node::TraitItem(_) | Node::Variant(_) |
        Node::Ctor(..) | Node::Field(_) => {
            let parent_id = tcx.hir().get_parent_item(hir_id);
            Some(tcx.hir().local_def_id_from_hir_id(parent_id))
        }
        Node::Expr(&hir::Expr {
            node: hir::ExprKind::Closure(..),
            ..
        }) => Some(tcx.closure_base_def_id(def_id)),
        Node::Item(item) => match item.node {
            ItemKind::Existential(hir::ExistTy { impl_trait_fn, .. }) => impl_trait_fn,
            _ => None,
        },
        _ => None,
    };

    let mut opt_self = None;
    let mut allow_defaults = false;

    let no_generics = hir::Generics::empty();
    let ast_generics = match node {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => &item.generics,

        Node::Item(item) => {
            match item.node {
                ItemKind::Fn(.., ref generics, _) | ItemKind::Impl(_, _, _, ref generics, ..) => {
                    generics
                }

                ItemKind::Ty(_, ref generics)
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Existential(hir::ExistTy { ref generics, .. })
                | ItemKind::Union(_, ref generics) => {
                    allow_defaults = true;
                    generics
                }

                ItemKind::Trait(_, _, ref generics, ..)
                | ItemKind::TraitAlias(ref generics, ..) => {
                    // Add in the self type parameter.
                    //
                    // Something of a hack: use the node id for the trait, also as
                    // the node id for the Self type parameter.
                    let param_id = item.hir_id;

                    opt_self = Some(ty::GenericParamDef {
                        index: 0,
                        name: kw::SelfUpper.as_interned_str(),
                        def_id: tcx.hir().local_def_id_from_hir_id(param_id),
                        pure_wrt_drop: false,
                        kind: ty::GenericParamDefKind::Type {
                            has_default: false,
                            object_lifetime_default: rl::Set1::Empty,
                            synthetic: None,
                        },
                    });

                    allow_defaults = true;
                    generics
                }

                _ => &no_generics,
            }
        }

        Node::ForeignItem(item) => match item.node {
            ForeignItemKind::Static(..) => &no_generics,
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            ForeignItemKind::Type => &no_generics,
        },

        _ => &no_generics,
    };

    let has_self = opt_self.is_some();
    let mut parent_has_self = false;
    let mut own_start = has_self as u32;
    let parent_count = parent_def_id.map_or(0, |def_id| {
        let generics = tcx.generics_of(def_id);
        assert_eq!(has_self, false);
        parent_has_self = generics.has_self;
        own_start = generics.count() as u32;
        generics.parent_count + generics.params.len()
    });

    let mut params: Vec<_> = opt_self.into_iter().collect();

    let early_lifetimes = early_bound_lifetimes_from_generics(tcx, ast_generics);
    params.extend(
        early_lifetimes
            .enumerate()
            .map(|(i, param)| ty::GenericParamDef {
                name: param.name.ident().as_interned_str(),
                index: own_start + i as u32,
                def_id: tcx.hir().local_def_id_from_hir_id(param.hir_id),
                pure_wrt_drop: param.pure_wrt_drop,
                kind: ty::GenericParamDefKind::Lifetime,
            }),
    );

    let object_lifetime_defaults = tcx.object_lifetime_defaults(hir_id);

    // Now create the real type parameters.
    let type_start = own_start - has_self as u32 + params.len() as u32;
    let mut i = 0;
    params.extend(
        ast_generics
            .params
            .iter()
            .filter_map(|param| {
                let kind = match param.kind {
                    GenericParamKind::Type {
                        ref default,
                        synthetic,
                        ..
                    } => {
                        if param.name.ident().name == kw::SelfUpper {
                            span_bug!(
                                param.span,
                                "`Self` should not be the name of a regular parameter"
                            );
                        }

                        if !allow_defaults && default.is_some() {
                            if !tcx.features().default_type_parameter_fallback {
                                tcx.lint_hir(
                                    lint::builtin::INVALID_TYPE_PARAM_DEFAULT,
                                    param.hir_id,
                                    param.span,
                                    &format!(
                                        "defaults for type parameters are only allowed in \
                                        `struct`, `enum`, `type`, or `trait` definitions."
                                    ),
                                );
                            }
                        }

                        ty::GenericParamDefKind::Type {
                            has_default: default.is_some(),
                            object_lifetime_default: object_lifetime_defaults
                                .as_ref()
                                .map_or(rl::Set1::Empty, |o| o[i]),
                            synthetic,
                        }
                    }
                    GenericParamKind::Const { .. } => {
                        if param.name.ident().name == kw::SelfUpper {
                            span_bug!(
                                param.span,
                                "`Self` should not be the name of a regular parameter",
                            );
                        }

                        ty::GenericParamDefKind::Const
                    }
                    _ => return None,
                };

                let param_def = ty::GenericParamDef {
                    index: type_start + i as u32,
                    name: param.name.ident().as_interned_str(),
                    def_id: tcx.hir().local_def_id_from_hir_id(param.hir_id),
                    pure_wrt_drop: param.pure_wrt_drop,
                    kind,
                };
                i += 1;
                Some(param_def)
            })
    );

    // provide junk type parameter defs - the only place that
    // cares about anything but the length is instantiation,
    // and we don't do that for closures.
    if let Node::Expr(&hir::Expr {
        node: hir::ExprKind::Closure(.., gen),
        ..
    }) = node
    {
        let dummy_args = if gen.is_some() {
            &["<yield_ty>", "<return_ty>", "<witness>"][..]
        } else {
            &["<closure_kind>", "<closure_signature>"][..]
        };

        params.extend(
            dummy_args
                .iter()
                .enumerate()
                .map(|(i, &arg)| ty::GenericParamDef {
                    index: type_start + i as u32,
                    name: InternedString::intern(arg),
                    def_id,
                    pure_wrt_drop: false,
                    kind: ty::GenericParamDefKind::Type {
                        has_default: false,
                        object_lifetime_default: rl::Set1::Empty,
                        synthetic: None,
                    },
                }),
        );

        if let Some(upvars) = tcx.upvars(def_id) {
            params.extend(upvars.iter().zip((dummy_args.len() as u32)..).map(|(_, i)| {
                ty::GenericParamDef {
                    index: type_start + i,
                    name: InternedString::intern("<upvar>"),
                    def_id,
                    pure_wrt_drop: false,
                    kind: ty::GenericParamDefKind::Type {
                        has_default: false,
                        object_lifetime_default: rl::Set1::Empty,
                        synthetic: None,
                    },
                }
            }));
        }
    }

    let param_def_id_to_index = params
        .iter()
        .map(|param| (param.def_id, param.index))
        .collect();

    tcx.arena.alloc(ty::Generics {
        parent: parent_def_id,
        parent_count,
        params,
        param_def_id_to_index,
        has_self: has_self || parent_has_self,
        has_late_bound_regions: has_late_bound_regions(tcx, node),
    })
}

fn report_assoc_ty_on_inherent_impl(tcx: TyCtxt<'_>, span: Span) {
    span_err!(
        tcx.sess,
        span,
        E0202,
        "associated types are not yet supported in inherent impls (see #8995)"
    );
}

fn type_of(tcx: TyCtxt<'_>, def_id: DefId) -> Ty<'_> {
    checked_type_of(tcx, def_id, true).unwrap()
}

/// Same as [`type_of`] but returns [`Option`] instead of failing.
///
/// If you want to fail anyway, you can set the `fail` parameter to true, but in this case,
/// you'd better just call [`type_of`] directly.
pub fn checked_type_of(tcx: TyCtxt<'_>, def_id: DefId, fail: bool) -> Option<Ty<'_>> {
    use rustc::hir::*;

    let hir_id = match tcx.hir().as_local_hir_id(def_id) {
        Some(hir_id) => hir_id,
        None => {
            if !fail {
                return None;
            }
            bug!("invalid node");
        }
    };

    let icx = ItemCtxt::new(tcx, def_id);

    Some(match tcx.hir().get(hir_id) {
        Node::TraitItem(item) => match item.node {
            TraitItemKind::Method(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
            TraitItemKind::Const(ref ty, _) | TraitItemKind::Type(_, Some(ref ty)) => icx.to_ty(ty),
            TraitItemKind::Type(_, None) => {
                if !fail {
                    return None;
                }
                span_bug!(item.span, "associated type missing default");
            }
        },

        Node::ImplItem(item) => match item.node {
            ImplItemKind::Method(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
            ImplItemKind::Const(ref ty, _) => icx.to_ty(ty),
            ImplItemKind::Existential(_) => {
                if tcx
                    .impl_trait_ref(tcx.hir().get_parent_did(hir_id))
                    .is_none()
                {
                    report_assoc_ty_on_inherent_impl(tcx, item.span);
                }

                find_existential_constraints(tcx, def_id)
            }
            ImplItemKind::Type(ref ty) => {
                if tcx
                    .impl_trait_ref(tcx.hir().get_parent_did(hir_id))
                    .is_none()
                {
                    report_assoc_ty_on_inherent_impl(tcx, item.span);
                }

                icx.to_ty(ty)
            }
        },

        Node::Item(item) => {
            match item.node {
                ItemKind::Static(ref t, ..)
                | ItemKind::Const(ref t, _)
                | ItemKind::Ty(ref t, _)
                | ItemKind::Impl(.., ref t, _) => icx.to_ty(t),
                ItemKind::Fn(..) => {
                    let substs = InternalSubsts::identity_for_item(tcx, def_id);
                    tcx.mk_fn_def(def_id, substs)
                }
                ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
                    let def = tcx.adt_def(def_id);
                    let substs = InternalSubsts::identity_for_item(tcx, def_id);
                    tcx.mk_adt(def, substs)
                }
                ItemKind::Existential(hir::ExistTy {
                    impl_trait_fn: None,
                    ..
                }) => find_existential_constraints(tcx, def_id),
                // Existential types desugared from `impl Trait`.
                ItemKind::Existential(hir::ExistTy {
                    impl_trait_fn: Some(owner),
                    ..
                }) => {
                    tcx.typeck_tables_of(owner)
                        .concrete_existential_types
                        .get(&def_id)
                        .map(|opaque| opaque.concrete_type)
                        .unwrap_or_else(|| {
                            // This can occur if some error in the
                            // owner fn prevented us from populating
                            // the `concrete_existential_types` table.
                            tcx.sess.delay_span_bug(
                                DUMMY_SP,
                                &format!(
                                    "owner {:?} has no existential type for {:?} in its tables",
                                    owner, def_id,
                                ),
                            );
                            tcx.types.err
                        })
                }
                ItemKind::Trait(..)
                | ItemKind::TraitAlias(..)
                | ItemKind::Mod(..)
                | ItemKind::ForeignMod(..)
                | ItemKind::GlobalAsm(..)
                | ItemKind::ExternCrate(..)
                | ItemKind::Use(..) => {
                    if !fail {
                        return None;
                    }
                    span_bug!(
                        item.span,
                        "compute_type_of_item: unexpected item type: {:?}",
                        item.node
                    );
                }
            }
        }

        Node::ForeignItem(foreign_item) => match foreign_item.node {
            ForeignItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
            ForeignItemKind::Static(ref t, _) => icx.to_ty(t),
            ForeignItemKind::Type => tcx.mk_foreign(def_id),
        },

        Node::Ctor(&ref def) | Node::Variant(&Spanned {
            node: hir::VariantKind { data: ref def, .. },
            ..
        }) => match *def {
            VariantData::Unit(..) | VariantData::Struct(..) => {
                tcx.type_of(tcx.hir().get_parent_did(hir_id))
            }
            VariantData::Tuple(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
        },

        Node::Field(field) => icx.to_ty(&field.ty),

        Node::Expr(&hir::Expr {
            node: hir::ExprKind::Closure(.., gen),
            ..
        }) => {
            if gen.is_some() {
                return Some(tcx.typeck_tables_of(def_id).node_type(hir_id));
            }

            let substs = ty::ClosureSubsts {
                substs: InternalSubsts::identity_for_item(tcx, def_id),
            };

            tcx.mk_closure(def_id, substs)
        }

        Node::AnonConst(_) => {
            let parent_node = tcx.hir().get(tcx.hir().get_parent_node(hir_id));
            match parent_node {
                Node::Ty(&hir::Ty {
                    node: hir::TyKind::Array(_, ref constant),
                    ..
                })
                | Node::Ty(&hir::Ty {
                    node: hir::TyKind::Typeof(ref constant),
                    ..
                })
                | Node::Expr(&hir::Expr {
                    node: ExprKind::Repeat(_, ref constant),
                    ..
                }) if constant.hir_id == hir_id =>
                {
                    tcx.types.usize
                }

                Node::Variant(&Spanned {
                    node:
                        VariantKind {
                            disr_expr: Some(ref e),
                            ..
                        },
                    ..
                }) if e.hir_id == hir_id =>
                {
                    tcx.adt_def(tcx.hir().get_parent_did(hir_id))
                        .repr
                        .discr_type()
                        .to_ty(tcx)
                }

                Node::Ty(&hir::Ty { node: hir::TyKind::Path(_), .. }) |
                Node::Expr(&hir::Expr { node: ExprKind::Struct(..), .. }) |
                Node::Expr(&hir::Expr { node: ExprKind::Path(_), .. }) |
                Node::TraitRef(..) => {
                    let path = match parent_node {
                        Node::Ty(&hir::Ty {
                            node: hir::TyKind::Path(QPath::Resolved(_, ref path)),
                            ..
                        })
                        | Node::Expr(&hir::Expr {
                            node: ExprKind::Path(QPath::Resolved(_, ref path)),
                            ..
                        }) => {
                            Some(&**path)
                        }
                        Node::Expr(&hir::Expr { node: ExprKind::Struct(ref path, ..), .. }) => {
                            if let QPath::Resolved(_, ref path) = **path {
                                Some(&**path)
                            } else {
                                None
                            }
                        }
                        Node::TraitRef(&hir::TraitRef { ref path, .. }) => Some(&**path),
                        _ => None,
                    };

                    if let Some(path) = path {
                        let arg_index = path.segments.iter()
                            .filter_map(|seg| seg.args.as_ref())
                            .map(|generic_args| generic_args.args.as_ref())
                            .find_map(|args| {
                                args.iter()
                                    .filter(|arg| arg.is_const())
                                    .enumerate()
                                    .filter(|(_, arg)| arg.id() == hir_id)
                                    .map(|(index, _)| index)
                                    .next()
                            })
                            .or_else(|| {
                                if !fail {
                                    None
                                } else {
                                    bug!("no arg matching AnonConst in path")
                                }
                            })?;

                        // We've encountered an `AnonConst` in some path, so we need to
                        // figure out which generic parameter it corresponds to and return
                        // the relevant type.
                        let generics = match path.res {
                            Res::Def(DefKind::Ctor(..), def_id) => {
                                tcx.generics_of(tcx.parent(def_id).unwrap())
                            }
                            Res::Def(_, def_id) => tcx.generics_of(def_id),
                            Res::Err => return Some(tcx.types.err),
                            _ if !fail => return None,
                            res => {
                                tcx.sess.delay_span_bug(
                                    DUMMY_SP,
                                    &format!(
                                        "unexpected const parent path def {:?}",
                                        res,
                                    ),
                                );
                                return Some(tcx.types.err);
                            }
                        };

                        generics.params.iter()
                            .filter(|param| {
                                if let ty::GenericParamDefKind::Const = param.kind {
                                    true
                                } else {
                                    false
                                }
                            })
                            .nth(arg_index)
                            .map(|param| tcx.type_of(param.def_id))
                            // This is no generic parameter associated with the arg. This is
                            // probably from an extra arg where one is not needed.
                            .unwrap_or(tcx.types.err)
                    } else {
                        if !fail {
                            return None;
                        }
                        tcx.sess.delay_span_bug(
                            DUMMY_SP,
                            &format!(
                                "unexpected const parent path {:?}",
                                parent_node,
                            ),
                        );
                        return Some(tcx.types.err);
                    }
                }

                x => {
                    if !fail {
                        return None;
                    }
                    tcx.sess.delay_span_bug(
                        DUMMY_SP,
                        &format!(
                            "unexpected const parent in type_of_def_id(): {:?}", x
                        ),
                    );
                    tcx.types.err
                }
            }
        }

        Node::GenericParam(param) => match &param.kind {
            hir::GenericParamKind::Type { default: Some(ref ty), .. } |
            hir::GenericParamKind::Const { ref ty, .. } => {
                icx.to_ty(ty)
            }
            x => {
                if !fail {
                    return None;
                }
                bug!("unexpected non-type Node::GenericParam: {:?}", x)
            },
        },

        x => {
            if !fail {
                return None;
            }
            bug!("unexpected sort of node in type_of_def_id(): {:?}", x);
        }
    })
}

fn find_existential_constraints(tcx: TyCtxt<'_>, def_id: DefId) -> Ty<'_> {
    use rustc::hir::{ImplItem, Item, TraitItem};

    debug!("find_existential_constraints({:?})", def_id);

    struct ConstraintLocator<'tcx> {
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        // (first found type span, actual type, mapping from the existential type's generic
        // parameters to the concrete type's generic parameters)
        //
        // The mapping is an index for each use site of a generic parameter in the concrete type
        //
        // The indices index into the generic parameters on the existential type.
        found: Option<(Span, Ty<'tcx>, Vec<usize>)>,
    }

    impl ConstraintLocator<'tcx> {
        fn check(&mut self, def_id: DefId) {
            // Don't try to check items that cannot possibly constrain the type.
            if !self.tcx.has_typeck_tables(def_id) {
                debug!(
                    "find_existential_constraints: no constraint for `{:?}` at `{:?}`: no tables",
                    self.def_id,
                    def_id,
                );
                return;
            }
            let ty = self
                .tcx
                .typeck_tables_of(def_id)
                .concrete_existential_types
                .get(&self.def_id);
            if let Some(ty::ResolvedOpaqueTy { concrete_type, substs }) = ty {
                debug!(
                    "find_existential_constraints: found constraint for `{:?}` at `{:?}`: {:?}",
                    self.def_id,
                    def_id,
                    ty,
                );

                // FIXME(oli-obk): trace the actual span from inference to improve errors.
                let span = self.tcx.def_span(def_id);
                // used to quickly look up the position of a generic parameter
                let mut index_map: FxHashMap<ty::ParamTy, usize> = FxHashMap::default();
                // Skipping binder is ok, since we only use this to find generic parameters and
                // their positions.
                for (idx, subst) in substs.iter().enumerate() {
                    if let UnpackedKind::Type(ty) = subst.unpack() {
                        if let ty::Param(p) = ty.sty {
                            if index_map.insert(p, idx).is_some() {
                                // There was already an entry for `p`, meaning a generic parameter
                                // was used twice.
                                self.tcx.sess.span_err(
                                    span,
                                    &format!(
                                        "defining existential type use restricts existential \
                                         type by using the generic parameter `{}` twice",
                                        p.name
                                    ),
                                );
                                return;
                            }
                        } else {
                            self.tcx.sess.delay_span_bug(
                                span,
                                &format!(
                                    "non-defining exist ty use in defining scope: {:?}, {:?}",
                                    concrete_type, substs,
                                ),
                            );
                        }
                    }
                }
                // Compute the index within the existential type for each generic parameter used in
                // the concrete type.
                let indices = concrete_type
                    .subst(self.tcx, substs)
                    .walk()
                    .filter_map(|t| match &t.sty {
                        ty::Param(p) => Some(*index_map.get(p).unwrap()),
                        _ => None,
                    }).collect();
                let is_param = |ty: Ty<'_>| match ty.sty {
                    ty::Param(_) => true,
                    _ => false,
                };
                if !substs.types().all(is_param) {
                    self.tcx.sess.span_err(
                        span,
                        "defining existential type use does not fully define existential type",
                    );
                } else if let Some((prev_span, prev_ty, ref prev_indices)) = self.found {
                    let mut ty = concrete_type.walk().fuse();
                    let mut p_ty = prev_ty.walk().fuse();
                    let iter_eq = (&mut ty).zip(&mut p_ty).all(|(t, p)| match (&t.sty, &p.sty) {
                        // Type parameters are equal to any other type parameter for the purpose of
                        // concrete type equality, as it is possible to obtain the same type just
                        // by passing matching parameters to a function.
                        (ty::Param(_), ty::Param(_)) => true,
                        _ => t == p,
                    });
                    if !iter_eq || ty.next().is_some() || p_ty.next().is_some() {
                        debug!("find_existential_constraints: span={:?}", span);
                        // Found different concrete types for the existential type.
                        let mut err = self.tcx.sess.struct_span_err(
                            span,
                            "concrete type differs from previous defining existential type use",
                        );
                        err.span_label(
                            span,
                            format!("expected `{}`, got `{}`", prev_ty, concrete_type),
                        );
                        err.span_note(prev_span, "previous use here");
                        err.emit();
                    } else if indices != *prev_indices {
                        // Found "same" concrete types, but the generic parameter order differs.
                        let mut err = self.tcx.sess.struct_span_err(
                            span,
                            "concrete type's generic parameters differ from previous defining use",
                        );
                        use std::fmt::Write;
                        let mut s = String::new();
                        write!(s, "expected [").unwrap();
                        let list = |s: &mut String, indices: &Vec<usize>| {
                            let mut indices = indices.iter().cloned();
                            if let Some(first) = indices.next() {
                                write!(s, "`{}`", substs[first]).unwrap();
                                for i in indices {
                                    write!(s, ", `{}`", substs[i]).unwrap();
                                }
                            }
                        };
                        list(&mut s, prev_indices);
                        write!(s, "], got [").unwrap();
                        list(&mut s, &indices);
                        write!(s, "]").unwrap();
                        err.span_label(span, s);
                        err.span_note(prev_span, "previous use here");
                        err.emit();
                    }
                } else {
                    self.found = Some((span, concrete_type, indices));
                }
            } else {
                debug!(
                    "find_existential_constraints: no constraint for `{:?}` at `{:?}`",
                    self.def_id,
                    def_id,
                );
            }
        }
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstraintLocator<'tcx> {
        fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
            intravisit::NestedVisitorMap::All(&self.tcx.hir())
        }
        fn visit_item(&mut self, it: &'tcx Item) {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(it.hir_id);
            // The existential type itself or its children are not within its reveal scope.
            if def_id != self.def_id {
                self.check(def_id);
                intravisit::walk_item(self, it);
            }
        }
        fn visit_impl_item(&mut self, it: &'tcx ImplItem) {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(it.hir_id);
            // The existential type itself or its children are not within its reveal scope.
            if def_id != self.def_id {
                self.check(def_id);
                intravisit::walk_impl_item(self, it);
            }
        }
        fn visit_trait_item(&mut self, it: &'tcx TraitItem) {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(it.hir_id);
            self.check(def_id);
            intravisit::walk_trait_item(self, it);
        }
    }

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let scope = tcx.hir()
        .get_defining_scope(hir_id)
        .expect("could not get defining scope");
    let mut locator = ConstraintLocator {
        def_id,
        tcx,
        found: None,
    };

    debug!("find_existential_constraints: scope={:?}", scope);

    if scope == hir::CRATE_HIR_ID {
        intravisit::walk_crate(&mut locator, tcx.hir().krate());
    } else {
        debug!("find_existential_constraints: scope={:?}", tcx.hir().get(scope));
        match tcx.hir().get(scope) {
            Node::Item(ref it) => intravisit::walk_item(&mut locator, it),
            Node::ImplItem(ref it) => intravisit::walk_impl_item(&mut locator, it),
            Node::TraitItem(ref it) => intravisit::walk_trait_item(&mut locator, it),
            other => bug!(
                "{:?} is not a valid scope for an existential type item",
                other
            ),
        }
    }

    match locator.found {
        Some((_, ty, _)) => ty,
        None => {
            let span = tcx.def_span(def_id);
            tcx.sess.span_err(span, "could not find defining uses");
            tcx.types.err
        }
    }
}

fn fn_sig(tcx: TyCtxt<'_>, def_id: DefId) -> ty::PolyFnSig<'_> {
    use rustc::hir::*;
    use rustc::hir::Node::*;

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

    let icx = ItemCtxt::new(tcx, def_id);

    match tcx.hir().get(hir_id) {
        TraitItem(hir::TraitItem {
            node: TraitItemKind::Method(sig, _),
            ..
        })
        | ImplItem(hir::ImplItem {
            node: ImplItemKind::Method(sig, _),
            ..
        }) => AstConv::ty_of_fn(&icx, sig.header.unsafety, sig.header.abi, &sig.decl),

        Item(hir::Item {
            node: ItemKind::Fn(decl, header, _, _),
            ..
        }) => AstConv::ty_of_fn(&icx, header.unsafety, header.abi, decl),

        ForeignItem(&hir::ForeignItem {
            node: ForeignItemKind::Fn(ref fn_decl, _, _),
            ..
        }) => {
            let abi = tcx.hir().get_foreign_abi(hir_id);
            compute_sig_of_foreign_fn_decl(tcx, def_id, fn_decl, abi)
        }

        Ctor(data) | Variant(Spanned {
            node: hir::VariantKind { data, ..  },
            ..
        }) if data.ctor_hir_id().is_some() => {
            let ty = tcx.type_of(tcx.hir().get_parent_did(hir_id));
            let inputs = data.fields()
                .iter()
                .map(|f| tcx.type_of(tcx.hir().local_def_id_from_hir_id(f.hir_id)));
            ty::Binder::bind(tcx.mk_fn_sig(
                inputs,
                ty,
                false,
                hir::Unsafety::Normal,
                abi::Abi::Rust,
            ))
        }

        Expr(&hir::Expr {
            node: hir::ExprKind::Closure(..),
            ..
        }) => {
            // Closure signatures are not like other function
            // signatures and cannot be accessed through `fn_sig`. For
            // example, a closure signature excludes the `self`
            // argument. In any case they are embedded within the
            // closure type as part of the `ClosureSubsts`.
            //
            // To get
            // the signature of a closure, you should use the
            // `closure_sig` method on the `ClosureSubsts`:
            //
            //    closure_substs.closure_sig(def_id, tcx)
            //
            // or, inside of an inference context, you can use
            //
            //    infcx.closure_sig(def_id, closure_substs)
            bug!("to get the signature of a closure, use `closure_sig()` not `fn_sig()`");
        }

        x => {
            bug!("unexpected sort of node in fn_sig(): {:?}", x);
        }
    }
}

fn impl_trait_ref(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ty::TraitRef<'_>> {
    let icx = ItemCtxt::new(tcx, def_id);

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    match tcx.hir().expect_item(hir_id).node {
        hir::ItemKind::Impl(.., ref opt_trait_ref, _, _) => {
            opt_trait_ref.as_ref().map(|ast_trait_ref| {
                let selfty = tcx.type_of(def_id);
                AstConv::instantiate_mono_trait_ref(&icx, ast_trait_ref, selfty)
            })
        }
        _ => bug!(),
    }
}

fn impl_polarity(tcx: TyCtxt<'_>, def_id: DefId) -> hir::ImplPolarity {
    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    match tcx.hir().expect_item(hir_id).node {
        hir::ItemKind::Impl(_, polarity, ..) => polarity,
        ref item => bug!("impl_polarity: {:?} not an impl", item),
    }
}

/// Returns the early-bound lifetimes declared in this generics
/// listing. For anything other than fns/methods, this is just all
/// the lifetimes that are declared. For fns or methods, we have to
/// screen out those that do not appear in any where-clauses etc using
/// `resolve_lifetime::early_bound_lifetimes`.
fn early_bound_lifetimes_from_generics<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx>,
    generics: &'a hir::Generics,
) -> impl Iterator<Item = &'a hir::GenericParam> + Captures<'tcx> {
    generics
        .params
        .iter()
        .filter(move |param| match param.kind {
            GenericParamKind::Lifetime { .. } => {
                !tcx.is_late_bound(param.hir_id)
            }
            _ => false,
        })
}

/// Returns a list of type predicates for the definition with ID `def_id`, including inferred
/// lifetime constraints. This includes all predicates returned by `explicit_predicates_of`, plus
/// inferred constraints concerning which regions outlive other regions.
fn predicates_defined_on(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> &ty::GenericPredicates<'_> {
    debug!("predicates_defined_on({:?})", def_id);
    let mut result = tcx.explicit_predicates_of(def_id);
    debug!(
        "predicates_defined_on: explicit_predicates_of({:?}) = {:?}",
        def_id,
        result,
    );
    let inferred_outlives = tcx.inferred_outlives_of(def_id);
    if !inferred_outlives.is_empty() {
        let span = tcx.def_span(def_id);
        debug!(
            "predicates_defined_on: inferred_outlives_of({:?}) = {:?}",
            def_id,
            inferred_outlives,
        );
        let mut predicates = (*result).clone();
        predicates.predicates.extend(inferred_outlives.iter().map(|&p| (p, span)));
        result = tcx.arena.alloc(predicates);
    }
    debug!("predicates_defined_on({:?}) = {:?}", def_id, result);
    result
}

/// Returns a list of all type predicates (explicit and implicit) for the definition with
/// ID `def_id`. This includes all predicates returned by `predicates_defined_on`, plus
/// `Self: Trait` predicates for traits.
fn predicates_of(tcx: TyCtxt<'_>, def_id: DefId) -> &ty::GenericPredicates<'_> {
    let mut result = tcx.predicates_defined_on(def_id);

    if tcx.is_trait(def_id) {
        // For traits, add `Self: Trait` predicate. This is
        // not part of the predicates that a user writes, but it
        // is something that one must prove in order to invoke a
        // method or project an associated type.
        //
        // In the chalk setup, this predicate is not part of the
        // "predicates" for a trait item. But it is useful in
        // rustc because if you directly (e.g.) invoke a trait
        // method like `Trait::method(...)`, you must naturally
        // prove that the trait applies to the types that were
        // used, and adding the predicate into this list ensures
        // that this is done.
        let span = tcx.def_span(def_id);
        let mut predicates = (*result).clone();
        predicates.predicates.push((ty::TraitRef::identity(tcx, def_id).to_predicate(), span));
        result = tcx.arena.alloc(predicates);
    }
    debug!("predicates_of(def_id={:?}) = {:?}", def_id, result);
    result
}

/// Returns a list of user-specified type predicates for the definition with ID `def_id`.
/// N.B., this does not include any implied/inferred constraints.
fn explicit_predicates_of(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> &ty::GenericPredicates<'_> {
    use rustc::hir::*;
    use rustc_data_structures::fx::FxHashSet;

    debug!("explicit_predicates_of(def_id={:?})", def_id);

    /// A data structure with unique elements, which preserves order of insertion.
    /// Preserving the order of insertion is important here so as not to break
    /// compile-fail UI tests.
    struct UniquePredicates<'tcx> {
        predicates: Vec<(ty::Predicate<'tcx>, Span)>,
        uniques: FxHashSet<(ty::Predicate<'tcx>, Span)>,
    }

    impl<'tcx> UniquePredicates<'tcx> {
        fn new() -> Self {
            UniquePredicates {
                predicates: vec![],
                uniques: FxHashSet::default(),
            }
        }

        fn push(&mut self, value: (ty::Predicate<'tcx>, Span)) {
            if self.uniques.insert(value) {
                self.predicates.push(value);
            }
        }

        fn extend<I: IntoIterator<Item = (ty::Predicate<'tcx>, Span)>>(&mut self, iter: I) {
            for value in iter {
                self.push(value);
            }
        }
    }

    let hir_id = match tcx.hir().as_local_hir_id(def_id) {
        Some(hir_id) => hir_id,
        None => return tcx.predicates_of(def_id),
    };
    let node = tcx.hir().get(hir_id);

    let mut is_trait = None;
    let mut is_default_impl_trait = None;

    let icx = ItemCtxt::new(tcx, def_id);

    const NO_GENERICS: &hir::Generics = &hir::Generics::empty();

    let empty_trait_items = HirVec::new();

    let mut predicates = UniquePredicates::new();

    let ast_generics = match node {
        Node::TraitItem(item) => &item.generics,

        Node::ImplItem(item) => match item.node {
            ImplItemKind::Existential(ref bounds) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                let opaque_ty = tcx.mk_opaque(def_id, substs);

                // Collect the bounds, i.e., the `A + B + 'c` in `impl A + B + 'c`.
                let bounds = AstConv::compute_bounds(
                    &icx,
                    opaque_ty,
                    bounds,
                    SizedByDefault::Yes,
                    tcx.def_span(def_id),
                );

                predicates.extend(bounds.predicates(tcx, opaque_ty));
                &item.generics
            }
            _ => &item.generics,
        },

        Node::Item(item) => {
            match item.node {
                ItemKind::Impl(_, _, defaultness, ref generics, ..) => {
                    if defaultness.is_default() {
                        is_default_impl_trait = tcx.impl_trait_ref(def_id);
                    }
                    generics
                }
                ItemKind::Fn(.., ref generics, _)
                | ItemKind::Ty(_, ref generics)
                | ItemKind::Enum(_, ref generics)
                | ItemKind::Struct(_, ref generics)
                | ItemKind::Union(_, ref generics) => generics,

                ItemKind::Trait(_, _, ref generics, .., ref items) => {
                    is_trait = Some((ty::TraitRef::identity(tcx, def_id), items));
                    generics
                }
                ItemKind::TraitAlias(ref generics, _) => {
                    is_trait = Some((ty::TraitRef::identity(tcx, def_id), &empty_trait_items));
                    generics
                }
                ItemKind::Existential(ExistTy {
                    ref bounds,
                    impl_trait_fn,
                    ref generics,
                    origin: _,
                }) => {
                    let substs = InternalSubsts::identity_for_item(tcx, def_id);
                    let opaque_ty = tcx.mk_opaque(def_id, substs);

                    // Collect the bounds, i.e., the `A + B + 'c` in `impl A + B + 'c`.
                    let bounds = AstConv::compute_bounds(
                        &icx,
                        opaque_ty,
                        bounds,
                        SizedByDefault::Yes,
                        tcx.def_span(def_id),
                    );

                    let bounds_predicates = bounds.predicates(tcx, opaque_ty);
                    if impl_trait_fn.is_some() {
                        // opaque types
                        return tcx.arena.alloc(ty::GenericPredicates {
                            parent: None,
                            predicates: bounds_predicates,
                        });
                    } else {
                        // named existential types
                        predicates.extend(bounds_predicates);
                        generics
                    }
                }

                _ => NO_GENERICS,
            }
        }

        Node::ForeignItem(item) => match item.node {
            ForeignItemKind::Static(..) => NO_GENERICS,
            ForeignItemKind::Fn(_, _, ref generics) => generics,
            ForeignItemKind::Type => NO_GENERICS,
        },

        _ => NO_GENERICS,
    };

    let generics = tcx.generics_of(def_id);
    let parent_count = generics.parent_count as u32;
    let has_own_self = generics.has_self && parent_count == 0;

    // Below we'll consider the bounds on the type parameters (including `Self`)
    // and the explicit where-clauses, but to get the full set of predicates
    // on a trait we need to add in the supertrait bounds and bounds found on
    // associated types.
    if let Some((_trait_ref, _)) = is_trait {
        predicates.extend(tcx.super_predicates_of(def_id).predicates.iter().cloned());
    }

    // In default impls, we can assume that the self type implements
    // the trait. So in:
    //
    //     default impl Foo for Bar { .. }
    //
    // we add a default where clause `Foo: Bar`. We do a similar thing for traits
    // (see below). Recall that a default impl is not itself an impl, but rather a
    // set of defaults that can be incorporated into another impl.
    if let Some(trait_ref) = is_default_impl_trait {
        predicates.push((trait_ref.to_poly_trait_ref().to_predicate(), tcx.def_span(def_id)));
    }

    // Collect the region predicates that were declared inline as
    // well. In the case of parameters declared on a fn or method, we
    // have to be careful to only iterate over early-bound regions.
    let mut index = parent_count + has_own_self as u32;
    for param in early_bound_lifetimes_from_generics(tcx, ast_generics) {
        let region = tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
            def_id: tcx.hir().local_def_id_from_hir_id(param.hir_id),
            index,
            name: param.name.ident().as_interned_str(),
        }));
        index += 1;

        match param.kind {
            GenericParamKind::Lifetime { .. } => {
                param.bounds.iter().for_each(|bound| match bound {
                    hir::GenericBound::Outlives(lt) => {
                        let bound = AstConv::ast_region_to_region(&icx, &lt, None);
                        let outlives = ty::Binder::bind(ty::OutlivesPredicate(region, bound));
                        predicates.push((outlives.to_predicate(), lt.span));
                    }
                    _ => bug!(),
                });
            }
            _ => bug!(),
        }
    }

    // Collect the predicates that were written inline by the user on each
    // type parameter (e.g., `<T: Foo>`).
    for param in &ast_generics.params {
        if let GenericParamKind::Type { .. } = param.kind {
            let name = param.name.ident().as_interned_str();
            let param_ty = ty::ParamTy::new(index, name).to_ty(tcx);
            index += 1;

            let sized = SizedByDefault::Yes;
            let bounds = AstConv::compute_bounds(&icx, param_ty, &param.bounds, sized, param.span);
            predicates.extend(bounds.predicates(tcx, param_ty));
        }
    }

    // Add in the bounds that appear in the where-clause.
    let where_clause = &ast_generics.where_clause;
    for predicate in &where_clause.predicates {
        match predicate {
            &hir::WherePredicate::BoundPredicate(ref bound_pred) => {
                let ty = icx.to_ty(&bound_pred.bounded_ty);

                // Keep the type around in a dummy predicate, in case of no bounds.
                // That way, `where Ty:` is not a complete noop (see #53696) and `Ty`
                // is still checked for WF.
                if bound_pred.bounds.is_empty() {
                    if let ty::Param(_) = ty.sty {
                        // This is a `where T:`, which can be in the HIR from the
                        // transformation that moves `?Sized` to `T`'s declaration.
                        // We can skip the predicate because type parameters are
                        // trivially WF, but also we *should*, to avoid exposing
                        // users who never wrote `where Type:,` themselves, to
                        // compiler/tooling bugs from not handling WF predicates.
                    } else {
                        let span = bound_pred.bounded_ty.span;
                        let predicate = ty::OutlivesPredicate(ty, tcx.mk_region(ty::ReEmpty));
                        predicates.push(
                            (ty::Predicate::TypeOutlives(ty::Binder::dummy(predicate)), span)
                        );
                    }
                }

                for bound in bound_pred.bounds.iter() {
                    match bound {
                        &hir::GenericBound::Trait(ref poly_trait_ref, _) => {
                            let mut bounds = Bounds::default();

                            let (trait_ref, _) = AstConv::instantiate_poly_trait_ref(
                                &icx,
                                poly_trait_ref,
                                ty,
                                &mut bounds,
                            );

                            predicates.push((trait_ref.to_predicate(), poly_trait_ref.span));
                            predicates.extend(bounds.predicates(tcx, ty));
                        }

                        &hir::GenericBound::Outlives(ref lifetime) => {
                            let region = AstConv::ast_region_to_region(&icx, lifetime, None);
                            let pred = ty::Binder::bind(ty::OutlivesPredicate(ty, region));
                            predicates.push((ty::Predicate::TypeOutlives(pred), lifetime.span))
                        }
                    }
                }
            }

            &hir::WherePredicate::RegionPredicate(ref region_pred) => {
                let r1 = AstConv::ast_region_to_region(&icx, &region_pred.lifetime, None);
                predicates.extend(region_pred.bounds.iter().map(|bound| {
                    let (r2, span) = match bound {
                        hir::GenericBound::Outlives(lt) => {
                            (AstConv::ast_region_to_region(&icx, lt, None), lt.span)
                        }
                        _ => bug!(),
                    };
                    let pred = ty::Binder::bind(ty::OutlivesPredicate(r1, r2));

                    (ty::Predicate::RegionOutlives(pred), span)
                }))
            }

            &hir::WherePredicate::EqPredicate(..) => {
                // FIXME(#20041)
            }
        }
    }

    // Add predicates from associated type bounds.
    if let Some((self_trait_ref, trait_items)) = is_trait {
        predicates.extend(trait_items.iter().flat_map(|trait_item_ref| {
            let trait_item = tcx.hir().trait_item(trait_item_ref.id);
            let bounds = match trait_item.node {
                hir::TraitItemKind::Type(ref bounds, _) => bounds,
                _ => return Vec::new().into_iter()
            };

            let assoc_ty =
                tcx.mk_projection(tcx.hir().local_def_id_from_hir_id(trait_item.hir_id),
                    self_trait_ref.substs);

            let bounds = AstConv::compute_bounds(
                &ItemCtxt::new(tcx, def_id),
                assoc_ty,
                bounds,
                SizedByDefault::Yes,
                trait_item.span,
            );

            bounds.predicates(tcx, assoc_ty).into_iter()
        }))
    }

    let mut predicates = predicates.predicates;

    // Subtle: before we store the predicates into the tcx, we
    // sort them so that predicates like `T: Foo<Item=U>` come
    // before uses of `U`.  This avoids false ambiguity errors
    // in trait checking. See `setup_constraining_predicates`
    // for details.
    if let Node::Item(&Item {
        node: ItemKind::Impl(..),
        ..
    }) = node
    {
        let self_ty = tcx.type_of(def_id);
        let trait_ref = tcx.impl_trait_ref(def_id);
        cgp::setup_constraining_predicates(
            tcx,
            &mut predicates,
            trait_ref,
            &mut cgp::parameters_for_impl(self_ty, trait_ref),
        );
    }

    let result = tcx.arena.alloc(ty::GenericPredicates {
        parent: generics.parent,
        predicates,
    });
    debug!("explicit_predicates_of(def_id={:?}) = {:?}", def_id, result);
    result
}

/// Converts a specific `GenericBound` from the AST into a set of
/// predicates that apply to the self type. A vector is returned
/// because this can be anywhere from zero predicates (`T: ?Sized` adds no
/// predicates) to one (`T: Foo`) to many (`T: Bar<X=i32>` adds `T: Bar`
/// and `<T as Bar>::X == i32`).
fn predicates_from_bound<'tcx>(
    astconv: &dyn AstConv<'tcx>,
    param_ty: Ty<'tcx>,
    bound: &'tcx hir::GenericBound,
) -> Vec<(ty::Predicate<'tcx>, Span)> {
    match *bound {
        hir::GenericBound::Trait(ref tr, hir::TraitBoundModifier::None) => {
            let mut bounds = Bounds::default();
            let (pred, _) = astconv.instantiate_poly_trait_ref(tr, param_ty, &mut bounds);
            iter::once((pred.to_predicate(), tr.span))
                .chain(bounds.predicates(astconv.tcx(), param_ty))
                .collect()
        }
        hir::GenericBound::Outlives(ref lifetime) => {
            let region = astconv.ast_region_to_region(lifetime, None);
            let pred = ty::Binder::bind(ty::OutlivesPredicate(param_ty, region));
            vec![(ty::Predicate::TypeOutlives(pred), lifetime.span)]
        }
        hir::GenericBound::Trait(_, hir::TraitBoundModifier::Maybe) => vec![],
    }
}

fn compute_sig_of_foreign_fn_decl<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    decl: &'tcx hir::FnDecl,
    abi: abi::Abi,
) -> ty::PolyFnSig<'tcx> {
    let unsafety = if abi == abi::Abi::RustIntrinsic {
        intrisic_operation_unsafety(&*tcx.item_name(def_id).as_str())
    } else {
        hir::Unsafety::Unsafe
    };
    let fty = AstConv::ty_of_fn(&ItemCtxt::new(tcx, def_id), unsafety, abi, decl);

    // Feature gate SIMD types in FFI, since I am not sure that the
    // ABIs are handled at all correctly. -huonw
    if abi != abi::Abi::RustIntrinsic
        && abi != abi::Abi::PlatformIntrinsic
        && !tcx.features().simd_ffi
    {
        let check = |ast_ty: &hir::Ty, ty: Ty<'_>| {
            if ty.is_simd() {
                tcx.sess
                   .struct_span_err(
                       ast_ty.span,
                       &format!(
                           "use of SIMD type `{}` in FFI is highly experimental and \
                            may result in invalid code",
                           tcx.hir().hir_to_pretty_string(ast_ty.hir_id)
                       ),
                   )
                   .help("add #![feature(simd_ffi)] to the crate attributes to enable")
                   .emit();
            }
        };
        for (input, ty) in decl.inputs.iter().zip(*fty.inputs().skip_binder()) {
            check(&input, ty)
        }
        if let hir::Return(ref ty) = decl.output {
            check(&ty, *fty.output().skip_binder())
        }
    }

    fty
}

fn is_foreign_item(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    match tcx.hir().get_if_local(def_id) {
        Some(Node::ForeignItem(..)) => true,
        Some(_) => false,
        _ => bug!("is_foreign_item applied to non-local def-id {:?}", def_id),
    }
}

fn static_mutability(tcx: TyCtxt<'_>, def_id: DefId) -> Option<hir::Mutability> {
    match tcx.hir().get_if_local(def_id) {
        Some(Node::Item(&hir::Item {
            node: hir::ItemKind::Static(_, mutbl, _), ..
        })) |
        Some(Node::ForeignItem( &hir::ForeignItem {
            node: hir::ForeignItemKind::Static(_, mutbl), ..
        })) => Some(mutbl),
        Some(_) => None,
        _ => bug!("static_mutability applied to non-local def-id {:?}", def_id),
    }
}

fn from_target_feature(
    tcx: TyCtxt<'_>,
    id: DefId,
    attr: &ast::Attribute,
    whitelist: &FxHashMap<String, Option<Symbol>>,
    target_features: &mut Vec<Symbol>,
) {
    let list = match attr.meta_item_list() {
        Some(list) => list,
        None => return,
    };
    let bad_item = |span| {
        let msg = "malformed `target_feature` attribute input";
        let code = "enable = \"..\"".to_owned();
        tcx.sess.struct_span_err(span, &msg)
            .span_suggestion(span, "must be of the form", code, Applicability::HasPlaceholders)
            .emit();
    };
    let rust_features = tcx.features();
    for item in list {
        // Only `enable = ...` is accepted in the meta-item list.
        if !item.check_name(sym::enable) {
            bad_item(item.span());
            continue;
        }

        // Must be of the form `enable = "..."` (a string).
        let value = match item.value_str() {
            Some(value) => value,
            None => {
                bad_item(item.span());
                continue;
            }
        };

        // We allow comma separation to enable multiple features.
        target_features.extend(value.as_str().split(',').filter_map(|feature| {
            // Only allow whitelisted features per platform.
            let feature_gate = match whitelist.get(feature) {
                Some(g) => g,
                None => {
                    let msg = format!(
                        "the feature named `{}` is not valid for this target",
                        feature
                    );
                    let mut err = tcx.sess.struct_span_err(item.span(), &msg);
                    err.span_label(
                        item.span(),
                        format!("`{}` is not valid for this target", feature),
                    );
                    if feature.starts_with("+") {
                        let valid = whitelist.contains_key(&feature[1..]);
                        if valid {
                            err.help("consider removing the leading `+` in the feature name");
                        }
                    }
                    err.emit();
                    return None;
                }
            };

            // Only allow features whose feature gates have been enabled.
            let allowed = match feature_gate.as_ref().map(|s| *s) {
                Some(sym::arm_target_feature) => rust_features.arm_target_feature,
                Some(sym::aarch64_target_feature) => rust_features.aarch64_target_feature,
                Some(sym::hexagon_target_feature) => rust_features.hexagon_target_feature,
                Some(sym::powerpc_target_feature) => rust_features.powerpc_target_feature,
                Some(sym::mips_target_feature) => rust_features.mips_target_feature,
                Some(sym::avx512_target_feature) => rust_features.avx512_target_feature,
                Some(sym::mmx_target_feature) => rust_features.mmx_target_feature,
                Some(sym::sse4a_target_feature) => rust_features.sse4a_target_feature,
                Some(sym::tbm_target_feature) => rust_features.tbm_target_feature,
                Some(sym::wasm_target_feature) => rust_features.wasm_target_feature,
                Some(sym::cmpxchg16b_target_feature) => rust_features.cmpxchg16b_target_feature,
                Some(sym::adx_target_feature) => rust_features.adx_target_feature,
                Some(sym::movbe_target_feature) => rust_features.movbe_target_feature,
                Some(sym::rtm_target_feature) => rust_features.rtm_target_feature,
                Some(sym::f16c_target_feature) => rust_features.f16c_target_feature,
                Some(name) => bug!("unknown target feature gate {}", name),
                None => true,
            };
            if !allowed && id.is_local() {
                feature_gate::emit_feature_err(
                    &tcx.sess.parse_sess,
                    feature_gate.unwrap(),
                    item.span(),
                    feature_gate::GateIssue::Language,
                    &format!("the target feature `{}` is currently unstable", feature),
                );
            }
            Some(Symbol::intern(feature))
        }));
    }
}

fn linkage_by_name(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> Linkage {
    use rustc::mir::mono::Linkage::*;

    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but whitelist them anyway and trust that
    // the user knows what s/he's doing. Who knows, unanticipated use cases
    // may pop up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "appending" => Appending,
        "available_externally" => AvailableExternally,
        "common" => Common,
        "extern_weak" => ExternalWeak,
        "external" => External,
        "internal" => Internal,
        "linkonce" => LinkOnceAny,
        "linkonce_odr" => LinkOnceODR,
        "private" => Private,
        "weak" => WeakAny,
        "weak_odr" => WeakODR,
        _ => {
            let span = tcx.hir().span_if_local(def_id);
            if let Some(span) = span {
                tcx.sess.span_fatal(span, "invalid linkage specified")
            } else {
                tcx.sess
                   .fatal(&format!("invalid linkage specified: {}", name))
            }
        }
    }
}

fn codegen_fn_attrs(tcx: TyCtxt<'_>, id: DefId) -> CodegenFnAttrs {
    let attrs = tcx.get_attrs(id);

    let mut codegen_fn_attrs = CodegenFnAttrs::new();

    let whitelist = tcx.target_features_whitelist(LOCAL_CRATE);

    let mut inline_span = None;
    for attr in attrs.iter() {
        if attr.check_name(sym::cold) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::COLD;
        } else if attr.check_name(sym::rustc_allocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR;
        } else if attr.check_name(sym::unwind) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::UNWIND;
        } else if attr.check_name(sym::ffi_returns_twice) {
            if tcx.is_foreign_item(id) {
                codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_RETURNS_TWICE;
            } else {
                // `#[ffi_returns_twice]` is only allowed `extern fn`s.
                struct_span_err!(
                    tcx.sess,
                    attr.span,
                    E0724,
                    "`#[ffi_returns_twice]` may only be used on foreign functions"
                ).emit();
            }
        } else if attr.check_name(sym::rustc_allocator_nounwind) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_ALLOCATOR_NOUNWIND;
        } else if attr.check_name(sym::naked) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NAKED;
        } else if attr.check_name(sym::no_mangle) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
        } else if attr.check_name(sym::rustc_std_internal_symbol) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
        } else if attr.check_name(sym::no_debug) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_DEBUG;
        } else if attr.check_name(sym::used) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED;
        } else if attr.check_name(sym::thread_local) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::THREAD_LOCAL;
        } else if attr.check_name(sym::export_name) {
            if let Some(s) = attr.value_str() {
                if s.as_str().contains("\0") {
                    // `#[export_name = ...]` will be converted to a null-terminated string,
                    // so it may not contain any null characters.
                    struct_span_err!(
                        tcx.sess,
                        attr.span,
                        E0648,
                        "`export_name` may not contain null characters"
                    ).emit();
                }
                codegen_fn_attrs.export_name = Some(s);
            }
        } else if attr.check_name(sym::target_feature) {
            if tcx.fn_sig(id).unsafety() == Unsafety::Normal {
                let msg = "#[target_feature(..)] can only be applied to `unsafe` functions";
                tcx.sess.struct_span_err(attr.span, msg)
                    .span_label(attr.span, "can only be applied to `unsafe` functions")
                    .span_label(tcx.def_span(id), "not an `unsafe` function")
                    .emit();
            }
            from_target_feature(
                tcx,
                id,
                attr,
                &whitelist,
                &mut codegen_fn_attrs.target_features,
            );
        } else if attr.check_name(sym::linkage) {
            if let Some(val) = attr.value_str() {
                codegen_fn_attrs.linkage = Some(linkage_by_name(tcx, id, &val.as_str()));
            }
        } else if attr.check_name(sym::link_section) {
            if let Some(val) = attr.value_str() {
                if val.as_str().bytes().any(|b| b == 0) {
                    let msg = format!(
                        "illegal null byte in link_section \
                         value: `{}`",
                        &val
                    );
                    tcx.sess.span_err(attr.span, &msg);
                } else {
                    codegen_fn_attrs.link_section = Some(val);
                }
            }
        } else if attr.check_name(sym::link_name) {
            codegen_fn_attrs.link_name = attr.value_str();
        }
    }

    codegen_fn_attrs.inline = attrs.iter().fold(InlineAttr::None, |ia, attr| {
        if attr.path != sym::inline {
            return ia;
        }
        match attr.meta().map(|i| i.node) {
            Some(MetaItemKind::Word) => {
                mark_used(attr);
                InlineAttr::Hint
            }
            Some(MetaItemKind::List(ref items)) => {
                mark_used(attr);
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    span_err!(
                        tcx.sess.diagnostic(),
                        attr.span,
                        E0534,
                        "expected one argument"
                    );
                    InlineAttr::None
                } else if list_contains_name(&items[..], sym::always) {
                    InlineAttr::Always
                } else if list_contains_name(&items[..], sym::never) {
                    InlineAttr::Never
                } else {
                    span_err!(
                        tcx.sess.diagnostic(),
                        items[0].span(),
                        E0535,
                        "invalid argument"
                    );

                    InlineAttr::None
                }
            }
            Some(MetaItemKind::NameValue(_)) => ia,
            None => ia,
        }
    });

    codegen_fn_attrs.optimize = attrs.iter().fold(OptimizeAttr::None, |ia, attr| {
        if attr.path != sym::optimize {
            return ia;
        }
        let err = |sp, s| span_err!(tcx.sess.diagnostic(), sp, E0722, "{}", s);
        match attr.meta().map(|i| i.node) {
            Some(MetaItemKind::Word) => {
                err(attr.span, "expected one argument");
                ia
            }
            Some(MetaItemKind::List(ref items)) => {
                mark_used(attr);
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    err(attr.span, "expected one argument");
                    OptimizeAttr::None
                } else if list_contains_name(&items[..], sym::size) {
                    OptimizeAttr::Size
                } else if list_contains_name(&items[..], sym::speed) {
                    OptimizeAttr::Speed
                } else {
                    err(items[0].span(), "invalid argument");
                    OptimizeAttr::None
                }
            }
            Some(MetaItemKind::NameValue(_)) => ia,
            None => ia,
        }
    });

    // If a function uses #[target_feature] it can't be inlined into general
    // purpose functions as they wouldn't have the right target features
    // enabled. For that reason we also forbid #[inline(always)] as it can't be
    // respected.
    if codegen_fn_attrs.target_features.len() > 0 {
        if codegen_fn_attrs.inline == InlineAttr::Always {
            if let Some(span) = inline_span {
                tcx.sess.span_err(
                    span,
                    "cannot use #[inline(always)] with \
                     #[target_feature]",
                );
            }
        }
    }

    // Weak lang items have the same semantics as "std internal" symbols in the
    // sense that they're preserved through all our LTO passes and only
    // strippable by the linker.
    //
    // Additionally weak lang items have predetermined symbol names.
    if tcx.is_weak_lang_item(id) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
    }
    if let Some(name) = weak_lang_items::link_name(&attrs) {
        codegen_fn_attrs.export_name = Some(name);
        codegen_fn_attrs.link_name = Some(name);
    }

    // Internal symbols to the standard library all have no_mangle semantics in
    // that they have defined symbol names present in the function name. This
    // also applies to weak symbols where they all have known symbol names.
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
    }

    codegen_fn_attrs
}
