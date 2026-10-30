use clippy_config::Conf;
use clippy_config::types::{DisallowedPath, create_disallowed_map};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::paths::PathNS;
use clippy_utils::ty::get_field_def_id_by_name;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{
    AmbigArg, Expr, ExprKind, Item, ItemKind, Pat, PatKind, PolyTraitRef, PrimTy, Ty, TyKind, UseKind, UseTree,
};
use rustc_lint::{LateContext, LateLintPass, impl_lint_pass};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured fields in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// fields are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some fields are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-fields = [
    ///     # Can use a string as the path of the disallowed field.
    ///     "std::ops::Range::start",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::ops::Range::start" },
    ///     # When using an inline table, can add a `reason` for why the field
    ///     # is disallowed.
    ///     { path = "std::ops::Range::start", reason = "The start of the range is not used" },
    /// ]
    /// ```
    ///
    /// ```rust
    /// use std::ops::Range;
    ///
    /// let range = Range { start: 0, end: 1 };
    /// println!("{}", range.start); // `start` is disallowed in the config.
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// use std::ops::Range;
    ///
    /// let range = Range { start: 0, end: 1 };
    /// println!("{}", range.end); // `end` is _not_ disallowed in the config.
    /// ```
    #[clippy::version = "1.95.0"]
    pub DISALLOWED_FIELDS,
    style,
    "declaration of a disallowed field use"
}

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured methods and functions in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// methods are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some methods are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = [
    ///     # Can use a string as the path of the disallowed method.
    ///     "std::boxed::Box::new",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::time::Instant::now" },
    ///     # When using an inline table, can add a `reason` for why the method
    ///     # is disallowed.
    ///     { path = "std::vec::Vec::leak", reason = "no leaking memory" },
    ///     # Can also add a `replacement` that will be offered as a suggestion.
    ///     { path = "std::sync::Mutex::new", reason = "prefer faster & simpler non-poisonable mutex", replacement = "parking_lot::Mutex::new" },
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::fs::InvalidPath", reason = "use alternative instead", allow-invalid = true },
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// let xs = vec![1, 2, 3, 4];
    /// xs.leak(); // Vec::leak is disallowed in the config.
    /// // The diagnostic contains the message "no leaking memory".
    ///
    /// let _now = Instant::now(); // Instant::now is disallowed in the config.
    ///
    /// let _box = Box::new(3); // Box::new is disallowed in the config.
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let mut xs = Vec::new(); // Vec::new is _not_ disallowed in the config.
    /// xs.push(123); // Vec::push is _not_ disallowed in the config.
    /// ```
    #[clippy::version = "1.49.0"]
    pub DISALLOWED_METHODS,
    style,
    "use of a disallowed method call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured types in clippy.toml.
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// types are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some types are undesirable in certain contexts.
    ///
    /// ### Example:
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-types = [
    ///     # Can use a string as the path of the disallowed type.
    ///     "std::collections::BTreeMap",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::net::TcpListener" },
    ///     # When using an inline table, can add a `reason` for why the type
    ///     # is disallowed.
    ///     { path = "std::net::Ipv4Addr", reason = "no IPv4 allowed" },
    ///     # Can also add a `replacement` that will be offered as a suggestion.
    ///     { path = "std::sync::Mutex", reason = "prefer faster & simpler non-poisonable mutex", replacement = "parking_lot::Mutex" },
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::invalid::Type", reason = "use alternative instead", allow-invalid = true }
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// use std::collections::BTreeMap;
    /// // or its use
    /// let x = std::collections::BTreeMap::new();
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // A similar type that is allowed by the config
    /// use std::collections::HashMap;
    /// ```
    #[clippy::version = "1.55.0"]
    pub DISALLOWED_TYPES,
    style,
    "use of disallowed types"
}

impl_lint_pass!(DisallowedItems => [
    DISALLOWED_FIELDS,
    DISALLOWED_METHODS,
    DISALLOWED_TYPES,
]);

pub struct DisallowedItems {
    methods: DefIdMap<(&'static str, &'static DisallowedPath)>,
    tys: DefIdMap<(&'static str, &'static DisallowedPath)>,
    prim_tys: FxHashMap<PrimTy, (&'static str, &'static DisallowedPath)>,
    fields: DefIdMap<(&'static str, &'static DisallowedPath)>,
}

impl DisallowedItems {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let (methods, _) = create_disallowed_map(
            tcx,
            &conf.disallowed_methods,
            PathNS::Value,
            |def_kind| {
                matches!(
                    def_kind,
                    DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::AssocFn
                )
            },
            "function",
            false,
        );
        let (tys, prim_tys) = create_disallowed_map(
            tcx,
            &conf.disallowed_types,
            PathNS::Type,
            |def_kind| {
                matches!(
                    def_kind,
                    DefKind::Struct
                        | DefKind::Union
                        | DefKind::Enum
                        | DefKind::Trait
                        | DefKind::TyAlias
                        | DefKind::ForeignTy
                        | DefKind::AssocTy
                )
            },
            "type",
            true,
        );
        let (fields, _) = create_disallowed_map(
            tcx,
            &conf.disallowed_fields,
            PathNS::Field,
            |def_kind| matches!(def_kind, DefKind::Field),
            "field",
            false,
        );
        Self {
            methods,
            tys,
            prim_tys,
            fields,
        }
    }

    fn check_ty_res_emit(&self, cx: &LateContext<'_>, res: &Res, span: Span) {
        let (path, disallowed_path) = match res {
            Res::Def(_, did) if let Some(&x) = self.tys.get(did) => x,
            Res::PrimTy(prim) if let Some(&x) = self.prim_tys.get(prim) => x,
            _ => return,
        };
        span_lint_and_then(
            cx,
            DISALLOWED_TYPES,
            span,
            format!("use of a disallowed type `{path}`"),
            disallowed_path.diag_amendment(span),
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for DisallowedItems {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (lint, msg, table, id, span) = match &expr.kind {
            ExprKind::Path(path) if let Res::Def(_, id) = cx.qpath_res(path, expr.hir_id) => (
                DISALLOWED_METHODS,
                "use of a disallowed method",
                &self.methods,
                id,
                expr.span,
            ),
            ExprKind::MethodCall(name, ..) if let Some(id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) => {
                (
                    DISALLOWED_METHODS,
                    "use of a disallowed method",
                    &self.methods,
                    id,
                    name.ident.span,
                )
            },
            // Very round-about way to get the field `DefId` from the expr: first we get its
            // parent `Ty`. Then we go through all its fields to find the one with the expected
            // name and get the `DefId` from it.
            ExprKind::Field(e, ident)
                if let Some(parent_ty) = cx.typeck_results().expr_ty_adjusted_opt(e)
                    && let Some(id) = get_field_def_id_by_name(parent_ty, ident.name) =>
            {
                (
                    DISALLOWED_FIELDS,
                    "use of a disallowed field",
                    &self.fields,
                    id,
                    ident.span,
                )
            },
            _ => return,
        };
        if let Some(&(path, disallowed_path)) = table.get(&id)
            && expr.span.desugaring_kind().is_none()
        {
            span_lint_and_then(
                cx,
                lint,
                span,
                format!("{msg} `{path}`"),
                disallowed_path.diag_amendment(span),
            );
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(UseTree {
            prefix,
            kind: UseKind::Single(_),
        }) = &item.kind
            && let Some(res) = prefix.res.type_ns
        {
            self.check_ty_res_emit(cx, &res, item.span);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx, AmbigArg>) {
        if let TyKind::Path(path) = &ty.kind {
            self.check_ty_res_emit(cx, &cx.qpath_res(path, ty.hir_id), ty.span);
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>) {
        self.check_ty_res_emit(cx, &poly.trait_ref.path.res, poly.trait_ref.path.span);
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        let PatKind::Struct(struct_path, pat_fields, _) = pat.kind else {
            return;
        };
        match cx.typeck_results().qpath_res(&struct_path, pat.hir_id) {
            Res::Def(DefKind::Struct, struct_def_id) => {
                let adt_def = cx.tcx.adt_def(struct_def_id);
                for field in pat_fields {
                    if let Some(def_id) = adt_def.all_fields().find_map(|adt_field| {
                        if field.ident.name == adt_field.name {
                            Some(adt_field.did)
                        } else {
                            None
                        }
                    }) && let Some(&(path, disallowed_path)) = self.fields.get(&def_id)
                    {
                        span_lint_and_then(
                            cx,
                            DISALLOWED_FIELDS,
                            field.span,
                            format!("use of a disallowed field `{path}`"),
                            disallowed_path.diag_amendment(field.span),
                        );
                    }
                }
            },
            Res::Def(DefKind::Variant, variant_def_id) => {
                let enum_def_id = cx.tcx.parent(variant_def_id);
                let variant = cx.tcx.adt_def(enum_def_id).variant_with_id(variant_def_id);

                for field in pat_fields {
                    if let Some(def_id) = variant.fields.iter().find_map(|adt_field| {
                        if field.ident.name == adt_field.name {
                            Some(adt_field.did)
                        } else {
                            None
                        }
                    }) && let Some(&(path, disallowed_path)) = self.fields.get(&def_id)
                    {
                        span_lint_and_then(
                            cx,
                            DISALLOWED_FIELDS,
                            field.span,
                            format!("use of a disallowed field `{path}`"),
                            disallowed_path.diag_amendment(field.span),
                        );
                    }
                }
            },
            _ => {},
        }
    }
}
