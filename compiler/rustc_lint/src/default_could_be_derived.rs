use hir::def::{CtorOf, DefKind, Res};
use rustc_ast::Recovered;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, Diag, MultiSpan};
use rustc_hir as hir;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::Symbol;
use rustc_span::def_id::DefId;
use rustc_span::symbol::sym;

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `default_overrides_default_fields` lint checks for manual `impl` blocks of the
    /// `Default` trait of types with default field values.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(default_field_values)]
    /// struct Foo {
    ///     x: i32 = 101,
    ///     y: NonDefault,
    /// }
    ///
    /// struct NonDefault;
    ///
    /// #[deny(default_overrides_default_fields)]
    /// impl Default for Foo {
    ///     fn default() -> Foo {
    ///         Foo { x: 100, y: NonDefault }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Manually writing a `Default` implementation for a type that has
    /// default field values runs the risk of diverging behavior between
    /// `Type { .. }` and `<Type as Default>::default()`, which would be a
    /// foot-gun for users of that type that would expect these to be
    /// equivalent. If `Default` can't be derived due to some fields not
    /// having a `Default` implementation, we encourage the use of `..` for
    /// the fields that do have a default field value.
    pub DEFAULT_OVERRIDES_DEFAULT_FIELDS,
    Deny,
    "detect `Default` impl that should use the type's default field values",
    @feature_gate = default_field_values;
}

pub(crate) struct DefaultCouldBeDerived;

impl_lint_pass!(DefaultCouldBeDerived => [DEFAULT_OVERRIDES_DEFAULT_FIELDS]);

impl<'tcx> LateLintPass<'tcx> for DefaultCouldBeDerived {
    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        // Look for manual implementations of `Default`.
        let Some(default_def_id) = cx.tcx.get_diagnostic_item(sym::Default) else { return };
        let hir::ImplItemKind::Fn(_sig, body_id) = impl_item.kind else { return };
        let assoc = cx.tcx.associated_item(impl_item.owner_id);
        let parent = assoc.container_id(cx.tcx);
        if cx.tcx.has_attr(parent, sym::automatically_derived) {
            // We don't care about what `#[derive(Default)]` produces in this lint.
            return;
        }
        let Some(trait_ref) = cx.tcx.impl_trait_ref(parent) else { return };
        let trait_ref = trait_ref.instantiate_identity();
        if trait_ref.def_id != default_def_id {
            return;
        }
        let ty = trait_ref.self_ty();
        let ty::Adt(def, _) = ty.kind() else { return };

        // We now know we have a manually written definition of a `<Type as Default>::default()`.

        let hir = cx.tcx.hir();

        let type_def_id = def.did();
        let body = hir.body(body_id);

        // FIXME: evaluate bodies with statements and evaluate bindings to see if they would be
        // derivable.
        let hir::ExprKind::Block(hir::Block { stmts: _, expr: Some(expr), .. }, None) =
            body.value.kind
        else {
            return;
        };

        // Keep a mapping of field name to `hir::FieldDef` for every field in the type. We'll use
        // these to check for things like checking whether it has a default or using its span for
        // suggestions.
        let orig_fields = match hir.get_if_local(type_def_id) {
            Some(hir::Node::Item(hir::Item {
                kind:
                    hir::ItemKind::Struct(hir::VariantData::Struct { fields, recovered: _ }, _generics),
                ..
            })) => fields.iter().map(|f| (f.ident.name, f)).collect::<FxHashMap<_, _>>(),
            _ => return,
        };

        // We check `fn default()` body is a single ADT literal and get all the fields that are
        // being set.
        let hir::ExprKind::Struct(_qpath, fields, tail) = expr.kind else { return };

        // We have a struct literal
        //
        // struct Foo {
        //     field: Type,
        // }
        //
        // impl Default for Foo {
        //     fn default() -> Foo {
        //         Foo {
        //             field: val,
        //         }
        //     }
        // }
        //
        // We would suggest `#[derive(Default)]` if `field` has a default value, regardless of what
        // it is; we don't want to encourage divergent behavior between `Default::default()` and
        // `..`.

        if let hir::StructTailExpr::Base(_) = tail {
            // This is *very* niche. We'd only get here if someone wrote
            // impl Default for Ty {
            //     fn default() -> Ty {
            //         Ty { ..something() }
            //     }
            // }
            // where `something()` would have to be a call or path.
            // We have nothing meaninful to do with this.
            return;
        }

        // At least one of the fields with a default value have been overriden in
        // the `Default` implementation. We suggest removing it and relying on `..`
        // instead.
        let any_default_field_given =
            fields.iter().any(|f| orig_fields.get(&f.ident.name).and_then(|f| f.default).is_some());

        if !any_default_field_given {
            // None of the default fields were actually provided explicitly, so the manual impl
            // doesn't override them (the user used `..`), so there's no risk of divergent behavior.
            return;
        }

        let Some(local) = parent.as_local() else { return };
        let hir_id = cx.tcx.local_def_id_to_hir_id(local);
        let hir::Node::Item(item) = cx.tcx.hir_node(hir_id) else { return };
        cx.tcx.node_span_lint(DEFAULT_OVERRIDES_DEFAULT_FIELDS, hir_id, item.span, |diag| {
            mk_lint(cx.tcx, diag, type_def_id, parent, orig_fields, fields);
        });
    }
}

fn mk_lint(
    tcx: TyCtxt<'_>,
    diag: &mut Diag<'_, ()>,
    type_def_id: DefId,
    impl_def_id: DefId,
    orig_fields: FxHashMap<Symbol, &hir::FieldDef<'_>>,
    fields: &[hir::ExprField<'_>],
) {
    diag.primary_message("`Default` impl doesn't use the declared default field values");

    // For each field in the struct expression
    //   - if the field in the type has a default value, it should be removed
    //   - elif the field is an expression that could be a default value, it should be used as the
    //     field's default value (FIXME: not done).
    //   - else, we wouldn't touch this field, it would remain in the manual impl
    let mut removed_all_fields = true;
    for field in fields {
        if orig_fields.get(&field.ident.name).and_then(|f| f.default).is_some() {
            diag.span_label(field.expr.span, "this field has a default value");
        } else {
            removed_all_fields = false;
        }
    }

    if removed_all_fields {
        let msg = "to avoid divergence in behavior between `Struct { .. }` and \
                   `<Struct as Default>::default()`, derive the `Default`";
        if let Some(hir::Node::Item(impl_)) = tcx.hir().get_if_local(impl_def_id) {
            diag.multipart_suggestion_verbose(
                msg,
                vec![
                    (tcx.def_span(type_def_id).shrink_to_lo(), "#[derive(Default)] ".to_string()),
                    (impl_.span, String::new()),
                ],
                Applicability::MachineApplicable,
            );
        } else {
            diag.help(msg);
        }
    } else {
        let msg = "use the default values in the `impl` with `Struct { mandatory_field, .. }` to \
                   avoid them diverging over time";
        diag.help(msg);
    }
}

declare_lint! {
    /// The `default_field_overrides_default_field` lint checks for struct literals in field default
    /// values with fields that have in turn default values. These should instead be skipped and
    /// rely on `..` for them.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(default_field_values)]
    /// #![deny(default_field_overrides_default_field)]
    ///
    /// struct Foo {
    ///     x: Bar = Bar { x: 0 }, // `Foo { .. }.x.x` != `Bar { .. }.x`
    /// }
    ///
    /// struct Bar {
    ///     x: i32 = 101,
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Defaulting a field to a value different to that field's type already defined default can
    /// easily lead to confusion due to diverging behavior. Acknowleding that there can be reasons
    /// for one to write an API that does this, this is not outright rejected by the compiler,
    /// merely linted against.
    pub DEFAULT_FIELD_OVERRIDES_DEFAULT_FIELD,
    Deny,
    "detect default field value that should use the type's default field values",
    @feature_gate = default_field_values;
}

pub(crate) struct DefaultFieldOverride;

impl_lint_pass!(DefaultFieldOverride => [DEFAULT_FIELD_OVERRIDES_DEFAULT_FIELD]);

impl DefaultFieldOverride {
    fn lint_variant(&mut self, cx: &LateContext<'_>, data: &hir::VariantData<'_>) {
        if !cx.tcx.features().default_field_values() {
            return;
        }
        let hir::VariantData::Struct { fields, recovered: Recovered::No } = data else {
            return;
        };

        for default in fields.iter().filter_map(|f| f.default) {
            let body = cx.tcx.hir().body(default.body);
            let hir::ExprKind::Struct(hir::QPath::Resolved(_, path), fields, _) = body.value.kind
            else {
                continue;
            };
            let Res::Def(
                DefKind::Variant
                | DefKind::Struct
                | DefKind::Ctor(CtorOf::Variant | CtorOf::Struct, ..),
                def_id,
            ) = path.res
            else {
                continue;
            };
            let fields_set: FxHashSet<_> = fields.iter().map(|f| f.ident.name).collect();
            let variant = cx.tcx.expect_variant_res(path.res);
            let mut to_lint = vec![];
            let mut defs = vec![];

            for field in &variant.fields {
                if fields_set.contains(&field.name) {
                    for f in fields {
                        if f.ident.name == field.name
                            && let Some(default) = field.value
                        {
                            to_lint.push((f.expr.span, f.ident.name));
                            defs.push(cx.tcx.def_span(default));
                        }
                    }
                }
            }

            if to_lint.is_empty() {
                continue;
            }
            cx.tcx.node_span_lint(
                DEFAULT_FIELD_OVERRIDES_DEFAULT_FIELD,
                body.value.hir_id,
                to_lint.iter().map(|&(span, _)| span).collect::<Vec<_>>(),
                |diag| {
                    diag.primary_message("default field overrides that field's type's default");
                    diag.span_label(path.span, "when constructing this value");
                    let type_name = cx.tcx.item_name(def_id);
                    for (span, name) in to_lint {
                        diag.span_label(
                            span,
                            format!(
                                "this overrides the default of field `{name}` in `{type_name}`"
                            ),
                        );
                    }
                    let mut overriden_spans: MultiSpan = defs.clone().into();
                    overriden_spans.push_span_label(cx.tcx.def_span(def_id), "");
                    diag.span_note(
                        overriden_spans,
                        format!(
                            "{this} field's default value in `{type_name}` is overriden",
                            this = if defs.len() == 1 { "this" } else { "these" }
                        ),
                    );
                },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for DefaultFieldOverride {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        let hir::ItemKind::Struct(data, _) = item.kind else { return };
        self.lint_variant(cx, &data);
    }
    fn check_variant(&mut self, cx: &LateContext<'_>, variant: &hir::Variant<'_>) {
        self.lint_variant(cx, &variant.data);
    }
}
