use std::cmp::min;

use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diag};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::DefId;
use rustc_middle::mir;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::{Session, declare_lint, impl_lint_pass};
use rustc_span::symbol::{kw, sym};
use rustc_span::{Span, Symbol};

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `default_could_be_derived` lint checks for manual `impl` blocks
    /// of the `Default` trait that could have been derived.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![feature(default_field_values)]
    /// struct A {
    ///     b: Option<i32>,
    ///     c: Option<i32> = None,
    /// }
    ///
    /// #[deny(default_could_be_derived)]
    /// impl Default for A {
    ///     fn default() -> A {
    ///         A {
    ///             b: None,
    ///             c: Some(0),
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `#[derive(Default)]` uses the `Default` impl for every field of your
    /// type. If your manual `Default` impl either invokes `Default::default()`
    /// or uses the same value that that associated function produces, then it
    /// is better to use the derive to avoid the different `Default` impls from
    /// diverging over time.
    ///
    /// This lint also triggers on cases where there the type has no fields,
    /// so the derive for `Default` for a struct is trivial, and for an enum
    /// variant with no fields, which can be annotated with `#[default]`.
    pub DEFAULT_COULD_BE_DERIVED,
    Warn,
    "detect `Default` impl that could be derived",
    // FIXME: this logic can be extended to check for types without default field values
    // doing the same as `clippy::derivable_impls`, with slightly more information thanks
    // to the added "default equivalence" table.
    @feature_gate = default_field_values;
}

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
    Warn,
    "detect `Default` impl that should use the type's default field values",
    @feature_gate = default_field_values;
}

#[derive(Default)]
pub(crate) struct DefaultCouldBeDerived {
    data: Option<Data>,
}

struct Data {
    type_def_id: DefId,
    parent: DefId,
}

impl_lint_pass!(DefaultCouldBeDerived => [DEFAULT_COULD_BE_DERIVED, DEFAULT_OVERRIDES_DEFAULT_FIELDS]);

impl<'tcx> LateLintPass<'tcx> for DefaultCouldBeDerived {
    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        let Some(default_def_id) = cx.tcx.get_diagnostic_item(sym::Default) else { return };
        let assoc = cx.tcx.associated_item(impl_item.owner_id);
        let parent = assoc.container_id(cx.tcx);
        if cx.tcx.has_attr(parent, sym::automatically_derived) {
            return;
        }
        let Some(trait_ref) = cx.tcx.impl_trait_ref(parent) else { return };
        let trait_ref = trait_ref.instantiate_identity();
        if trait_ref.def_id != default_def_id {
            return;
        }
        let ty = trait_ref.self_ty();
        let ty::Adt(def, _) = ty.kind() else { return };

        // We have a manually written definition of a `<Type as Default>::default()`. We store the
        // necessary metadata for further analysis of its body in `check_body`.
        self.data = Some(Data { type_def_id: def.did(), parent });
    }

    fn check_impl_item_post(&mut self, _cx: &LateContext<'_>, _impl_item: &hir::ImplItem<'_>) {
        // Clean up.
        self.data = None;
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &hir::Body<'tcx>) {
        // We perform this logic here instead of in `check_impl_item` so that we have unconditional
        // permission to call `cx.typeck_results()` (because we're within a body).
        let Some(Data { type_def_id, parent }) = self.data else {
            return;
        };
        // FIXME: evaluate bodies with statements and evaluate bindings to see if they would be
        // derivable.
        let hir::ExprKind::Block(hir::Block { stmts: _, expr: Some(expr), .. }, None) =
            body.value.kind
        else {
            return;
        };

        let hir = cx.tcx.hir();

        // Keep a mapping of field name to `hir::FieldDef` for every field in the type. We'll use
        // these to check for things like checking whether it has a default or using its span for
        // suggestions.
        let orig_fields = match hir.get_if_local(type_def_id) {
            Some(hir::Node::Item(hir::Item {
                kind:
                    hir::ItemKind::Struct(hir::VariantData::Struct { fields, recovered: _ }, _generics),
                ..
            })) => fields.iter().map(|f| (f.ident.name, f)).collect::<FxHashMap<_, _>>(),
            _ => Default::default(),
        };

        // We check `fn default()` body is a single ADT literal and all the fields are being
        // set to something equivalent to the corresponding types' `Default::default()`.
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
        // We suggest #[derive(Default)] if
        //  - `field` has a default value, regardless of what it is; we don't want to
        //    encourage divergent behavior between `Default::default()` and `..`
        //  - `val` is `Default::default()`
        //  - `val` matches the `Default::default()` body for that type
        //  - `val` is `0`
        //  - `val` is `false`

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

        // All the fields that have been provided can be used by `#[derive(Default)]`. This checks
        // against what is *already* in the type. It doesn't account for fields where we can suggest
        // adding a default field value.
        let all_defaultable = fields.iter().all(|f| {
            orig_fields.get(&f.ident.name).and_then(|f| f.default).is_some()
                || check_expr(cx, f.expr)
        });

        // Any of the fields provided either already has a default value or the expression passed
        // is either a `Default::default()` (or equivalent) call or an expression suitable to be
        // used as a default field value. In the former, we suggest `#[derive(Default)]`. In the
        // latter, we also suggest setting the field to the expression used in the impl for that
        // field.
        let any_defaultable = fields.iter().all(|f| {
            orig_fields.get(&f.ident.name).and_then(|f| f.default).is_some()
                || check_expr(cx, f.expr)
                || is_expr_const(cx, f.expr)
        });
        #[allow(rustc::potential_query_instability)]
        // The type has default field values. We will not attempt to suggest anything unless the
        // type already has at least one default.
        let any_default_field = orig_fields.iter().any(|(_, f)| f.default.is_some());

        // At least one of the fields with a default value have been overriden in
        // the `Default` implementation. We suggest removing it and relying on `..`
        // instead.
        let any_default_field_given =
            fields.iter().any(|f| orig_fields.get(&f.ident.name).and_then(|f| f.default).is_some());

        if !(any_default_field_given || (any_defaultable && any_default_field)) {
            return;
        }

        let Some(local) = parent.as_local() else { return };
        let hir_id = cx.tcx.local_def_id_to_hir_id(local);
        let hir::Node::Item(item) = cx.tcx.hir_node(hir_id) else { return };
        cx.tcx.node_span_lint(
            if any_default_field_given && !all_defaultable {
                DEFAULT_OVERRIDES_DEFAULT_FIELDS
            } else {
                DEFAULT_COULD_BE_DERIVED
            },
            hir_id,
            item.span,
            |diag| {
                mk_lint(
                    cx,
                    diag,
                    type_def_id,
                    item,
                    any_default_field_given,
                    orig_fields,
                    fields,
                    &tail,
                );
            },
        );
    }
}

fn mk_lint<'tcx>(
    cx: &LateContext<'tcx>,
    diag: &mut Diag<'_, ()>,
    type_def_id: DefId,
    item: &hir::Item<'_>,
    any_default_field_given: bool,
    orig_fields: FxHashMap<Symbol, &hir::FieldDef<'_>>,
    fields: &[hir::ExprField<'_>],
    tail: &hir::StructTailExpr<'_>,
) {
    diag.primary_message(if any_default_field_given {
        "`Default` impl doesn't use the declared default field values"
    } else {
        "`Default` impl that could be derived"
    });

    let mut removals = vec![];
    let mut additions = vec![];
    let removal_span = |removals: &mut Vec<Span>, idx: usize| {
        // We get the span for the field at `idx` *including* either the previous or
        // following `,`.
        let field = fields[idx];
        if idx > 0
            && let Some(prev_field) = fields.get(idx - 1)
        {
            // Span covering the current field *and* the prior `,` for the prior field.
            removals.push(prev_field.span.shrink_to_hi().to(field.span));
        } else if let Some(next_field) = fields.get(idx + 1) {
            // Span for the current field *and* its trailing comma, all the way to the
            // next field.
            removals.push(field.span.until(next_field.span));
        } else if idx + 1 == fields.len()
            && let hir::StructTailExpr::DefaultFields(span) = tail
        {
            // This is the last field *and* there's a `, ..`. This span covers this
            // entire field and the `, ..`.
            removals.push(field.span.until(*span));
        } else {
            // The span for the current field, without any commas. This is a fallback
            // that shouldn't really trigger.
            removals.push(field.span);
        }
    };

    // Hold on to your butts, friends. This is a lot of subte logic that I'm trying to
    // make as easy to follow as possible, but in the end I managed to confuse myself
    // so I'm writing the high level idea here first.
    //
    // For each field in the struct expression
    //   - if the field in the type has a default value, we will remove it
    //   - elif the field is an expression that could be a default value, we will remove
    //     it and add it to the type's field
    //   - else, we won't touch this field, it will remain in the impl, but if the
    //     previous field *was* removed we look for the span of the previous `,` to
    //     remove it
    let mut prev_removed = 0;
    let mut removed_all_fields = true;
    for (i, field) in fields.iter().enumerate() {
        if orig_fields.get(&field.ident.name).and_then(|f| f.default).is_some() {
            diag.span_label(field.expr.span, "this field has a default value");
            removal_span(&mut removals, i);
            prev_removed = i;
        } else if is_expr_const(cx, &field.expr) {
            diag.span_label(field.expr.span, "this value can be used as a default field value");
            removal_span(&mut removals, i);
            prev_removed = i;
            let snippet = cx.tcx.sess.source_map().span_to_snippet(field.expr.span);
            if let Ok(snippet) = snippet
                && let Some(def_field) = orig_fields.get(&field.ident.name)
                // If the value is the same as their type's `Default::default()`, don't
                // set the default field value. The user can do that if they want it.
                && !check_expr(cx, field.expr)
            {
                let snippet = reindent(&cx.tcx.sess, snippet, def_field.span);
                additions.push(match def_field.default {
                    Some(anon) => (anon.span, snippet),
                    None => (def_field.span.shrink_to_hi(), format!(" = {snippet}")),
                });
            }
        } else {
            if prev_removed + 1 == i {
                // Remove the `,` between the previous field which was
                // removed and the current one, which is kept.
                removals.push(fields[prev_removed].span.shrink_to_hi().until(field.span));
            }
            removed_all_fields = false;
        }
    }

    // We cheated above, given `S { a: x, b: y, c: z }` we might
    // suggest to remove one of the commas *twice*, which suggestions
    // really don't like, so we ensure that even if we're removing
    // multiple fields, they never overlap.
    removals.sort();
    let mut removals_iter = removals.iter_mut().peekable();
    while let Some(removal) = removals_iter.next() {
        if let Some(next) = removals_iter.peek() {
            if removal.overlaps(**next) {
                *removal = removal.with_hi(next.lo());
            }
        } else if let hir::StructTailExpr::None(span) = tail
            && removal.overlaps(*span)
        {
            *removal = removal.with_hi(span.lo());
        }
    }
    // We don't want empty spans, which suggestions also don't like.
    removals.retain(|s| s.lo() != s.hi());

    // Construct the suggestions.
    if !additions.is_empty() && removed_all_fields {
        // Suggest `#[derive(Default)]` as we removed all the fields and
        // moved their values to their fields' default.
        additions.insert(
            0,
            (cx.tcx.def_span(type_def_id).shrink_to_lo(), "#[derive(Default)] ".to_string()),
        );
        additions.push((item.span, String::new()));
        diag.multipart_suggestion_verbose(
            "set the default field values of your type to the value used in the \
                `Default` implementation and derive it",
            additions,
            Applicability::MachineApplicable,
        );
    } else if removed_all_fields {
        // Suggest `#[derive(Default)]` as we removed all the fields.
        diag.multipart_suggestion_verbose(
            "to avoid divergence in behavior between `Struct { .. }` and \
                `<Struct as Default>::default()`, derive the `Default`",
            vec![
                (cx.tcx.def_span(type_def_id).shrink_to_lo(), "#[derive(Default)] ".to_string()),
                (item.span, String::new()),
            ],
            Applicability::MachineApplicable,
        );
    } else {
        // Suggest moving some values to their field's default, leaving
        // the `impl Default for Type {}` item in place but using the
        // default values with `..`.
        let mut sugg: Vec<(Span, String)> = removals
            .into_iter()
            .map(|sp| (sp, String::new()))
            .chain(additions.into_iter())
            .collect();
        match tail {
            hir::StructTailExpr::Base(_) => {
                // There's already a trailing `..`, we don't need to suggest anything.
            }
            hir::StructTailExpr::DefaultFields(_) => {}
            hir::StructTailExpr::None(span) => {
                sugg.push((*span, ", ..".to_string()));
            }
        }
        // For:
        //
        // struct S {
        //     a: Ty,
        //     b: i32 = 101,
        // }
        //
        // impl Default for S {
        //     fn default() -> S {
        //         S {
        //             a: foo(),
        //             b: 100,
        //         }
        //     }
        // }
        //
        // We suggest
        //
        // impl Default for S {
        //     fn default() -> S {
        //         S {
        //             a: foo(), ..
        //         }
        //     }
        // }
        diag.multipart_suggestion_verbose(
            "use the default values in the `impl` to avoid them diverging over time",
            sugg,
            Applicability::MachineApplicable,
        );
    }
}

/// For the `Default` impl for this type, we see if it has a `Default::default()` body composed
/// only of a path, ctor or function call with no arguments. If so, we compare that `DefId`
/// against the `DefId` of this field's value if it is also a call/path/ctor.
/// If there's a match, it means that the contents of that type's `Default` impl are the
/// same to what the user wrote on *their* `Default` impl for this field.
fn check_path<'tcx>(
    cx: &LateContext<'tcx>,
    path: &hir::QPath<'_>,
    hir_id: hir::HirId,
    ty: Ty<'tcx>,
) -> bool {
    let res = cx.qpath_res(&path, hir_id);
    let Some(def_id) = res.opt_def_id() else { return false };
    let Some(default_fn_def_id) = cx.tcx.get_diagnostic_item(sym::default_fn) else {
        return false;
    };
    if default_fn_def_id == def_id {
        // We have `field: Default::default(),`. This is what the derive would do already.
        return true;
    }

    let args = ty::GenericArgs::for_item(cx.tcx, default_fn_def_id, |param, _| {
        if let ty::GenericParamDefKind::Lifetime = param.kind {
            cx.tcx.lifetimes.re_erased.into()
        } else if param.index == 0 && param.name == kw::SelfUpper {
            ty.into()
        } else {
            param.to_error(cx.tcx)
        }
    });
    let instance = Instance::try_resolve(cx.tcx, cx.typing_env(), default_fn_def_id, args);

    let Ok(Some(instance)) = instance else { return false };
    if let ty::InstanceKind::Item(def) = instance.def
        && !cx.tcx.is_mir_available(def)
    {
        // Avoid ICE while running rustdoc for not providing `optimized_mir` query.
        return false;
    }

    // Get the MIR Body for the `<FieldTy as Default>::default()` function.
    // If it is a value or call (either fn or ctor), we compare its DefId against the one for the
    // resolution of the expression we had in the path. This lets us identify, for example, that
    // the body of `<Vec<T> as Default>::default()` is a `Vec::new()`, and the field was being
    // initialized to `Vec::new()` as well.
    let body = cx.tcx.instance_mir(instance.def);
    for block_data in body.basic_blocks.iter() {
        if block_data.statements.len() == 1
            && let mir::StatementKind::Assign(assign) = &block_data.statements[0].kind
            && assign.0.local == mir::RETURN_PLACE
            && let mir::Rvalue::Aggregate(kind, _places) = &assign.1
            && let mir::AggregateKind::Adt(did, variant_index, _, _, _) = &**kind
            && let def = cx.tcx.adt_def(did)
            && let variant = &def.variant(*variant_index)
            && variant.fields.is_empty()
            && let Some((_, did)) = variant.ctor
            && did == def_id
        {
            return true;
        } else if block_data.statements.len() == 0
            && let Some(term) = &block_data.terminator
        {
            match &term.kind {
                mir::TerminatorKind::Call { func: mir::Operand::Constant(c), .. }
                    if let ty::FnDef(did, _args) = c.ty().kind()
                        && *did == def_id =>
                {
                    return true;
                }
                mir::TerminatorKind::TailCall { func: mir::Operand::Constant(c), .. }
                    if let ty::FnDef(did, _args) = c.ty().kind()
                        && *did == def_id =>
                {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

/// Given the snippet for an expression, indent it to fit in a field default value.
/// We remove the indent of every line but the first (keeping the general relative shape), and then
/// add the indent for the field span.
fn reindent(sess: &Session, snippet: String, field_span: Span) -> String {
    // Remove the preexisting indentation...
    let mut indent = usize::MAX;
    for line in snippet.lines().skip(1) {
        indent = min(indent, line.len() - line.trim_start().len());
    }
    if indent == usize::MAX {
        indent = 0;
    }
    let new_indent = sess.source_map().indentation_before(field_span).unwrap_or_else(String::new);
    snippet
        .lines()
        .enumerate()
        .map(|(i, s)| {
            if i == 0 {
                s.to_string()
            } else {
                // ...and add the indentation of the field.
                format!("{new_indent}{}", &s[indent..])
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Given an expression, determine if it would have been the same that would be used by
/// `#[derive(Default)]`.
fn check_expr(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    match expr.kind {
        hir::ExprKind::Lit(spanned_lit) => match spanned_lit.node {
            LitKind::Int(val, _) if val == 0 => true, // field: 0,
            LitKind::Bool(false) => true,             // field: false,
            _ => false,
        },
        hir::ExprKind::Call(hir::Expr { kind: hir::ExprKind::Path(path), hir_id, .. }, []) => {
            // `field: foo(),` or `field: Ty::assoc(),`
            let Some(ty) = cx.typeck_results().expr_ty_adjusted_opt(expr) else {
                return false;
            };
            check_path(cx, &path, *hir_id, ty)
        }
        hir::ExprKind::Path(path) => {
            // `field: qualified::Path,` or `field: <Ty as Trait>::Assoc,`
            let Some(ty) = cx.typeck_results().expr_ty_adjusted_opt(expr) else {
                return false;
            };
            check_path(cx, &path, expr.hir_id, ty)
        }
        _ => false,
    }
}

/// Given a path, determine if it corresponds to a `const` item.
fn is_path_const<'tcx>(cx: &LateContext<'tcx>, path: &hir::QPath<'_>, hir_id: hir::HirId) -> bool {
    let res = cx.qpath_res(&path, hir_id);
    let Some(def_id) = res.opt_def_id() else { return false };
    let def_kind = cx.tcx.def_kind(def_id);
    match def_kind {
        DefKind::Const
        | DefKind::ConstParam
        | DefKind::AssocConst
        | DefKind::AnonConst
        | DefKind::Ctor(_, CtorKind::Const) => true,

        DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(_, CtorKind::Fn) => {
            cx.tcx.is_const_fn(def_id)
        }

        _ => false,
    }
}

/// Given an expression, determine if it would be suitable for a default field value.
fn is_expr_const(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    match expr.kind {
        hir::ExprKind::Lit(_) => true,
        hir::ExprKind::Call(hir::Expr { kind: hir::ExprKind::Path(path), hir_id, .. }, args) => {
            is_path_const(cx, &path, *hir_id) && args.iter().all(|e| is_expr_const(cx, e))
        }
        hir::ExprKind::Path(path) => is_path_const(cx, &path, expr.hir_id),
        hir::ExprKind::Struct(_, fields, _) => fields.iter().all(|f| is_expr_const(cx, &f.expr)),
        // FIXME: support const traits
        _ => false,
    }
}
