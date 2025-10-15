use clippy_utils::diagnostics::span_lint;
use clippy_utils::paths;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::{AttrStyle, DelimArgs};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{
    AttrArgs, AttrItem, AttrPath, Attribute, HirId, Impl, Item, ItemKind, Path, QPath, TraitImplHeader, TraitRef, Ty,
    TyKind, find_attr,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_middle::ty::TyCtxt;
use rustc_session::declare_lint_pass;

declare_tool_lint! {
    /// ### What it does
    /// Checks for structs or enums that derive `serde::Deserialize` and that
    /// do not have a `#[serde(deny_unknown_fields)]` attribute.
    ///
    /// ### Why is this bad?
    /// If the struct or enum is used in [`clippy_config::conf::Conf`] and a
    /// user inserts an unknown field by mistake, the user's error will be
    /// silently ignored.
    ///
    /// ### Example
    /// ```rust
    /// #[derive(serde::Deserialize)]
    /// pub struct DisallowedPath {
    ///     path: String,
    ///     reason: Option<String>,
    ///     replacement: Option<String>,
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// #[derive(serde::Deserialize)]
    /// #[serde(deny_unknown_fields)]
    /// pub struct DisallowedPath {
    ///     path: String,
    ///     reason: Option<String>,
    ///     replacement: Option<String>,
    /// }
    /// ```
    pub clippy::DERIVE_DESERIALIZE_ALLOWING_UNKNOWN,
    Allow,
    "`#[derive(serde::Deserialize)]` without `#[serde(deny_unknown_fields)]`",
    report_in_external_macro: true
}

declare_lint_pass!(DeriveDeserializeAllowingUnknown => [DERIVE_DESERIALIZE_ALLOWING_UNKNOWN]);

impl<'tcx> LateLintPass<'tcx> for DeriveDeserializeAllowingUnknown {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        // Is this an `impl` (of a certain form)?
        let ItemKind::Impl(Impl {
            of_trait:
                Some(TraitImplHeader {
                    trait_ref:
                        TraitRef {
                            path:
                                Path {
                                    res: Res::Def(_, trait_def_id),
                                    ..
                                },
                            ..
                        },
                    ..
                }),
            self_ty:
                Ty {
                    kind:
                        TyKind::Path(QPath::Resolved(
                            None,
                            Path {
                                res: Res::Def(_, self_ty_def_id),
                                ..
                            },
                        )),
                    ..
                },
            ..
        }) = item.kind
        else {
            return;
        };

        // Is it an `impl` of the trait `serde::Deserialize`?
        if !paths::SERDE_DESERIALIZE.get(cx).contains(trait_def_id) {
            return;
        }

        // Is it derived?
        if !find_attr!(
            cx.tcx.get_all_attrs(item.owner_id),
            AttributeKind::AutomaticallyDerived(..)
        ) {
            return;
        }

        // Is `self_ty` local?
        let Some(local_def_id) = self_ty_def_id.as_local() else {
            return;
        };

        // Does `self_ty` have a variant with named fields?
        if !has_variant_with_named_fields(cx.tcx, local_def_id) {
            return;
        }

        let hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id);

        // Does `self_ty` have `#[serde(deny_unknown_fields)]`?
        if let Some(tokens) = find_serde_attr_item(cx.tcx, hir_id)
            && tokens.iter().any(is_deny_unknown_fields_token)
        {
            return;
        }

        span_lint(
            cx,
            DERIVE_DESERIALIZE_ALLOWING_UNKNOWN,
            item.span,
            "`#[derive(serde::Deserialize)]` without `#[serde(deny_unknown_fields)]`",
        );
    }
}

// Determines whether `def_id` corresponds to an ADT with at least one variant with named fields. A
// variant has named fields if its `ctor` field is `None`.
fn has_variant_with_named_fields(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let ty = tcx.type_of(def_id).skip_binder();

    let rustc_middle::ty::Adt(adt_def, _) = ty.kind() else {
        return false;
    };

    adt_def.variants().iter().any(|variant_def| variant_def.ctor.is_none())
}

fn find_serde_attr_item(tcx: TyCtxt<'_>, hir_id: HirId) -> Option<&TokenStream> {
    tcx.hir_attrs(hir_id).iter().find_map(|attribute| {
        if let Attribute::Unparsed(attr_item) = attribute
            && let AttrItem {
                path: AttrPath { segments, .. },
                args: AttrArgs::Delimited(DelimArgs { tokens, .. }),
                style: AttrStyle::Outer,
                ..
            } = &**attr_item
            && segments.len() == 1
            && segments[0].as_str() == "serde"
        {
            Some(tokens)
        } else {
            None
        }
    })
}

fn is_deny_unknown_fields_token(tt: &TokenTree) -> bool {
    if let TokenTree::Token(token, _) = tt
        && token
            .ident()
            .is_some_and(|(token, _)| token.as_str() == "deny_unknown_fields")
    {
        true
    } else {
        false
    }
}
