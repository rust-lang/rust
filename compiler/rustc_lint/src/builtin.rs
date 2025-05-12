//! Lints in the Rust compiler.
//!
//! This contains lints which can feasibly be implemented as their own
//! AST visitor. Also see `rustc_session::lint::builtin`, which contains the
//! definitions of lints that are emitted directly inside the main compiler.
//!
//! To add a new lint to rustc, declare it here using [`declare_lint!`].
//! Then add code to emit the new lint in the appropriate circumstances.
//!
//! If you define a new [`EarlyLintPass`], you will also need to add it to the
//! [`crate::early_lint_methods!`] invocation in `lib.rs`.
//!
//! If you define a new [`LateLintPass`], you will also need to add it to the
//! [`crate::late_lint_methods!`] invocation in `lib.rs`.

use std::fmt::Write;

use ast::token::TokenKind;
use rustc_abi::BackendRepr;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::visit::{FnCtxt, FnKind};
use rustc_ast::{self as ast, *};
use rustc_ast_pretty::pprust::expr_to_string;
use rustc_errors::{Applicability, LintDiagnostic};
use rustc_feature::{AttributeGate, BuiltinAttribute, GateIssue, Stability, deprecated_attributes};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LocalDefId};
use rustc_hir::intravisit::FnKind as HirFnKind;
use rustc_hir::{Body, FnDecl, GenericParamKind, PatKind, PredicateOrigin};
use rustc_middle::bug;
use rustc_middle::lint::LevelAndSource;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, Upcast, VariantDef};
use rustc_session::lint::FutureIncompatibilityReason;
// hardwired lints from rustc_lint_defs
pub use rustc_session::lint::builtin::*;
use rustc_session::{declare_lint, declare_lint_pass, impl_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::source_map::Spanned;
use rustc_span::{BytePos, Ident, InnerSpan, Span, Symbol, kw, sym};
use rustc_target::asm::InlineAsmArch;
use rustc_trait_selection::infer::{InferCtxtExt, TyCtxtInferExt};
use rustc_trait_selection::traits::misc::type_allowed_to_implement_copy;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{self};

use crate::errors::BuiltinEllipsisInclusiveRangePatterns;
use crate::lints::{
    BuiltinAnonymousParams, BuiltinConstNoMangle, BuiltinDeprecatedAttrLink,
    BuiltinDeprecatedAttrLinkSuggestion, BuiltinDerefNullptr, BuiltinDoubleNegations,
    BuiltinDoubleNegationsAddParens, BuiltinEllipsisInclusiveRangePatternsLint,
    BuiltinExplicitOutlives, BuiltinExplicitOutlivesSuggestion, BuiltinFeatureIssueNote,
    BuiltinIncompleteFeatures, BuiltinIncompleteFeaturesHelp, BuiltinInternalFeatures,
    BuiltinKeywordIdents, BuiltinMissingCopyImpl, BuiltinMissingDebugImpl, BuiltinMissingDoc,
    BuiltinMutablesTransmutes, BuiltinNoMangleGeneric, BuiltinNonShorthandFieldPatterns,
    BuiltinSpecialModuleNameUsed, BuiltinTrivialBounds, BuiltinTypeAliasBounds,
    BuiltinUngatedAsyncFnTrackCaller, BuiltinUnpermittedTypeInit, BuiltinUnpermittedTypeInitSub,
    BuiltinUnreachablePub, BuiltinUnsafe, BuiltinUnstableFeatures, BuiltinUnusedDocComment,
    BuiltinUnusedDocCommentSub, BuiltinWhileTrue, InvalidAsmLabel,
};
use crate::nonstandard_style::{MethodLateContext, method_context};
use crate::{
    EarlyContext, EarlyLintPass, LateContext, LateLintPass, Level, LintContext,
    fluent_generated as fluent,
};
declare_lint! {
    /// The `while_true` lint detects `while true { }`.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// while true {
    ///
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `while true` should be replaced with `loop`. A `loop` expression is
    /// the preferred way to write an infinite loop because it more directly
    /// expresses the intent of the loop.
    WHILE_TRUE,
    Warn,
    "suggest using `loop { }` instead of `while true { }`"
}

declare_lint_pass!(WhileTrue => [WHILE_TRUE]);

impl EarlyLintPass for WhileTrue {
    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let ast::ExprKind::While(cond, _, label) = &e.kind
            && let ast::ExprKind::Lit(token_lit) = cond.peel_parens().kind
            && let token::Lit { kind: token::Bool, symbol: kw::True, .. } = token_lit
            && !cond.span.from_expansion()
        {
            let condition_span = e.span.with_hi(cond.span.hi());
            let replace = format!(
                "{}loop",
                label.map_or_else(String::new, |label| format!("{}: ", label.ident,))
            );
            cx.emit_span_lint(
                WHILE_TRUE,
                condition_span,
                BuiltinWhileTrue { suggestion: condition_span, replace },
            );
        }
    }
}

declare_lint! {
    /// The `non_shorthand_field_patterns` lint detects using `Struct { x: x }`
    /// instead of `Struct { x }` in a pattern.
    ///
    /// ### Example
    ///
    /// ```rust
    /// struct Point {
    ///     x: i32,
    ///     y: i32,
    /// }
    ///
    ///
    /// fn main() {
    ///     let p = Point {
    ///         x: 5,
    ///         y: 5,
    ///     };
    ///
    ///     match p {
    ///         Point { x: x, y: y } => (),
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The preferred style is to avoid the repetition of specifying both the
    /// field name and the binding name if both identifiers are the same.
    NON_SHORTHAND_FIELD_PATTERNS,
    Warn,
    "using `Struct { x: x }` instead of `Struct { x }` in a pattern"
}

declare_lint_pass!(NonShorthandFieldPatterns => [NON_SHORTHAND_FIELD_PATTERNS]);

impl<'tcx> LateLintPass<'tcx> for NonShorthandFieldPatterns {
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &hir::Pat<'_>) {
        if let PatKind::Struct(ref qpath, field_pats, _) = pat.kind {
            let variant = cx
                .typeck_results()
                .pat_ty(pat)
                .ty_adt_def()
                .expect("struct pattern type is not an ADT")
                .variant_of_res(cx.qpath_res(qpath, pat.hir_id));
            for fieldpat in field_pats {
                if fieldpat.is_shorthand {
                    continue;
                }
                if fieldpat.span.from_expansion() {
                    // Don't lint if this is a macro expansion: macro authors
                    // shouldn't have to worry about this kind of style issue
                    // (Issue #49588)
                    continue;
                }
                if let PatKind::Binding(binding_annot, _, ident, None) = fieldpat.pat.kind {
                    if cx.tcx.find_field_index(ident, variant)
                        == Some(cx.typeck_results().field_index(fieldpat.hir_id))
                    {
                        cx.emit_span_lint(
                            NON_SHORTHAND_FIELD_PATTERNS,
                            fieldpat.span,
                            BuiltinNonShorthandFieldPatterns {
                                ident,
                                suggestion: fieldpat.span,
                                prefix: binding_annot.prefix_str(),
                            },
                        );
                    }
                }
            }
        }
    }
}

declare_lint! {
    /// The `unsafe_code` lint catches usage of `unsafe` code and other
    /// potentially unsound constructs like `no_mangle`, `export_name`,
    /// and `link_section`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unsafe_code)]
    /// fn main() {
    ///     unsafe {
    ///
    ///     }
    /// }
    ///
    /// #[no_mangle]
    /// fn func_0() { }
    ///
    /// #[export_name = "exported_symbol_name"]
    /// pub fn name_in_rust() { }
    ///
    /// #[no_mangle]
    /// #[link_section = ".example_section"]
    /// pub static VAR1: u32 = 1;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is intended to restrict the usage of `unsafe` blocks and other
    /// constructs (including, but not limited to `no_mangle`, `link_section`
    /// and `export_name` attributes) wrong usage of which causes undefined
    /// behavior.
    UNSAFE_CODE,
    Allow,
    "usage of `unsafe` code and other potentially unsound constructs",
    @eval_always = true
}

declare_lint_pass!(UnsafeCode => [UNSAFE_CODE]);

impl UnsafeCode {
    fn report_unsafe(
        &self,
        cx: &EarlyContext<'_>,
        span: Span,
        decorate: impl for<'a> LintDiagnostic<'a, ()>,
    ) {
        // This comes from a macro that has `#[allow_internal_unsafe]`.
        if span.allows_unsafe() {
            return;
        }

        cx.emit_span_lint(UNSAFE_CODE, span, decorate);
    }
}

impl EarlyLintPass for UnsafeCode {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &ast::Attribute) {
        if attr.has_name(sym::allow_internal_unsafe) {
            self.report_unsafe(cx, attr.span, BuiltinUnsafe::AllowInternalUnsafe);
        }
    }

    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let ast::ExprKind::Block(ref blk, _) = e.kind {
            // Don't warn about generated blocks; that'll just pollute the output.
            if blk.rules == ast::BlockCheckMode::Unsafe(ast::UserProvided) {
                self.report_unsafe(cx, blk.span, BuiltinUnsafe::UnsafeBlock);
            }
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, it: &ast::Item) {
        match it.kind {
            ast::ItemKind::Trait(box ast::Trait { safety: ast::Safety::Unsafe(_), .. }) => {
                self.report_unsafe(cx, it.span, BuiltinUnsafe::UnsafeTrait);
            }

            ast::ItemKind::Impl(box ast::Impl { safety: ast::Safety::Unsafe(_), .. }) => {
                self.report_unsafe(cx, it.span, BuiltinUnsafe::UnsafeImpl);
            }

            ast::ItemKind::Fn(..) => {
                if let Some(attr) = attr::find_by_name(&it.attrs, sym::no_mangle) {
                    self.report_unsafe(cx, attr.span, BuiltinUnsafe::NoMangleFn);
                }

                if let Some(attr) = attr::find_by_name(&it.attrs, sym::export_name) {
                    self.report_unsafe(cx, attr.span, BuiltinUnsafe::ExportNameFn);
                }

                if let Some(attr) = attr::find_by_name(&it.attrs, sym::link_section) {
                    self.report_unsafe(cx, attr.span, BuiltinUnsafe::LinkSectionFn);
                }
            }

            ast::ItemKind::Static(..) => {
                if let Some(attr) = attr::find_by_name(&it.attrs, sym::no_mangle) {
                    self.report_unsafe(cx, attr.span, BuiltinUnsafe::NoMangleStatic);
                }

                if let Some(attr) = attr::find_by_name(&it.attrs, sym::export_name) {
                    self.report_unsafe(cx, attr.span, BuiltinUnsafe::ExportNameStatic);
                }

                if let Some(attr) = attr::find_by_name(&it.attrs, sym::link_section) {
                    self.report_unsafe(cx, attr.span, BuiltinUnsafe::LinkSectionStatic);
                }
            }

            ast::ItemKind::GlobalAsm(..) => {
                self.report_unsafe(cx, it.span, BuiltinUnsafe::GlobalAsm);
            }

            ast::ItemKind::ForeignMod(ForeignMod { safety, .. }) => {
                if let Safety::Unsafe(_) = safety {
                    self.report_unsafe(cx, it.span, BuiltinUnsafe::UnsafeExternBlock);
                }
            }

            _ => {}
        }
    }

    fn check_impl_item(&mut self, cx: &EarlyContext<'_>, it: &ast::AssocItem) {
        if let ast::AssocItemKind::Fn(..) = it.kind {
            if let Some(attr) = attr::find_by_name(&it.attrs, sym::no_mangle) {
                self.report_unsafe(cx, attr.span, BuiltinUnsafe::NoMangleMethod);
            }
            if let Some(attr) = attr::find_by_name(&it.attrs, sym::export_name) {
                self.report_unsafe(cx, attr.span, BuiltinUnsafe::ExportNameMethod);
            }
        }
    }

    fn check_fn(&mut self, cx: &EarlyContext<'_>, fk: FnKind<'_>, span: Span, _: ast::NodeId) {
        if let FnKind::Fn(
            ctxt,
            _,
            ast::Fn {
                sig: ast::FnSig { header: ast::FnHeader { safety: ast::Safety::Unsafe(_), .. }, .. },
                body,
                ..
            },
        ) = fk
        {
            let decorator = match ctxt {
                FnCtxt::Foreign => return,
                FnCtxt::Free => BuiltinUnsafe::DeclUnsafeFn,
                FnCtxt::Assoc(_) if body.is_none() => BuiltinUnsafe::DeclUnsafeMethod,
                FnCtxt::Assoc(_) => BuiltinUnsafe::ImplUnsafeMethod,
            };
            self.report_unsafe(cx, span, decorator);
        }
    }
}

declare_lint! {
    /// The `missing_docs` lint detects missing documentation for public items.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(missing_docs)]
    /// pub fn foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is intended to ensure that a library is well-documented.
    /// Items without documentation can be difficult for users to understand
    /// how to use properly.
    ///
    /// This lint is "allow" by default because it can be noisy, and not all
    /// projects may want to enforce everything to be documented.
    pub MISSING_DOCS,
    Allow,
    "detects missing documentation for public members",
    report_in_external_macro
}

#[derive(Default)]
pub struct MissingDoc;

impl_lint_pass!(MissingDoc => [MISSING_DOCS]);

fn has_doc(attr: &hir::Attribute) -> bool {
    if attr.is_doc_comment() {
        return true;
    }

    if !attr.has_name(sym::doc) {
        return false;
    }

    if attr.value_str().is_some() {
        return true;
    }

    if let Some(list) = attr.meta_item_list() {
        for meta in list {
            if meta.has_name(sym::hidden) {
                return true;
            }
        }
    }

    false
}

impl MissingDoc {
    fn check_missing_docs_attrs(
        &self,
        cx: &LateContext<'_>,
        def_id: LocalDefId,
        article: &'static str,
        desc: &'static str,
    ) {
        // Only check publicly-visible items, using the result from the privacy pass.
        // It's an option so the crate root can also use this function (it doesn't
        // have a `NodeId`).
        if def_id != CRATE_DEF_ID && !cx.effective_visibilities.is_exported(def_id) {
            return;
        }

        let attrs = cx.tcx.hir_attrs(cx.tcx.local_def_id_to_hir_id(def_id));
        let has_doc = attrs.iter().any(has_doc);
        if !has_doc {
            cx.emit_span_lint(
                MISSING_DOCS,
                cx.tcx.def_span(def_id),
                BuiltinMissingDoc { article, desc },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for MissingDoc {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        self.check_missing_docs_attrs(cx, CRATE_DEF_ID, "the", "crate");
    }

    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        // Previously the Impl and Use types have been excluded from missing docs,
        // so we will continue to exclude them for compatibility.
        //
        // The documentation on `ExternCrate` is not used at the moment so no need to warn for it.
        if let hir::ItemKind::Impl(..) | hir::ItemKind::Use(..) | hir::ItemKind::ExternCrate(..) =
            it.kind
        {
            return;
        }

        let (article, desc) = cx.tcx.article_and_description(it.owner_id.to_def_id());
        self.check_missing_docs_attrs(cx, it.owner_id.def_id, article, desc);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, trait_item: &hir::TraitItem<'_>) {
        let (article, desc) = cx.tcx.article_and_description(trait_item.owner_id.to_def_id());

        self.check_missing_docs_attrs(cx, trait_item.owner_id.def_id, article, desc);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        let context = method_context(cx, impl_item.owner_id.def_id);

        match context {
            // If the method is an impl for a trait, don't doc.
            MethodLateContext::TraitImpl => return,
            MethodLateContext::TraitAutoImpl => {}
            // If the method is an impl for an item with docs_hidden, don't doc.
            MethodLateContext::PlainImpl => {
                let parent = cx.tcx.hir_get_parent_item(impl_item.hir_id());
                let impl_ty = cx.tcx.type_of(parent).instantiate_identity();
                let outerdef = match impl_ty.kind() {
                    ty::Adt(def, _) => Some(def.did()),
                    ty::Foreign(def_id) => Some(*def_id),
                    _ => None,
                };
                let is_hidden = match outerdef {
                    Some(id) => cx.tcx.is_doc_hidden(id),
                    None => false,
                };
                if is_hidden {
                    return;
                }
            }
        }

        let (article, desc) = cx.tcx.article_and_description(impl_item.owner_id.to_def_id());
        self.check_missing_docs_attrs(cx, impl_item.owner_id.def_id, article, desc);
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'_>, foreign_item: &hir::ForeignItem<'_>) {
        let (article, desc) = cx.tcx.article_and_description(foreign_item.owner_id.to_def_id());
        self.check_missing_docs_attrs(cx, foreign_item.owner_id.def_id, article, desc);
    }

    fn check_field_def(&mut self, cx: &LateContext<'_>, sf: &hir::FieldDef<'_>) {
        if !sf.is_positional() {
            self.check_missing_docs_attrs(cx, sf.def_id, "a", "struct field")
        }
    }

    fn check_variant(&mut self, cx: &LateContext<'_>, v: &hir::Variant<'_>) {
        self.check_missing_docs_attrs(cx, v.def_id, "a", "variant");
    }
}

declare_lint! {
    /// The `missing_copy_implementations` lint detects potentially-forgotten
    /// implementations of [`Copy`] for public types.
    ///
    /// [`Copy`]: https://doc.rust-lang.org/std/marker/trait.Copy.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(missing_copy_implementations)]
    /// pub struct Foo {
    ///     pub field: i32
    /// }
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Historically (before 1.0), types were automatically marked as `Copy`
    /// if possible. This was changed so that it required an explicit opt-in
    /// by implementing the `Copy` trait. As part of this change, a lint was
    /// added to alert if a copyable type was not marked `Copy`.
    ///
    /// This lint is "allow" by default because this code isn't bad; it is
    /// common to write newtypes like this specifically so that a `Copy` type
    /// is no longer `Copy`. `Copy` types can result in unintended copies of
    /// large data which can impact performance.
    pub MISSING_COPY_IMPLEMENTATIONS,
    Allow,
    "detects potentially-forgotten implementations of `Copy`"
}

declare_lint_pass!(MissingCopyImplementations => [MISSING_COPY_IMPLEMENTATIONS]);

impl<'tcx> LateLintPass<'tcx> for MissingCopyImplementations {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if !cx.effective_visibilities.is_reachable(item.owner_id.def_id) {
            return;
        }
        let (def, ty) = match item.kind {
            hir::ItemKind::Struct(_, _, ast_generics) => {
                if !ast_generics.params.is_empty() {
                    return;
                }
                let def = cx.tcx.adt_def(item.owner_id);
                (def, Ty::new_adt(cx.tcx, def, ty::List::empty()))
            }
            hir::ItemKind::Union(_, _, ast_generics) => {
                if !ast_generics.params.is_empty() {
                    return;
                }
                let def = cx.tcx.adt_def(item.owner_id);
                (def, Ty::new_adt(cx.tcx, def, ty::List::empty()))
            }
            hir::ItemKind::Enum(_, _, ast_generics) => {
                if !ast_generics.params.is_empty() {
                    return;
                }
                let def = cx.tcx.adt_def(item.owner_id);
                (def, Ty::new_adt(cx.tcx, def, ty::List::empty()))
            }
            _ => return,
        };
        if def.has_dtor(cx.tcx) {
            return;
        }

        // If the type contains a raw pointer, it may represent something like a handle,
        // and recommending Copy might be a bad idea.
        for field in def.all_fields() {
            let did = field.did;
            if cx.tcx.type_of(did).instantiate_identity().is_raw_ptr() {
                return;
            }
        }
        if cx.type_is_copy_modulo_regions(ty) {
            return;
        }
        if type_implements_negative_copy_modulo_regions(cx.tcx, ty, cx.typing_env()) {
            return;
        }
        if def.is_variant_list_non_exhaustive()
            || def.variants().iter().any(|variant| variant.is_field_list_non_exhaustive())
        {
            return;
        }

        // We shouldn't recommend implementing `Copy` on stateful things,
        // such as iterators.
        if let Some(iter_trait) = cx.tcx.get_diagnostic_item(sym::Iterator)
            && cx
                .tcx
                .infer_ctxt()
                .build(cx.typing_mode())
                .type_implements_trait(iter_trait, [ty], cx.param_env)
                .must_apply_modulo_regions()
        {
            return;
        }

        // Default value of clippy::trivially_copy_pass_by_ref
        const MAX_SIZE: u64 = 256;

        if let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes()) {
            if size > MAX_SIZE {
                return;
            }
        }

        if type_allowed_to_implement_copy(
            cx.tcx,
            cx.param_env,
            ty,
            traits::ObligationCause::misc(item.span, item.owner_id.def_id),
            hir::Safety::Safe,
        )
        .is_ok()
        {
            cx.emit_span_lint(MISSING_COPY_IMPLEMENTATIONS, item.span, BuiltinMissingCopyImpl);
        }
    }
}

/// Check whether a `ty` has a negative `Copy` implementation, ignoring outlives constraints.
fn type_implements_negative_copy_modulo_regions<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
) -> bool {
    let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
    let trait_ref = ty::TraitRef::new(tcx, tcx.require_lang_item(hir::LangItem::Copy, None), [ty]);
    let pred = ty::TraitPredicate { trait_ref, polarity: ty::PredicatePolarity::Negative };
    let obligation = traits::Obligation {
        cause: traits::ObligationCause::dummy(),
        param_env,
        recursion_depth: 0,
        predicate: pred.upcast(tcx),
    };
    infcx.predicate_must_hold_modulo_regions(&obligation)
}

declare_lint! {
    /// The `missing_debug_implementations` lint detects missing
    /// implementations of [`fmt::Debug`] for public types.
    ///
    /// [`fmt::Debug`]: https://doc.rust-lang.org/std/fmt/trait.Debug.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(missing_debug_implementations)]
    /// pub struct Foo;
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Having a `Debug` implementation on all types can assist with
    /// debugging, as it provides a convenient way to format and display a
    /// value. Using the `#[derive(Debug)]` attribute will automatically
    /// generate a typical implementation, or a custom implementation can be
    /// added by manually implementing the `Debug` trait.
    ///
    /// This lint is "allow" by default because adding `Debug` to all types can
    /// have a negative impact on compile time and code size. It also requires
    /// boilerplate to be added to every type, which can be an impediment.
    MISSING_DEBUG_IMPLEMENTATIONS,
    Allow,
    "detects missing implementations of Debug"
}

#[derive(Default)]
pub(crate) struct MissingDebugImplementations;

impl_lint_pass!(MissingDebugImplementations => [MISSING_DEBUG_IMPLEMENTATIONS]);

impl<'tcx> LateLintPass<'tcx> for MissingDebugImplementations {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if !cx.effective_visibilities.is_reachable(item.owner_id.def_id) {
            return;
        }

        match item.kind {
            hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) | hir::ItemKind::Enum(..) => {}
            _ => return,
        }

        // Avoid listing trait impls if the trait is allowed.
        let LevelAndSource { level, .. } =
            cx.tcx.lint_level_at_node(MISSING_DEBUG_IMPLEMENTATIONS, item.hir_id());
        if level == Level::Allow {
            return;
        }

        let Some(debug) = cx.tcx.get_diagnostic_item(sym::Debug) else { return };

        let has_impl = cx
            .tcx
            .non_blanket_impls_for_ty(debug, cx.tcx.type_of(item.owner_id).instantiate_identity())
            .next()
            .is_some();
        if !has_impl {
            cx.emit_span_lint(
                MISSING_DEBUG_IMPLEMENTATIONS,
                item.span,
                BuiltinMissingDebugImpl { tcx: cx.tcx, def_id: debug },
            );
        }
    }
}

declare_lint! {
    /// The `anonymous_parameters` lint detects anonymous parameters in trait
    /// definitions.
    ///
    /// ### Example
    ///
    /// ```rust,edition2015,compile_fail
    /// #![deny(anonymous_parameters)]
    /// // edition 2015
    /// pub trait Foo {
    ///     fn foo(usize);
    /// }
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This syntax is mostly a historical accident, and can be worked around
    /// quite easily by adding an `_` pattern or a descriptive identifier:
    ///
    /// ```rust
    /// trait Foo {
    ///     fn foo(_: usize);
    /// }
    /// ```
    ///
    /// This syntax is now a hard error in the 2018 edition. In the 2015
    /// edition, this lint is "warn" by default. This lint
    /// enables the [`cargo fix`] tool with the `--edition` flag to
    /// automatically transition old code from the 2015 edition to 2018. The
    /// tool will run this lint and automatically apply the
    /// suggested fix from the compiler (which is to add `_` to each
    /// parameter). This provides a completely automated way to update old
    /// code for a new edition. See [issue #41686] for more details.
    ///
    /// [issue #41686]: https://github.com/rust-lang/rust/issues/41686
    /// [`cargo fix`]: https://doc.rust-lang.org/cargo/commands/cargo-fix.html
    pub ANONYMOUS_PARAMETERS,
    Warn,
    "detects anonymous parameters",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2018),
        reference: "issue #41686 <https://github.com/rust-lang/rust/issues/41686>",
    };
}

declare_lint_pass!(
    /// Checks for use of anonymous parameters (RFC 1685).
    AnonymousParameters => [ANONYMOUS_PARAMETERS]
);

impl EarlyLintPass for AnonymousParameters {
    fn check_trait_item(&mut self, cx: &EarlyContext<'_>, it: &ast::AssocItem) {
        if cx.sess().edition() != Edition::Edition2015 {
            // This is a hard error in future editions; avoid linting and erroring
            return;
        }
        if let ast::AssocItemKind::Fn(box Fn { ref sig, .. }) = it.kind {
            for arg in sig.decl.inputs.iter() {
                if let ast::PatKind::Missing = arg.pat.kind {
                    let ty_snip = cx.sess().source_map().span_to_snippet(arg.ty.span);

                    let (ty_snip, appl) = if let Ok(ref snip) = ty_snip {
                        (snip.as_str(), Applicability::MachineApplicable)
                    } else {
                        ("<type>", Applicability::HasPlaceholders)
                    };
                    cx.emit_span_lint(
                        ANONYMOUS_PARAMETERS,
                        arg.pat.span,
                        BuiltinAnonymousParams { suggestion: (arg.pat.span, appl), ty_snip },
                    );
                }
            }
        }
    }
}

/// Check for use of attributes which have been deprecated.
#[derive(Clone)]
pub struct DeprecatedAttr {
    // This is not free to compute, so we want to keep it around, rather than
    // compute it for every attribute.
    depr_attrs: Vec<&'static BuiltinAttribute>,
}

impl_lint_pass!(DeprecatedAttr => []);

impl Default for DeprecatedAttr {
    fn default() -> Self {
        DeprecatedAttr { depr_attrs: deprecated_attributes() }
    }
}

impl EarlyLintPass for DeprecatedAttr {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &ast::Attribute) {
        for BuiltinAttribute { name, gate, .. } in &self.depr_attrs {
            if attr.ident().map(|ident| ident.name) == Some(*name) {
                if let &AttributeGate::Gated(
                    Stability::Deprecated(link, suggestion),
                    name,
                    reason,
                    _,
                ) = gate
                {
                    let suggestion = match suggestion {
                        Some(msg) => {
                            BuiltinDeprecatedAttrLinkSuggestion::Msg { suggestion: attr.span, msg }
                        }
                        None => {
                            BuiltinDeprecatedAttrLinkSuggestion::Default { suggestion: attr.span }
                        }
                    };
                    cx.emit_span_lint(
                        DEPRECATED,
                        attr.span,
                        BuiltinDeprecatedAttrLink { name, reason, link, suggestion },
                    );
                }
                return;
            }
        }
    }
}

fn warn_if_doc(cx: &EarlyContext<'_>, node_span: Span, node_kind: &str, attrs: &[ast::Attribute]) {
    use rustc_ast::token::CommentKind;

    let mut attrs = attrs.iter().peekable();

    // Accumulate a single span for sugared doc comments.
    let mut sugared_span: Option<Span> = None;

    while let Some(attr) = attrs.next() {
        let is_doc_comment = attr.is_doc_comment();
        if is_doc_comment {
            sugared_span =
                Some(sugared_span.map_or(attr.span, |span| span.with_hi(attr.span.hi())));
        }

        if attrs.peek().is_some_and(|next_attr| next_attr.is_doc_comment()) {
            continue;
        }

        let span = sugared_span.take().unwrap_or(attr.span);

        if is_doc_comment || attr.has_name(sym::doc) {
            let sub = match attr.kind {
                AttrKind::DocComment(CommentKind::Line, _) | AttrKind::Normal(..) => {
                    BuiltinUnusedDocCommentSub::PlainHelp
                }
                AttrKind::DocComment(CommentKind::Block, _) => {
                    BuiltinUnusedDocCommentSub::BlockHelp
                }
            };
            cx.emit_span_lint(
                UNUSED_DOC_COMMENTS,
                span,
                BuiltinUnusedDocComment { kind: node_kind, label: node_span, sub },
            );
        }
    }
}

impl EarlyLintPass for UnusedDocComment {
    fn check_stmt(&mut self, cx: &EarlyContext<'_>, stmt: &ast::Stmt) {
        let kind = match stmt.kind {
            ast::StmtKind::Let(..) => "statements",
            // Disabled pending discussion in #78306
            ast::StmtKind::Item(..) => return,
            // expressions will be reported by `check_expr`.
            ast::StmtKind::Empty
            | ast::StmtKind::Semi(_)
            | ast::StmtKind::Expr(_)
            | ast::StmtKind::MacCall(_) => return,
        };

        warn_if_doc(cx, stmt.span, kind, stmt.kind.attrs());
    }

    fn check_arm(&mut self, cx: &EarlyContext<'_>, arm: &ast::Arm) {
        if let Some(body) = &arm.body {
            let arm_span = arm.pat.span.with_hi(body.span.hi());
            warn_if_doc(cx, arm_span, "match arms", &arm.attrs);
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &ast::Pat) {
        if let ast::PatKind::Struct(_, _, fields, _) = &pat.kind {
            for field in fields {
                warn_if_doc(cx, field.span, "pattern fields", &field.attrs);
            }
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        warn_if_doc(cx, expr.span, "expressions", &expr.attrs);

        if let ExprKind::Struct(s) = &expr.kind {
            for field in &s.fields {
                warn_if_doc(cx, field.span, "expression fields", &field.attrs);
            }
        }
    }

    fn check_generic_param(&mut self, cx: &EarlyContext<'_>, param: &ast::GenericParam) {
        warn_if_doc(cx, param.ident.span, "generic parameters", &param.attrs);
    }

    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &ast::Block) {
        warn_if_doc(cx, block.span, "blocks", block.attrs());
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        if let ast::ItemKind::ForeignMod(_) = item.kind {
            warn_if_doc(cx, item.span, "extern blocks", &item.attrs);
        }
    }
}

declare_lint! {
    /// The `no_mangle_const_items` lint detects any `const` items with the
    /// [`no_mangle` attribute].
    ///
    /// [`no_mangle` attribute]: https://doc.rust-lang.org/reference/abi.html#the-no_mangle-attribute
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail,edition2021
    /// #[no_mangle]
    /// const FOO: i32 = 5;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Constants do not have their symbols exported, and therefore, this
    /// probably means you meant to use a [`static`], not a [`const`].
    ///
    /// [`static`]: https://doc.rust-lang.org/reference/items/static-items.html
    /// [`const`]: https://doc.rust-lang.org/reference/items/constant-items.html
    NO_MANGLE_CONST_ITEMS,
    Deny,
    "const items will not have their symbols exported"
}

declare_lint! {
    /// The `no_mangle_generic_items` lint detects generic items that must be
    /// mangled.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[unsafe(no_mangle)]
    /// fn foo<T>(t: T) {}
    ///
    /// #[unsafe(export_name = "bar")]
    /// fn bar<T>(t: T) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A function with generics must have its symbol mangled to accommodate
    /// the generic parameter. The [`no_mangle`] and [`export_name`] attributes
    /// have no effect in this situation, and should be removed.
    ///
    /// [`no_mangle`]: https://doc.rust-lang.org/reference/abi.html#the-no_mangle-attribute
    /// [`export_name`]: https://doc.rust-lang.org/reference/abi.html#the-export_name-attribute
    NO_MANGLE_GENERIC_ITEMS,
    Warn,
    "generic items must be mangled"
}

declare_lint_pass!(InvalidNoMangleItems => [NO_MANGLE_CONST_ITEMS, NO_MANGLE_GENERIC_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for InvalidNoMangleItems {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        let attrs = cx.tcx.hir_attrs(it.hir_id());
        let check_no_mangle_on_generic_fn = |attr: &hir::Attribute,
                                             impl_generics: Option<&hir::Generics<'_>>,
                                             generics: &hir::Generics<'_>,
                                             span| {
            for param in
                generics.params.iter().chain(impl_generics.map(|g| g.params).into_iter().flatten())
            {
                match param.kind {
                    GenericParamKind::Lifetime { .. } => {}
                    GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                        cx.emit_span_lint(
                            NO_MANGLE_GENERIC_ITEMS,
                            span,
                            BuiltinNoMangleGeneric { suggestion: attr.span() },
                        );
                        break;
                    }
                }
            }
        };
        match it.kind {
            hir::ItemKind::Fn { generics, .. } => {
                if let Some(attr) = attr::find_by_name(attrs, sym::export_name)
                    .or_else(|| attr::find_by_name(attrs, sym::no_mangle))
                {
                    check_no_mangle_on_generic_fn(attr, None, generics, it.span);
                }
            }
            hir::ItemKind::Const(..) => {
                if attr::contains_name(attrs, sym::no_mangle) {
                    // account for "pub const" (#45562)
                    let start = cx
                        .tcx
                        .sess
                        .source_map()
                        .span_to_snippet(it.span)
                        .map(|snippet| snippet.find("const").unwrap_or(0))
                        .unwrap_or(0) as u32;
                    // `const` is 5 chars
                    let suggestion = it.span.with_hi(BytePos(it.span.lo().0 + start + 5));

                    // Const items do not refer to a particular location in memory, and therefore
                    // don't have anything to attach a symbol to
                    cx.emit_span_lint(
                        NO_MANGLE_CONST_ITEMS,
                        it.span,
                        BuiltinConstNoMangle { suggestion },
                    );
                }
            }
            hir::ItemKind::Impl(hir::Impl { generics, items, .. }) => {
                for it in *items {
                    if let hir::AssocItemKind::Fn { .. } = it.kind {
                        let attrs = cx.tcx.hir_attrs(it.id.hir_id());
                        if let Some(attr) = attr::find_by_name(attrs, sym::export_name)
                            .or_else(|| attr::find_by_name(attrs, sym::no_mangle))
                        {
                            check_no_mangle_on_generic_fn(
                                attr,
                                Some(generics),
                                cx.tcx.hir_get_generics(it.id.owner_id.def_id).unwrap(),
                                it.span,
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

declare_lint! {
    /// The `mutable_transmutes` lint catches transmuting from `&T` to `&mut
    /// T` because it is [undefined behavior].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// unsafe {
    ///     let y = std::mem::transmute::<&i32, &mut i32>(&5);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Certain assumptions are made about aliasing of data, and this transmute
    /// violates those assumptions. Consider using [`UnsafeCell`] instead.
    ///
    /// [`UnsafeCell`]: https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html
    MUTABLE_TRANSMUTES,
    Deny,
    "transmuting &T to &mut T is undefined behavior, even if the reference is unused"
}

declare_lint_pass!(MutableTransmutes => [MUTABLE_TRANSMUTES]);

impl<'tcx> LateLintPass<'tcx> for MutableTransmutes {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        if let Some((&ty::Ref(_, _, from_mutbl), &ty::Ref(_, _, to_mutbl))) =
            get_transmute_from_to(cx, expr).map(|(ty1, ty2)| (ty1.kind(), ty2.kind()))
        {
            if from_mutbl < to_mutbl {
                cx.emit_span_lint(MUTABLE_TRANSMUTES, expr.span, BuiltinMutablesTransmutes);
            }
        }

        fn get_transmute_from_to<'tcx>(
            cx: &LateContext<'tcx>,
            expr: &hir::Expr<'_>,
        ) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
            let def = if let hir::ExprKind::Path(ref qpath) = expr.kind {
                cx.qpath_res(qpath, expr.hir_id)
            } else {
                return None;
            };
            if let Res::Def(DefKind::Fn, did) = def {
                if !def_id_is_transmute(cx, did) {
                    return None;
                }
                let sig = cx.typeck_results().node_type(expr.hir_id).fn_sig(cx.tcx);
                let from = sig.inputs().skip_binder()[0];
                let to = sig.output().skip_binder();
                return Some((from, to));
            }
            None
        }

        fn def_id_is_transmute(cx: &LateContext<'_>, def_id: DefId) -> bool {
            cx.tcx.is_intrinsic(def_id, sym::transmute)
        }
    }
}

declare_lint! {
    /// The `unstable_features` lint detects uses of `#![feature]`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unstable_features)]
    /// #![feature(test)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In larger nightly-based projects which
    ///
    /// * consist of a multitude of crates where a subset of crates has to compile on
    ///   stable either unconditionally or depending on a `cfg` flag to for example
    ///   allow stable users to depend on them,
    /// * don't use nightly for experimental features but for, e.g., unstable options only,
    ///
    /// this lint may come in handy to enforce policies of these kinds.
    UNSTABLE_FEATURES,
    Allow,
    "enabling unstable features"
}

declare_lint_pass!(
    /// Forbids using the `#[feature(...)]` attribute
    UnstableFeatures => [UNSTABLE_FEATURES]
);

impl<'tcx> LateLintPass<'tcx> for UnstableFeatures {
    fn check_attribute(&mut self, cx: &LateContext<'_>, attr: &hir::Attribute) {
        if attr.has_name(sym::feature)
            && let Some(items) = attr.meta_item_list()
        {
            for item in items {
                cx.emit_span_lint(UNSTABLE_FEATURES, item.span(), BuiltinUnstableFeatures);
            }
        }
    }
}

declare_lint! {
    /// The `ungated_async_fn_track_caller` lint warns when the
    /// `#[track_caller]` attribute is used on an async function
    /// without enabling the corresponding unstable feature flag.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #[track_caller]
    /// async fn foo() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The attribute must be used in conjunction with the
    /// [`async_fn_track_caller` feature flag]. Otherwise, the `#[track_caller]`
    /// annotation will function as a no-op.
    ///
    /// [`async_fn_track_caller` feature flag]: https://doc.rust-lang.org/beta/unstable-book/language-features/async-fn-track-caller.html
    UNGATED_ASYNC_FN_TRACK_CALLER,
    Warn,
    "enabling track_caller on an async fn is a no-op unless the async_fn_track_caller feature is enabled"
}

declare_lint_pass!(
    /// Explains corresponding feature flag must be enabled for the `#[track_caller]` attribute to
    /// do anything
    UngatedAsyncFnTrackCaller => [UNGATED_ASYNC_FN_TRACK_CALLER]
);

impl<'tcx> LateLintPass<'tcx> for UngatedAsyncFnTrackCaller {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        fn_kind: HirFnKind<'_>,
        _: &'tcx FnDecl<'_>,
        _: &'tcx Body<'_>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if fn_kind.asyncness().is_async()
            && !cx.tcx.features().async_fn_track_caller()
            // Now, check if the function has the `#[track_caller]` attribute
            && let Some(attr) = cx.tcx.get_attr(def_id, sym::track_caller)
        {
            cx.emit_span_lint(
                UNGATED_ASYNC_FN_TRACK_CALLER,
                attr.span(),
                BuiltinUngatedAsyncFnTrackCaller { label: span, session: &cx.tcx.sess },
            );
        }
    }
}

declare_lint! {
    /// The `unreachable_pub` lint triggers for `pub` items not reachable from other crates - that
    /// means neither directly accessible, nor reexported (with `pub use`), nor leaked through
    /// things like return types (which the [`unnameable_types`] lint can detect if desired).
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unreachable_pub)]
    /// mod foo {
    ///     pub mod bar {
    ///
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `pub` keyword both expresses an intent for an item to be publicly available, and also
    /// signals to the compiler to make the item publicly accessible. The intent can only be
    /// satisfied, however, if all items which contain this item are *also* publicly accessible.
    /// Thus, this lint serves to identify situations where the intent does not match the reality.
    ///
    /// If you wish the item to be accessible elsewhere within the crate, but not outside it, the
    /// `pub(crate)` visibility is recommended to be used instead. This more clearly expresses the
    /// intent that the item is only visible within its own crate.
    ///
    /// This lint is "allow" by default because it will trigger for a large amount of existing Rust code.
    /// Eventually it is desired for this to become warn-by-default.
    ///
    /// [`unnameable_types`]: #unnameable-types
    pub UNREACHABLE_PUB,
    Allow,
    "`pub` items not reachable from crate root"
}

declare_lint_pass!(
    /// Lint for items marked `pub` that aren't reachable from other crates.
    UnreachablePub => [UNREACHABLE_PUB]
);

impl UnreachablePub {
    fn perform_lint(
        &self,
        cx: &LateContext<'_>,
        what: &str,
        def_id: LocalDefId,
        vis_span: Span,
        exportable: bool,
    ) {
        let mut applicability = Applicability::MachineApplicable;
        if cx.tcx.visibility(def_id).is_public() && !cx.effective_visibilities.is_reachable(def_id)
        {
            // prefer suggesting `pub(super)` instead of `pub(crate)` when possible,
            // except when `pub(super) == pub(crate)`
            let new_vis = if let Some(ty::Visibility::Restricted(restricted_did)) =
                cx.effective_visibilities.effective_vis(def_id).map(|effective_vis| {
                    effective_vis.at_level(rustc_middle::middle::privacy::Level::Reachable)
                })
                && let parent_parent = cx
                    .tcx
                    .parent_module_from_def_id(cx.tcx.parent_module_from_def_id(def_id).into())
                && *restricted_did == parent_parent.to_local_def_id()
                && !restricted_did.to_def_id().is_crate_root()
            {
                "pub(super)"
            } else {
                "pub(crate)"
            };

            if vis_span.from_expansion() {
                applicability = Applicability::MaybeIncorrect;
            }
            let def_span = cx.tcx.def_span(def_id);
            cx.emit_span_lint(
                UNREACHABLE_PUB,
                def_span,
                BuiltinUnreachablePub {
                    what,
                    new_vis,
                    suggestion: (vis_span, applicability),
                    help: exportable,
                },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for UnreachablePub {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        // Do not warn for fake `use` statements.
        if let hir::ItemKind::Use(_, hir::UseKind::ListStem) = &item.kind {
            return;
        }
        self.perform_lint(cx, "item", item.owner_id.def_id, item.vis_span, true);
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'_>, foreign_item: &hir::ForeignItem<'tcx>) {
        self.perform_lint(cx, "item", foreign_item.owner_id.def_id, foreign_item.vis_span, true);
    }

    fn check_field_def(&mut self, _cx: &LateContext<'_>, _field: &hir::FieldDef<'_>) {
        // - If an ADT definition is reported then we don't need to check fields
        //   (as it would add unnecessary complexity to the source code, the struct
        //   definition is in the immediate proximity to give the "real" visibility).
        // - If an ADT is not reported because it's not `pub` - we don't need to
        //   check fields.
        // - If an ADT is not reported because it's reachable - we also don't need
        //   to check fields because then they are reachable by construction if they
        //   are pub.
        //
        // Therefore in no case we check the fields.
        //
        // cf. https://github.com/rust-lang/rust/pull/126013#issuecomment-2152839205
        // cf. https://github.com/rust-lang/rust/pull/126040#issuecomment-2152944506
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        // Only lint inherent impl items.
        if cx.tcx.associated_item(impl_item.owner_id).trait_item_def_id.is_none() {
            self.perform_lint(cx, "item", impl_item.owner_id.def_id, impl_item.vis_span, false);
        }
    }
}

declare_lint! {
    /// The `type_alias_bounds` lint detects bounds in type aliases.
    ///
    /// ### Example
    ///
    /// ```rust
    /// type SendVec<T: Send> = Vec<T>;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Trait and lifetime bounds on generic parameters and in where clauses of
    /// type aliases are not checked at usage sites of the type alias. Moreover,
    /// they are not thoroughly checked for correctness at their definition site
    /// either similar to the aliased type.
    ///
    /// This is a known limitation of the type checker that may be lifted in a
    /// future edition. Permitting such bounds in light of this was unintentional.
    ///
    /// While these bounds may have secondary effects such as enabling the use of
    /// "shorthand" associated type paths[^1] and affecting the default trait
    /// object lifetime[^2] of trait object types passed to the type alias, this
    /// should not have been allowed until the aforementioned restrictions of the
    /// type checker have been lifted.
    ///
    /// Using such bounds is highly discouraged as they are actively misleading.
    ///
    /// [^1]: I.e., paths of the form `T::Assoc` where `T` is a type parameter
    /// bounded by trait `Trait` which defines an associated type called `Assoc`
    /// as opposed to a fully qualified path of the form `<T as Trait>::Assoc`.
    /// [^2]: <https://doc.rust-lang.org/reference/lifetime-elision.html#default-trait-object-lifetimes>
    TYPE_ALIAS_BOUNDS,
    Warn,
    "bounds in type aliases are not enforced"
}

declare_lint_pass!(TypeAliasBounds => [TYPE_ALIAS_BOUNDS]);

impl TypeAliasBounds {
    pub(crate) fn affects_object_lifetime_defaults(pred: &hir::WherePredicate<'_>) -> bool {
        // Bounds of the form `T: 'a` with `T` type param affect object lifetime defaults.
        if let hir::WherePredicateKind::BoundPredicate(pred) = pred.kind
            && pred.bounds.iter().any(|bound| matches!(bound, hir::GenericBound::Outlives(_)))
            && pred.bound_generic_params.is_empty() // indeed, even if absent from the RHS
            && pred.bounded_ty.as_generic_param().is_some()
        {
            return true;
        }
        false
    }
}

impl<'tcx> LateLintPass<'tcx> for TypeAliasBounds {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        let hir::ItemKind::TyAlias(_, hir_ty, generics) = item.kind else { return };

        // There must not be a where clause.
        if generics.predicates.is_empty() {
            return;
        }

        // Bounds of lazy type aliases and TAITs are respected.
        if cx.tcx.type_alias_is_lazy(item.owner_id) {
            return;
        }

        // FIXME(generic_const_exprs): Revisit this before stabilization.
        // See also `tests/ui/const-generics/generic_const_exprs/type-alias-bounds.rs`.
        let ty = cx.tcx.type_of(item.owner_id).instantiate_identity();
        if ty.has_type_flags(ty::TypeFlags::HAS_CT_PROJECTION)
            && cx.tcx.features().generic_const_exprs()
        {
            return;
        }

        // NOTE(inherent_associated_types): While we currently do take some bounds in type
        // aliases into consideration during IAT *selection*, we don't perform full use+def
        // site wfchecking for such type aliases. Therefore TAB should still trigger.
        // See also `tests/ui/associated-inherent-types/type-alias-bounds.rs`.

        let mut where_spans = Vec::new();
        let mut inline_spans = Vec::new();
        let mut inline_sugg = Vec::new();

        for p in generics.predicates {
            let span = p.span;
            if p.kind.in_where_clause() {
                where_spans.push(span);
            } else {
                for b in p.kind.bounds() {
                    inline_spans.push(b.span());
                }
                inline_sugg.push((span, String::new()));
            }
        }

        let mut ty = Some(hir_ty);
        let enable_feat_help = cx.tcx.sess.is_nightly_build();

        if let [.., label_sp] = *where_spans {
            cx.emit_span_lint(
                TYPE_ALIAS_BOUNDS,
                where_spans,
                BuiltinTypeAliasBounds {
                    in_where_clause: true,
                    label: label_sp,
                    enable_feat_help,
                    suggestions: vec![(generics.where_clause_span, String::new())],
                    preds: generics.predicates,
                    ty: ty.take(),
                },
            );
        }
        if let [.., label_sp] = *inline_spans {
            cx.emit_span_lint(
                TYPE_ALIAS_BOUNDS,
                inline_spans,
                BuiltinTypeAliasBounds {
                    in_where_clause: false,
                    label: label_sp,
                    enable_feat_help,
                    suggestions: inline_sugg,
                    preds: generics.predicates,
                    ty,
                },
            );
        }
    }
}

pub(crate) struct ShorthandAssocTyCollector {
    pub(crate) qselves: Vec<Span>,
}

impl hir::intravisit::Visitor<'_> for ShorthandAssocTyCollector {
    fn visit_qpath(&mut self, qpath: &hir::QPath<'_>, id: hir::HirId, _: Span) {
        // Look for "type-parameter shorthand-associated-types". I.e., paths of the
        // form `T::Assoc` with `T` type param. These are reliant on trait bounds.
        if let hir::QPath::TypeRelative(qself, _) = qpath
            && qself.as_generic_param().is_some()
        {
            self.qselves.push(qself.span);
        }
        hir::intravisit::walk_qpath(self, qpath, id)
    }
}

declare_lint! {
    /// The `trivial_bounds` lint detects trait bounds that don't depend on
    /// any type parameters.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(trivial_bounds)]
    /// pub struct A where i32: Copy;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Usually you would not write a trait bound that you know is always
    /// true, or never true. However, when using macros, the macro may not
    /// know whether or not the constraint would hold or not at the time when
    /// generating the code. Currently, the compiler does not alert you if the
    /// constraint is always true, and generates an error if it is never true.
    /// The `trivial_bounds` feature changes this to be a warning in both
    /// cases, giving macros more freedom and flexibility to generate code,
    /// while still providing a signal when writing non-macro code that
    /// something is amiss.
    ///
    /// See [RFC 2056] for more details. This feature is currently only
    /// available on the nightly channel, see [tracking issue #48214].
    ///
    /// [RFC 2056]: https://github.com/rust-lang/rfcs/blob/master/text/2056-allow-trivial-where-clause-constraints.md
    /// [tracking issue #48214]: https://github.com/rust-lang/rust/issues/48214
    TRIVIAL_BOUNDS,
    Warn,
    "these bounds don't depend on an type parameters"
}

declare_lint_pass!(
    /// Lint for trait and lifetime bounds that don't depend on type parameters
    /// which either do nothing, or stop the item from being used.
    TrivialConstraints => [TRIVIAL_BOUNDS]
);

impl<'tcx> LateLintPass<'tcx> for TrivialConstraints {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        use rustc_middle::ty::ClauseKind;

        if cx.tcx.features().trivial_bounds() {
            let predicates = cx.tcx.predicates_of(item.owner_id);
            for &(predicate, span) in predicates.predicates {
                let predicate_kind_name = match predicate.kind().skip_binder() {
                    ClauseKind::Trait(..) => "trait",
                    ClauseKind::TypeOutlives(..) |
                    ClauseKind::RegionOutlives(..) => "lifetime",

                    // `ConstArgHasType` is never global as `ct` is always a param
                    ClauseKind::ConstArgHasType(..)
                    // Ignore projections, as they can only be global
                    // if the trait bound is global
                    | ClauseKind::Projection(..)
                    // Ignore bounds that a user can't type
                    | ClauseKind::WellFormed(..)
                    // FIXME(generic_const_exprs): `ConstEvaluatable` can be written
                    | ClauseKind::ConstEvaluatable(..)
                    // Users don't write this directly, only via another trait ref.
                    | ty::ClauseKind::HostEffect(..) => continue,
                };
                if predicate.is_global() {
                    cx.emit_span_lint(
                        TRIVIAL_BOUNDS,
                        span,
                        BuiltinTrivialBounds { predicate_kind_name, predicate },
                    );
                }
            }
        }
    }
}

declare_lint! {
    /// The `double_negations` lint detects expressions of the form `--x`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn main() {
    ///     let x = 1;
    ///     let _b = --x;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Negating something twice is usually the same as not negating it at all.
    /// However, a double negation in Rust can easily be confused with the
    /// prefix decrement operator that exists in many languages derived from C.
    /// Use `-(-x)` if you really wanted to negate the value twice.
    ///
    /// To decrement a value, use `x -= 1` instead.
    pub DOUBLE_NEGATIONS,
    Warn,
    "detects expressions of the form `--x`"
}

declare_lint_pass!(
    /// Lint for expressions of the form `--x` that can be confused with C's
    /// prefix decrement operator.
    DoubleNegations => [DOUBLE_NEGATIONS]
);

impl EarlyLintPass for DoubleNegations {
    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        // only lint on the innermost `--` in a chain of `-` operators,
        // even if there are 3 or more negations
        if let ExprKind::Unary(UnOp::Neg, ref inner) = expr.kind
            && let ExprKind::Unary(UnOp::Neg, ref inner2) = inner.kind
            && !matches!(inner2.kind, ExprKind::Unary(UnOp::Neg, _))
        {
            cx.emit_span_lint(
                DOUBLE_NEGATIONS,
                expr.span,
                BuiltinDoubleNegations {
                    add_parens: BuiltinDoubleNegationsAddParens {
                        start_span: inner.span.shrink_to_lo(),
                        end_span: inner.span.shrink_to_hi(),
                    },
                },
            );
        }
    }
}

declare_lint_pass!(
    /// Does nothing as a lint pass, but registers some `Lint`s
    /// which are used by other parts of the compiler.
    SoftLints => [
        WHILE_TRUE,
        NON_SHORTHAND_FIELD_PATTERNS,
        UNSAFE_CODE,
        MISSING_DOCS,
        MISSING_COPY_IMPLEMENTATIONS,
        MISSING_DEBUG_IMPLEMENTATIONS,
        ANONYMOUS_PARAMETERS,
        UNUSED_DOC_COMMENTS,
        NO_MANGLE_CONST_ITEMS,
        NO_MANGLE_GENERIC_ITEMS,
        MUTABLE_TRANSMUTES,
        UNSTABLE_FEATURES,
        UNREACHABLE_PUB,
        TYPE_ALIAS_BOUNDS,
        TRIVIAL_BOUNDS,
        DOUBLE_NEGATIONS
    ]
);

declare_lint! {
    /// The `ellipsis_inclusive_range_patterns` lint detects the [`...` range
    /// pattern], which is deprecated.
    ///
    /// [`...` range pattern]: https://doc.rust-lang.org/reference/patterns.html#range-patterns
    ///
    /// ### Example
    ///
    /// ```rust,edition2018
    /// let x = 123;
    /// match x {
    ///     0...100 => {}
    ///     _ => {}
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `...` range pattern syntax was changed to `..=` to avoid potential
    /// confusion with the [`..` range expression]. Use the new form instead.
    ///
    /// [`..` range expression]: https://doc.rust-lang.org/reference/expressions/range-expr.html
    pub ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,
    Warn,
    "`...` range patterns are deprecated",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2021),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/warnings-promoted-to-error.html>",
    };
}

#[derive(Default)]
pub struct EllipsisInclusiveRangePatterns {
    /// If `Some(_)`, suppress all subsequent pattern
    /// warnings for better diagnostics.
    node_id: Option<ast::NodeId>,
}

impl_lint_pass!(EllipsisInclusiveRangePatterns => [ELLIPSIS_INCLUSIVE_RANGE_PATTERNS]);

impl EarlyLintPass for EllipsisInclusiveRangePatterns {
    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &ast::Pat) {
        if self.node_id.is_some() {
            // Don't recursively warn about patterns inside range endpoints.
            return;
        }

        use self::ast::PatKind;
        use self::ast::RangeSyntax::DotDotDot;

        /// If `pat` is a `...` pattern, return the start and end of the range, as well as the span
        /// corresponding to the ellipsis.
        fn matches_ellipsis_pat(pat: &ast::Pat) -> Option<(Option<&Expr>, &Expr, Span)> {
            match &pat.kind {
                PatKind::Range(
                    a,
                    Some(b),
                    Spanned { span, node: RangeEnd::Included(DotDotDot) },
                ) => Some((a.as_deref(), b, *span)),
                _ => None,
            }
        }

        let (parentheses, endpoints) = match &pat.kind {
            PatKind::Ref(subpat, _) => (true, matches_ellipsis_pat(subpat)),
            _ => (false, matches_ellipsis_pat(pat)),
        };

        if let Some((start, end, join)) = endpoints {
            if parentheses {
                self.node_id = Some(pat.id);
                let end = expr_to_string(end);
                let replace = match start {
                    Some(start) => format!("&({}..={})", expr_to_string(start), end),
                    None => format!("&(..={end})"),
                };
                if join.edition() >= Edition::Edition2021 {
                    cx.sess().dcx().emit_err(BuiltinEllipsisInclusiveRangePatterns {
                        span: pat.span,
                        suggestion: pat.span,
                        replace,
                    });
                } else {
                    cx.emit_span_lint(
                        ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,
                        pat.span,
                        BuiltinEllipsisInclusiveRangePatternsLint::Parenthesise {
                            suggestion: pat.span,
                            replace,
                        },
                    );
                }
            } else {
                let replace = "..=";
                if join.edition() >= Edition::Edition2021 {
                    cx.sess().dcx().emit_err(BuiltinEllipsisInclusiveRangePatterns {
                        span: pat.span,
                        suggestion: join,
                        replace: replace.to_string(),
                    });
                } else {
                    cx.emit_span_lint(
                        ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,
                        join,
                        BuiltinEllipsisInclusiveRangePatternsLint::NonParenthesise {
                            suggestion: join,
                        },
                    );
                }
            };
        }
    }

    fn check_pat_post(&mut self, _cx: &EarlyContext<'_>, pat: &ast::Pat) {
        if let Some(node_id) = self.node_id {
            if pat.id == node_id {
                self.node_id = None
            }
        }
    }
}

declare_lint! {
    /// The `keyword_idents_2018` lint detects edition keywords being used as an
    /// identifier.
    ///
    /// ### Example
    ///
    /// ```rust,edition2015,compile_fail
    /// #![deny(keyword_idents_2018)]
    /// // edition 2015
    /// fn dyn() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust [editions] allow the language to evolve without breaking
    /// backwards compatibility. This lint catches code that uses new keywords
    /// that are added to the language that are used as identifiers (such as a
    /// variable name, function name, etc.). If you switch the compiler to a
    /// new edition without updating the code, then it will fail to compile if
    /// you are using a new keyword as an identifier.
    ///
    /// You can manually change the identifiers to a non-keyword, or use a
    /// [raw identifier], for example `r#dyn`, to transition to a new edition.
    ///
    /// This lint solves the problem automatically. It is "allow" by default
    /// because the code is perfectly valid in older editions. The [`cargo
    /// fix`] tool with the `--edition` flag will switch this lint to "warn"
    /// and automatically apply the suggested fix from the compiler (which is
    /// to use a raw identifier). This provides a completely automated way to
    /// update old code for a new edition.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    /// [raw identifier]: https://doc.rust-lang.org/reference/identifiers.html
    /// [`cargo fix`]: https://doc.rust-lang.org/cargo/commands/cargo-fix.html
    pub KEYWORD_IDENTS_2018,
    Allow,
    "detects edition keywords being used as an identifier",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2018),
        reference: "issue #49716 <https://github.com/rust-lang/rust/issues/49716>",
    };
}

declare_lint! {
    /// The `keyword_idents_2024` lint detects edition keywords being used as an
    /// identifier.
    ///
    /// ### Example
    ///
    /// ```rust,edition2015,compile_fail
    /// #![deny(keyword_idents_2024)]
    /// // edition 2015
    /// fn gen() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust [editions] allow the language to evolve without breaking
    /// backwards compatibility. This lint catches code that uses new keywords
    /// that are added to the language that are used as identifiers (such as a
    /// variable name, function name, etc.). If you switch the compiler to a
    /// new edition without updating the code, then it will fail to compile if
    /// you are using a new keyword as an identifier.
    ///
    /// You can manually change the identifiers to a non-keyword, or use a
    /// [raw identifier], for example `r#gen`, to transition to a new edition.
    ///
    /// This lint solves the problem automatically. It is "allow" by default
    /// because the code is perfectly valid in older editions. The [`cargo
    /// fix`] tool with the `--edition` flag will switch this lint to "warn"
    /// and automatically apply the suggested fix from the compiler (which is
    /// to use a raw identifier). This provides a completely automated way to
    /// update old code for a new edition.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    /// [raw identifier]: https://doc.rust-lang.org/reference/identifiers.html
    /// [`cargo fix`]: https://doc.rust-lang.org/cargo/commands/cargo-fix.html
    pub KEYWORD_IDENTS_2024,
    Allow,
    "detects edition keywords being used as an identifier",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/gen-keyword.html>",
    };
}

declare_lint_pass!(
    /// Check for uses of edition keywords used as an identifier.
    KeywordIdents => [KEYWORD_IDENTS_2018, KEYWORD_IDENTS_2024]
);

struct UnderMacro(bool);

impl KeywordIdents {
    fn check_tokens(&mut self, cx: &EarlyContext<'_>, tokens: &TokenStream) {
        // Check if the preceding token is `$`, because we want to allow `$async`, etc.
        let mut prev_dollar = false;
        for tt in tokens.iter() {
            match tt {
                // Only report non-raw idents.
                TokenTree::Token(token, _) => {
                    if let Some((ident, token::IdentIsRaw::No)) = token.ident() {
                        if !prev_dollar {
                            self.check_ident_token(cx, UnderMacro(true), ident, "");
                        }
                    } else if let Some((ident, token::IdentIsRaw::No)) = token.lifetime() {
                        self.check_ident_token(
                            cx,
                            UnderMacro(true),
                            ident.without_first_quote(),
                            "'",
                        );
                    } else if token.kind == TokenKind::Dollar {
                        prev_dollar = true;
                        continue;
                    }
                }
                TokenTree::Delimited(.., tts) => self.check_tokens(cx, tts),
            }
            prev_dollar = false;
        }
    }

    fn check_ident_token(
        &mut self,
        cx: &EarlyContext<'_>,
        UnderMacro(under_macro): UnderMacro,
        ident: Ident,
        prefix: &'static str,
    ) {
        let (lint, edition) = match ident.name {
            kw::Async | kw::Await | kw::Try => (KEYWORD_IDENTS_2018, Edition::Edition2018),

            // rust-lang/rust#56327: Conservatively do not
            // attempt to report occurrences of `dyn` within
            // macro definitions or invocations, because `dyn`
            // can legitimately occur as a contextual keyword
            // in 2015 code denoting its 2018 meaning, and we
            // do not want rustfix to inject bugs into working
            // code by rewriting such occurrences.
            //
            // But if we see `dyn` outside of a macro, we know
            // its precise role in the parsed AST and thus are
            // assured this is truly an attempt to use it as
            // an identifier.
            kw::Dyn if !under_macro => (KEYWORD_IDENTS_2018, Edition::Edition2018),

            kw::Gen => (KEYWORD_IDENTS_2024, Edition::Edition2024),

            _ => return,
        };

        // Don't lint `r#foo`.
        if ident.span.edition() >= edition
            || cx.sess().psess.raw_identifier_spans.contains(ident.span)
        {
            return;
        }

        cx.emit_span_lint(
            lint,
            ident.span,
            BuiltinKeywordIdents { kw: ident, next: edition, suggestion: ident.span, prefix },
        );
    }
}

impl EarlyLintPass for KeywordIdents {
    fn check_mac_def(&mut self, cx: &EarlyContext<'_>, mac_def: &ast::MacroDef) {
        self.check_tokens(cx, &mac_def.body.tokens);
    }
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &ast::MacCall) {
        self.check_tokens(cx, &mac.args.tokens);
    }
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: &Ident) {
        if ident.name.as_str().starts_with('\'') {
            self.check_ident_token(cx, UnderMacro(false), ident.without_first_quote(), "'");
        } else {
            self.check_ident_token(cx, UnderMacro(false), *ident, "");
        }
    }
}

declare_lint_pass!(ExplicitOutlivesRequirements => [EXPLICIT_OUTLIVES_REQUIREMENTS]);

impl ExplicitOutlivesRequirements {
    fn lifetimes_outliving_lifetime<'tcx>(
        tcx: TyCtxt<'tcx>,
        inferred_outlives: impl Iterator<Item = &'tcx (ty::Clause<'tcx>, Span)>,
        item: LocalDefId,
        lifetime: LocalDefId,
    ) -> Vec<ty::Region<'tcx>> {
        let item_generics = tcx.generics_of(item);

        inferred_outlives
            .filter_map(|(clause, _)| match clause.kind().skip_binder() {
                ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(a, b)) => match a.kind() {
                    ty::ReEarlyParam(ebr)
                        if item_generics.region_param(ebr, tcx).def_id == lifetime.to_def_id() =>
                    {
                        Some(b)
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    fn lifetimes_outliving_type<'tcx>(
        inferred_outlives: impl Iterator<Item = &'tcx (ty::Clause<'tcx>, Span)>,
        index: u32,
    ) -> Vec<ty::Region<'tcx>> {
        inferred_outlives
            .filter_map(|(clause, _)| match clause.kind().skip_binder() {
                ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(a, b)) => {
                    a.is_param(index).then_some(b)
                }
                _ => None,
            })
            .collect()
    }

    fn collect_outlives_bound_spans<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        bounds: &hir::GenericBounds<'_>,
        inferred_outlives: &[ty::Region<'tcx>],
        predicate_span: Span,
        item: DefId,
    ) -> Vec<(usize, Span)> {
        use rustc_middle::middle::resolve_bound_vars::ResolvedArg;

        let item_generics = tcx.generics_of(item);

        bounds
            .iter()
            .enumerate()
            .filter_map(|(i, bound)| {
                let hir::GenericBound::Outlives(lifetime) = bound else {
                    return None;
                };

                let is_inferred = match tcx.named_bound_var(lifetime.hir_id) {
                    Some(ResolvedArg::EarlyBound(def_id)) => inferred_outlives
                        .iter()
                        .any(|r| matches!(r.kind(), ty::ReEarlyParam(ebr) if { item_generics.region_param(ebr, tcx).def_id == def_id.to_def_id() })),
                    _ => false,
                };

                if !is_inferred {
                    return None;
                }

                let span = bound.span().find_ancestor_inside(predicate_span)?;
                if span.in_external_macro(tcx.sess.source_map()) {
                    return None;
                }

                Some((i, span))
            })
            .collect()
    }

    fn consolidate_outlives_bound_spans(
        &self,
        lo: Span,
        bounds: &hir::GenericBounds<'_>,
        bound_spans: Vec<(usize, Span)>,
    ) -> Vec<Span> {
        if bounds.is_empty() {
            return Vec::new();
        }
        if bound_spans.len() == bounds.len() {
            let (_, last_bound_span) = bound_spans[bound_spans.len() - 1];
            // If all bounds are inferable, we want to delete the colon, so
            // start from just after the parameter (span passed as argument)
            vec![lo.to(last_bound_span)]
        } else {
            let mut merged = Vec::new();
            let mut last_merged_i = None;

            let mut from_start = true;
            for (i, bound_span) in bound_spans {
                match last_merged_i {
                    // If the first bound is inferable, our span should also eat the leading `+`.
                    None if i == 0 => {
                        merged.push(bound_span.to(bounds[1].span().shrink_to_lo()));
                        last_merged_i = Some(0);
                    }
                    // If consecutive bounds are inferable, merge their spans
                    Some(h) if i == h + 1 => {
                        if let Some(tail) = merged.last_mut() {
                            // Also eat the trailing `+` if the first
                            // more-than-one bound is inferable
                            let to_span = if from_start && i < bounds.len() {
                                bounds[i + 1].span().shrink_to_lo()
                            } else {
                                bound_span
                            };
                            *tail = tail.to(to_span);
                            last_merged_i = Some(i);
                        } else {
                            bug!("another bound-span visited earlier");
                        }
                    }
                    _ => {
                        // When we find a non-inferable bound, subsequent inferable bounds
                        // won't be consecutive from the start (and we'll eat the leading
                        // `+` rather than the trailing one)
                        from_start = false;
                        merged.push(bounds[i - 1].span().shrink_to_hi().to(bound_span));
                        last_merged_i = Some(i);
                    }
                }
            }
            merged
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ExplicitOutlivesRequirements {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        use rustc_middle::middle::resolve_bound_vars::ResolvedArg;

        let def_id = item.owner_id.def_id;
        if let hir::ItemKind::Struct(_, _, hir_generics)
        | hir::ItemKind::Enum(_, _, hir_generics)
        | hir::ItemKind::Union(_, _, hir_generics) = item.kind
        {
            let inferred_outlives = cx.tcx.inferred_outlives_of(def_id);
            if inferred_outlives.is_empty() {
                return;
            }

            let ty_generics = cx.tcx.generics_of(def_id);
            let num_where_predicates = hir_generics
                .predicates
                .iter()
                .filter(|predicate| predicate.kind.in_where_clause())
                .count();

            let mut bound_count = 0;
            let mut lint_spans = Vec::new();
            let mut where_lint_spans = Vec::new();
            let mut dropped_where_predicate_count = 0;
            for (i, where_predicate) in hir_generics.predicates.iter().enumerate() {
                let (relevant_lifetimes, bounds, predicate_span, in_where_clause) =
                    match where_predicate.kind {
                        hir::WherePredicateKind::RegionPredicate(predicate) => {
                            if let Some(ResolvedArg::EarlyBound(region_def_id)) =
                                cx.tcx.named_bound_var(predicate.lifetime.hir_id)
                            {
                                (
                                    Self::lifetimes_outliving_lifetime(
                                        cx.tcx,
                                        // don't warn if the inferred span actually came from the predicate we're looking at
                                        // this happens if the type is recursively defined
                                        inferred_outlives.iter().filter(|(_, span)| {
                                            !where_predicate.span.contains(*span)
                                        }),
                                        item.owner_id.def_id,
                                        region_def_id,
                                    ),
                                    &predicate.bounds,
                                    where_predicate.span,
                                    predicate.in_where_clause,
                                )
                            } else {
                                continue;
                            }
                        }
                        hir::WherePredicateKind::BoundPredicate(predicate) => {
                            // FIXME we can also infer bounds on associated types,
                            // and should check for them here.
                            match predicate.bounded_ty.kind {
                                hir::TyKind::Path(hir::QPath::Resolved(None, path)) => {
                                    let Res::Def(DefKind::TyParam, def_id) = path.res else {
                                        continue;
                                    };
                                    let index = ty_generics.param_def_id_to_index[&def_id];
                                    (
                                        Self::lifetimes_outliving_type(
                                            // don't warn if the inferred span actually came from the predicate we're looking at
                                            // this happens if the type is recursively defined
                                            inferred_outlives.iter().filter(|(_, span)| {
                                                !where_predicate.span.contains(*span)
                                            }),
                                            index,
                                        ),
                                        &predicate.bounds,
                                        where_predicate.span,
                                        predicate.origin == PredicateOrigin::WhereClause,
                                    )
                                }
                                _ => {
                                    continue;
                                }
                            }
                        }
                        _ => continue,
                    };
                if relevant_lifetimes.is_empty() {
                    continue;
                }

                let bound_spans = self.collect_outlives_bound_spans(
                    cx.tcx,
                    bounds,
                    &relevant_lifetimes,
                    predicate_span,
                    item.owner_id.to_def_id(),
                );
                bound_count += bound_spans.len();

                let drop_predicate = bound_spans.len() == bounds.len();
                if drop_predicate && in_where_clause {
                    dropped_where_predicate_count += 1;
                }

                if drop_predicate {
                    if !in_where_clause {
                        lint_spans.push(predicate_span);
                    } else if predicate_span.from_expansion() {
                        // Don't try to extend the span if it comes from a macro expansion.
                        where_lint_spans.push(predicate_span);
                    } else if i + 1 < num_where_predicates {
                        // If all the bounds on a predicate were inferable and there are
                        // further predicates, we want to eat the trailing comma.
                        let next_predicate_span = hir_generics.predicates[i + 1].span;
                        if next_predicate_span.from_expansion() {
                            where_lint_spans.push(predicate_span);
                        } else {
                            where_lint_spans
                                .push(predicate_span.to(next_predicate_span.shrink_to_lo()));
                        }
                    } else {
                        // Eat the optional trailing comma after the last predicate.
                        let where_span = hir_generics.where_clause_span;
                        if where_span.from_expansion() {
                            where_lint_spans.push(predicate_span);
                        } else {
                            where_lint_spans.push(predicate_span.to(where_span.shrink_to_hi()));
                        }
                    }
                } else {
                    where_lint_spans.extend(self.consolidate_outlives_bound_spans(
                        predicate_span.shrink_to_lo(),
                        bounds,
                        bound_spans,
                    ));
                }
            }

            // If all predicates in where clause are inferable, drop the entire clause
            // (including the `where`)
            if hir_generics.has_where_clause_predicates
                && dropped_where_predicate_count == num_where_predicates
            {
                let where_span = hir_generics.where_clause_span;
                // Extend the where clause back to the closing `>` of the
                // generics, except for tuple struct, which have the `where`
                // after the fields of the struct.
                let full_where_span =
                    if let hir::ItemKind::Struct(_, hir::VariantData::Tuple(..), _) = item.kind {
                        where_span
                    } else {
                        hir_generics.span.shrink_to_hi().to(where_span)
                    };

                // Due to macro expansions, the `full_where_span` might not actually contain all
                // predicates.
                if where_lint_spans.iter().all(|&sp| full_where_span.contains(sp)) {
                    lint_spans.push(full_where_span);
                } else {
                    lint_spans.extend(where_lint_spans);
                }
            } else {
                lint_spans.extend(where_lint_spans);
            }

            if !lint_spans.is_empty() {
                // Do not automatically delete outlives requirements from macros.
                let applicability = if lint_spans.iter().all(|sp| sp.can_be_used_for_suggestions())
                {
                    Applicability::MachineApplicable
                } else {
                    Applicability::MaybeIncorrect
                };

                // Due to macros, there might be several predicates with the same span
                // and we only want to suggest removing them once.
                lint_spans.sort_unstable();
                lint_spans.dedup();

                cx.emit_span_lint(
                    EXPLICIT_OUTLIVES_REQUIREMENTS,
                    lint_spans.clone(),
                    BuiltinExplicitOutlives {
                        count: bound_count,
                        suggestion: BuiltinExplicitOutlivesSuggestion {
                            spans: lint_spans,
                            applicability,
                        },
                    },
                );
            }
        }
    }
}

declare_lint! {
    /// The `incomplete_features` lint detects unstable features enabled with
    /// the [`feature` attribute] that may function improperly in some or all
    /// cases.
    ///
    /// [`feature` attribute]: https://doc.rust-lang.org/nightly/unstable-book/
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Although it is encouraged for people to experiment with unstable
    /// features, some of them are known to be incomplete or faulty. This lint
    /// is a signal that the feature has not yet been finished, and you may
    /// experience problems with it.
    pub INCOMPLETE_FEATURES,
    Warn,
    "incomplete features that may function improperly in some or all cases"
}

declare_lint! {
    /// The `internal_features` lint detects unstable features enabled with
    /// the [`feature` attribute] that are internal to the compiler or standard
    /// library.
    ///
    /// [`feature` attribute]: https://doc.rust-lang.org/nightly/unstable-book/
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(rustc_attrs)]
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// These features are an implementation detail of the compiler and standard
    /// library and are not supposed to be used in user code.
    pub INTERNAL_FEATURES,
    Warn,
    "internal features are not supposed to be used"
}

declare_lint_pass!(
    /// Check for used feature gates in `INCOMPLETE_FEATURES` in `rustc_feature/src/unstable.rs`.
    IncompleteInternalFeatures => [INCOMPLETE_FEATURES, INTERNAL_FEATURES]
);

impl EarlyLintPass for IncompleteInternalFeatures {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &ast::Crate) {
        let features = cx.builder.features();
        let lang_features =
            features.enabled_lang_features().iter().map(|feat| (feat.gate_name, feat.attr_sp));
        let lib_features =
            features.enabled_lib_features().iter().map(|feat| (feat.gate_name, feat.attr_sp));

        lang_features
            .chain(lib_features)
            .filter(|(name, _)| features.incomplete(*name) || features.internal(*name))
            .for_each(|(name, span)| {
                if features.incomplete(name) {
                    let note = rustc_feature::find_feature_issue(name, GateIssue::Language)
                        .map(|n| BuiltinFeatureIssueNote { n });
                    let help =
                        HAS_MIN_FEATURES.contains(&name).then_some(BuiltinIncompleteFeaturesHelp);

                    cx.emit_span_lint(
                        INCOMPLETE_FEATURES,
                        span,
                        BuiltinIncompleteFeatures { name, note, help },
                    );
                } else {
                    cx.emit_span_lint(INTERNAL_FEATURES, span, BuiltinInternalFeatures { name });
                }
            });
    }
}

const HAS_MIN_FEATURES: &[Symbol] = &[sym::specialization];

declare_lint! {
    /// The `invalid_value` lint detects creating a value that is not valid,
    /// such as a null reference.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// # #![allow(unused)]
    /// unsafe {
    ///     let x: &'static i32 = std::mem::zeroed();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In some situations the compiler can detect that the code is creating
    /// an invalid value, which should be avoided.
    ///
    /// In particular, this lint will check for improper use of
    /// [`mem::zeroed`], [`mem::uninitialized`], [`mem::transmute`], and
    /// [`MaybeUninit::assume_init`] that can cause [undefined behavior]. The
    /// lint should provide extra information to indicate what the problem is
    /// and a possible solution.
    ///
    /// [`mem::zeroed`]: https://doc.rust-lang.org/std/mem/fn.zeroed.html
    /// [`mem::uninitialized`]: https://doc.rust-lang.org/std/mem/fn.uninitialized.html
    /// [`mem::transmute`]: https://doc.rust-lang.org/std/mem/fn.transmute.html
    /// [`MaybeUninit::assume_init`]: https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#method.assume_init
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub INVALID_VALUE,
    Warn,
    "an invalid value is being created (such as a null reference)"
}

declare_lint_pass!(InvalidValue => [INVALID_VALUE]);

/// Information about why a type cannot be initialized this way.
pub struct InitError {
    pub(crate) message: String,
    /// Spans from struct fields and similar that can be obtained from just the type.
    pub(crate) span: Option<Span>,
    /// Used to report a trace through adts.
    pub(crate) nested: Option<Box<InitError>>,
}
impl InitError {
    fn spanned(self, span: Span) -> InitError {
        Self { span: Some(span), ..self }
    }

    fn nested(self, nested: impl Into<Option<InitError>>) -> InitError {
        assert!(self.nested.is_none());
        Self { nested: nested.into().map(Box::new), ..self }
    }
}

impl<'a> From<&'a str> for InitError {
    fn from(s: &'a str) -> Self {
        s.to_owned().into()
    }
}
impl From<String> for InitError {
    fn from(message: String) -> Self {
        Self { message, span: None, nested: None }
    }
}

impl<'tcx> LateLintPass<'tcx> for InvalidValue {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &hir::Expr<'_>) {
        #[derive(Debug, Copy, Clone, PartialEq)]
        enum InitKind {
            Zeroed,
            Uninit,
        }

        /// Test if this constant is all-0.
        fn is_zero(expr: &hir::Expr<'_>) -> bool {
            use hir::ExprKind::*;
            use rustc_ast::LitKind::*;
            match &expr.kind {
                Lit(lit) => {
                    if let Int(i, _) = lit.node {
                        i == 0
                    } else {
                        false
                    }
                }
                Tup(tup) => tup.iter().all(is_zero),
                _ => false,
            }
        }

        /// Determine if this expression is a "dangerous initialization".
        fn is_dangerous_init(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<InitKind> {
            if let hir::ExprKind::Call(path_expr, args) = expr.kind {
                // Find calls to `mem::{uninitialized,zeroed}` methods.
                if let hir::ExprKind::Path(ref qpath) = path_expr.kind {
                    let def_id = cx.qpath_res(qpath, path_expr.hir_id).opt_def_id()?;
                    match cx.tcx.get_diagnostic_name(def_id) {
                        Some(sym::mem_zeroed) => return Some(InitKind::Zeroed),
                        Some(sym::mem_uninitialized) => return Some(InitKind::Uninit),
                        Some(sym::transmute) if is_zero(&args[0]) => return Some(InitKind::Zeroed),
                        _ => {}
                    }
                }
            } else if let hir::ExprKind::MethodCall(_, receiver, ..) = expr.kind {
                // Find problematic calls to `MaybeUninit::assume_init`.
                let def_id = cx.typeck_results().type_dependent_def_id(expr.hir_id)?;
                if cx.tcx.is_diagnostic_item(sym::assume_init, def_id) {
                    // This is a call to *some* method named `assume_init`.
                    // See if the `self` parameter is one of the dangerous constructors.
                    if let hir::ExprKind::Call(path_expr, _) = receiver.kind {
                        if let hir::ExprKind::Path(ref qpath) = path_expr.kind {
                            let def_id = cx.qpath_res(qpath, path_expr.hir_id).opt_def_id()?;
                            match cx.tcx.get_diagnostic_name(def_id) {
                                Some(sym::maybe_uninit_zeroed) => return Some(InitKind::Zeroed),
                                Some(sym::maybe_uninit_uninit) => return Some(InitKind::Uninit),
                                _ => {}
                            }
                        }
                    }
                }
            }

            None
        }

        fn variant_find_init_error<'tcx>(
            cx: &LateContext<'tcx>,
            ty: Ty<'tcx>,
            variant: &VariantDef,
            args: ty::GenericArgsRef<'tcx>,
            descr: &str,
            init: InitKind,
        ) -> Option<InitError> {
            let mut field_err = variant.fields.iter().find_map(|field| {
                ty_find_init_error(cx, field.ty(cx.tcx, args), init).map(|mut err| {
                    if !field.did.is_local() {
                        err
                    } else if err.span.is_none() {
                        err.span = Some(cx.tcx.def_span(field.did));
                        write!(&mut err.message, " (in this {descr})").unwrap();
                        err
                    } else {
                        InitError::from(format!("in this {descr}"))
                            .spanned(cx.tcx.def_span(field.did))
                            .nested(err)
                    }
                })
            });

            // Check if this ADT has a constrained layout (like `NonNull` and friends).
            if let Ok(layout) = cx.tcx.layout_of(cx.typing_env().as_query_input(ty)) {
                if let BackendRepr::Scalar(scalar) | BackendRepr::ScalarPair(scalar, _) =
                    &layout.backend_repr
                {
                    let range = scalar.valid_range(cx);
                    let msg = if !range.contains(0) {
                        "must be non-null"
                    } else if init == InitKind::Uninit && !scalar.is_always_valid(cx) {
                        // Prefer reporting on the fields over the entire struct for uninit,
                        // as the information bubbles out and it may be unclear why the type can't
                        // be null from just its outside signature.

                        "must be initialized inside its custom valid range"
                    } else {
                        return field_err;
                    };
                    if let Some(field_err) = &mut field_err {
                        // Most of the time, if the field error is the same as the struct error,
                        // the struct error only happens because of the field error.
                        if field_err.message.contains(msg) {
                            field_err.message = format!("because {}", field_err.message);
                        }
                    }
                    return Some(InitError::from(format!("`{ty}` {msg}")).nested(field_err));
                }
            }
            field_err
        }

        /// Return `Some` only if we are sure this type does *not*
        /// allow zero initialization.
        fn ty_find_init_error<'tcx>(
            cx: &LateContext<'tcx>,
            ty: Ty<'tcx>,
            init: InitKind,
        ) -> Option<InitError> {
            let ty = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty).unwrap_or(ty);

            match ty.kind() {
                // Primitive types that don't like 0 as a value.
                ty::Ref(..) => Some("references must be non-null".into()),
                ty::Adt(..) if ty.is_box() => Some("`Box` must be non-null".into()),
                ty::FnPtr(..) => Some("function pointers must be non-null".into()),
                ty::Never => Some("the `!` type has no valid value".into()),
                ty::RawPtr(ty, _) if matches!(ty.kind(), ty::Dynamic(..)) =>
                // raw ptr to dyn Trait
                {
                    Some("the vtable of a wide raw pointer must be non-null".into())
                }
                // Primitive types with other constraints.
                ty::Bool if init == InitKind::Uninit => {
                    Some("booleans must be either `true` or `false`".into())
                }
                ty::Char if init == InitKind::Uninit => {
                    Some("characters must be a valid Unicode codepoint".into())
                }
                ty::Int(_) | ty::Uint(_) if init == InitKind::Uninit => {
                    Some("integers must be initialized".into())
                }
                ty::Float(_) if init == InitKind::Uninit => {
                    Some("floats must be initialized".into())
                }
                ty::RawPtr(_, _) if init == InitKind::Uninit => {
                    Some("raw pointers must be initialized".into())
                }
                // Recurse and checks for some compound types. (but not unions)
                ty::Adt(adt_def, args) if !adt_def.is_union() => {
                    // Handle structs.
                    if adt_def.is_struct() {
                        return variant_find_init_error(
                            cx,
                            ty,
                            adt_def.non_enum_variant(),
                            args,
                            "struct field",
                            init,
                        );
                    }
                    // And now, enums.
                    let span = cx.tcx.def_span(adt_def.did());
                    let mut potential_variants = adt_def.variants().iter().filter_map(|variant| {
                        let definitely_inhabited = match variant
                            .inhabited_predicate(cx.tcx, *adt_def)
                            .instantiate(cx.tcx, args)
                            .apply_any_module(cx.tcx, cx.typing_env())
                        {
                            // Entirely skip uninhabited variants.
                            Some(false) => return None,
                            // Forward the others, but remember which ones are definitely inhabited.
                            Some(true) => true,
                            None => false,
                        };
                        Some((variant, definitely_inhabited))
                    });
                    let Some(first_variant) = potential_variants.next() else {
                        return Some(
                            InitError::from("enums with no inhabited variants have no valid value")
                                .spanned(span),
                        );
                    };
                    // So we have at least one potentially inhabited variant. Might we have two?
                    let Some(second_variant) = potential_variants.next() else {
                        // There is only one potentially inhabited variant. So we can recursively
                        // check that variant!
                        return variant_find_init_error(
                            cx,
                            ty,
                            first_variant.0,
                            args,
                            "field of the only potentially inhabited enum variant",
                            init,
                        );
                    };
                    // So we have at least two potentially inhabited variants. If we can prove that
                    // we have at least two *definitely* inhabited variants, then we have a tag and
                    // hence leaving this uninit is definitely disallowed. (Leaving it zeroed could
                    // be okay, depending on which variant is encoded as zero tag.)
                    if init == InitKind::Uninit {
                        let definitely_inhabited = (first_variant.1 as usize)
                            + (second_variant.1 as usize)
                            + potential_variants
                                .filter(|(_variant, definitely_inhabited)| *definitely_inhabited)
                                .count();
                        if definitely_inhabited > 1 {
                            return Some(InitError::from(
                                "enums with multiple inhabited variants have to be initialized to a variant",
                            ).spanned(span));
                        }
                    }
                    // We couldn't find anything wrong here.
                    None
                }
                ty::Tuple(..) => {
                    // Proceed recursively, check all fields.
                    ty.tuple_fields().iter().find_map(|field| ty_find_init_error(cx, field, init))
                }
                ty::Array(ty, len) => {
                    if matches!(len.try_to_target_usize(cx.tcx), Some(v) if v > 0) {
                        // Array length known at array non-empty -- recurse.
                        ty_find_init_error(cx, *ty, init)
                    } else {
                        // Empty array or size unknown.
                        None
                    }
                }
                // Conservative fallback.
                _ => None,
            }
        }

        if let Some(init) = is_dangerous_init(cx, expr) {
            // This conjures an instance of a type out of nothing,
            // using zeroed or uninitialized memory.
            // We are extremely conservative with what we warn about.
            let conjured_ty = cx.typeck_results().expr_ty(expr);
            if let Some(err) = with_no_trimmed_paths!(ty_find_init_error(cx, conjured_ty, init)) {
                let msg = match init {
                    InitKind::Zeroed => fluent::lint_builtin_unpermitted_type_init_zeroed,
                    InitKind::Uninit => fluent::lint_builtin_unpermitted_type_init_uninit,
                };
                let sub = BuiltinUnpermittedTypeInitSub { err };
                cx.emit_span_lint(
                    INVALID_VALUE,
                    expr.span,
                    BuiltinUnpermittedTypeInit {
                        msg,
                        ty: conjured_ty,
                        label: expr.span,
                        sub,
                        tcx: cx.tcx,
                    },
                );
            }
        }
    }
}

declare_lint! {
    /// The `deref_nullptr` lint detects when a null pointer is dereferenced,
    /// which causes [undefined behavior].
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// # #![allow(unused)]
    /// use std::ptr;
    /// unsafe {
    ///     let x = &*ptr::null::<i32>();
    ///     let x = ptr::addr_of!(*ptr::null::<i32>());
    ///     let x = *(0 as *const i32);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Dereferencing a null pointer causes [undefined behavior] if it is accessed
    /// (loaded from or stored to).
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub DEREF_NULLPTR,
    Warn,
    "detects when an null pointer is dereferenced"
}

declare_lint_pass!(DerefNullPtr => [DEREF_NULLPTR]);

impl<'tcx> LateLintPass<'tcx> for DerefNullPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &hir::Expr<'_>) {
        /// test if expression is a null ptr
        fn is_null_ptr(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
            match &expr.kind {
                hir::ExprKind::Cast(expr, ty) => {
                    if let hir::TyKind::Ptr(_) = ty.kind {
                        return is_zero(expr) || is_null_ptr(cx, expr);
                    }
                }
                // check for call to `core::ptr::null` or `core::ptr::null_mut`
                hir::ExprKind::Call(path, _) => {
                    if let hir::ExprKind::Path(ref qpath) = path.kind {
                        if let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id() {
                            return matches!(
                                cx.tcx.get_diagnostic_name(def_id),
                                Some(sym::ptr_null | sym::ptr_null_mut)
                            );
                        }
                    }
                }
                _ => {}
            }
            false
        }

        /// test if expression is the literal `0`
        fn is_zero(expr: &hir::Expr<'_>) -> bool {
            match &expr.kind {
                hir::ExprKind::Lit(lit) => {
                    if let LitKind::Int(a, _) = lit.node {
                        return a == 0;
                    }
                }
                _ => {}
            }
            false
        }

        if let hir::ExprKind::Unary(hir::UnOp::Deref, expr_deref) = expr.kind
            && is_null_ptr(cx, expr_deref)
        {
            if let hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::AddrOf(hir::BorrowKind::Raw, ..),
                ..
            }) = cx.tcx.parent_hir_node(expr.hir_id)
            {
                // `&raw *NULL` is ok.
            } else {
                cx.emit_span_lint(
                    DEREF_NULLPTR,
                    expr.span,
                    BuiltinDerefNullptr { label: expr.span },
                );
            }
        }
    }
}

declare_lint! {
    /// The `named_asm_labels` lint detects the use of named labels in the
    /// inline `asm!` macro.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![feature(asm_experimental_arch)]
    /// use std::arch::asm;
    ///
    /// fn main() {
    ///     unsafe {
    ///         asm!("foo: bar");
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// LLVM is allowed to duplicate inline assembly blocks for any
    /// reason, for example when it is in a function that gets inlined. Because
    /// of this, GNU assembler [local labels] *must* be used instead of labels
    /// with a name. Using named labels might cause assembler or linker errors.
    ///
    /// See the explanation in [Rust By Example] for more details.
    ///
    /// [local labels]: https://sourceware.org/binutils/docs/as/Symbol-Names.html#Local-Labels
    /// [Rust By Example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels
    pub NAMED_ASM_LABELS,
    Deny,
    "named labels in inline assembly",
}

declare_lint! {
    /// The `binary_asm_labels` lint detects the use of numeric labels containing only binary
    /// digits in the inline `asm!` macro.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (fails on non-x86_64)
    /// #![cfg(target_arch = "x86_64")]
    ///
    /// use std::arch::asm;
    ///
    /// fn main() {
    ///     unsafe {
    ///         asm!("0: jmp 0b");
    ///     }
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// error: avoid using labels containing only the digits `0` and `1` in inline assembly
    ///  --> <source>:7:15
    ///   |
    /// 7 |         asm!("0: jmp 0b");
    ///   |               ^ use a different label that doesn't start with `0` or `1`
    ///   |
    ///   = help: start numbering with `2` instead
    ///   = note: an LLVM bug makes these labels ambiguous with a binary literal number on x86
    ///   = note: see <https://github.com/llvm/llvm-project/issues/99547> for more information
    ///   = note: `#[deny(binary_asm_labels)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// An [LLVM bug] causes this code to fail to compile because it interprets the `0b` as a binary
    /// literal instead of a reference to the previous local label `0`. To work around this bug,
    /// don't use labels that could be confused with a binary literal.
    ///
    /// This behavior is platform-specific to x86 and x86-64.
    ///
    /// See the explanation in [Rust By Example] for more details.
    ///
    /// [LLVM bug]: https://github.com/llvm/llvm-project/issues/99547
    /// [Rust By Example]: https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels
    pub BINARY_ASM_LABELS,
    Deny,
    "labels in inline assembly containing only 0 or 1 digits",
}

declare_lint_pass!(AsmLabels => [NAMED_ASM_LABELS, BINARY_ASM_LABELS]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsmLabelKind {
    Named,
    FormatArg,
    Binary,
}

impl<'tcx> LateLintPass<'tcx> for AsmLabels {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::Expr {
            kind:
                hir::ExprKind::InlineAsm(hir::InlineAsm {
                    asm_macro: AsmMacro::Asm | AsmMacro::NakedAsm,
                    template_strs,
                    options,
                    ..
                }),
            ..
        } = expr
        {
            // asm with `options(raw)` does not do replacement with `{` and `}`.
            let raw = options.contains(InlineAsmOptions::RAW);

            for (template_sym, template_snippet, template_span) in template_strs.iter() {
                let template_str = template_sym.as_str();
                let find_label_span = |needle: &str| -> Option<Span> {
                    if let Some(template_snippet) = template_snippet {
                        let snippet = template_snippet.as_str();
                        if let Some(pos) = snippet.find(needle) {
                            let end = pos
                                + snippet[pos..]
                                    .find(|c| c == ':')
                                    .unwrap_or(snippet[pos..].len() - 1);
                            let inner = InnerSpan::new(pos, end);
                            return Some(template_span.from_inner(inner));
                        }
                    }

                    None
                };

                // diagnostics are emitted per-template, so this is created here as opposed to the outer loop
                let mut spans = Vec::new();

                // A semicolon might not actually be specified as a separator for all targets, but
                // it seems like LLVM accepts it always.
                let statements = template_str.split(|c| matches!(c, '\n' | ';'));
                for statement in statements {
                    // If there's a comment, trim it from the statement
                    let statement = statement.find("//").map_or(statement, |idx| &statement[..idx]);

                    // In this loop, if there is ever a non-label, no labels can come after it.
                    let mut start_idx = 0;
                    'label_loop: for (idx, _) in statement.match_indices(':') {
                        let possible_label = statement[start_idx..idx].trim();
                        let mut chars = possible_label.chars();

                        let Some(start) = chars.next() else {
                            // Empty string means a leading ':' in this section, which is not a
                            // label.
                            break 'label_loop;
                        };

                        // Whether a { bracket has been seen and its } hasn't been found yet.
                        let mut in_bracket = false;
                        let mut label_kind = AsmLabelKind::Named;

                        // A label can also start with a format arg, if it's not a raw asm block.
                        if !raw && start == '{' {
                            in_bracket = true;
                            label_kind = AsmLabelKind::FormatArg;
                        } else if matches!(start, '0' | '1') {
                            // Binary labels have only the characters `0` or `1`.
                            label_kind = AsmLabelKind::Binary;
                        } else if !(start.is_ascii_alphabetic() || matches!(start, '.' | '_')) {
                            // Named labels start with ASCII letters, `.` or `_`.
                            // anything else is not a label
                            break 'label_loop;
                        }

                        for c in chars {
                            // Inside a template format arg, any character is permitted for the
                            // puproses of label detection because we assume that it can be
                            // replaced with some other valid label string later. `options(raw)`
                            // asm blocks cannot have format args, so they are excluded from this
                            // special case.
                            if !raw && in_bracket {
                                if c == '{' {
                                    // Nested brackets are not allowed in format args, this cannot
                                    // be a label.
                                    break 'label_loop;
                                }

                                if c == '}' {
                                    // The end of the format arg.
                                    in_bracket = false;
                                }
                            } else if !raw && c == '{' {
                                // Start of a format arg.
                                in_bracket = true;
                                label_kind = AsmLabelKind::FormatArg;
                            } else {
                                let can_continue = match label_kind {
                                    // Format arg labels are considered to be named labels for the purposes
                                    // of continuing outside of their {} pair.
                                    AsmLabelKind::Named | AsmLabelKind::FormatArg => {
                                        c.is_ascii_alphanumeric() || matches!(c, '_' | '$')
                                    }
                                    AsmLabelKind::Binary => matches!(c, '0' | '1'),
                                };

                                if !can_continue {
                                    // The potential label had an invalid character inside it, it
                                    // cannot be a label.
                                    break 'label_loop;
                                }
                            }
                        }

                        // If all characters passed the label checks, this is a label.
                        spans.push((find_label_span(possible_label), label_kind));
                        start_idx = idx + 1;
                    }
                }

                for (span, label_kind) in spans {
                    let missing_precise_span = span.is_none();
                    let span = span.unwrap_or(*template_span);
                    match label_kind {
                        AsmLabelKind::Named => {
                            cx.emit_span_lint(
                                NAMED_ASM_LABELS,
                                span,
                                InvalidAsmLabel::Named { missing_precise_span },
                            );
                        }
                        AsmLabelKind::FormatArg => {
                            cx.emit_span_lint(
                                NAMED_ASM_LABELS,
                                span,
                                InvalidAsmLabel::FormatArg { missing_precise_span },
                            );
                        }
                        // the binary asm issue only occurs when using intel syntax on x86 targets
                        AsmLabelKind::Binary
                            if !options.contains(InlineAsmOptions::ATT_SYNTAX)
                                && matches!(
                                    cx.tcx.sess.asm_arch,
                                    Some(InlineAsmArch::X86 | InlineAsmArch::X86_64) | None
                                ) =>
                        {
                            cx.emit_span_lint(
                                BINARY_ASM_LABELS,
                                span,
                                InvalidAsmLabel::Binary { missing_precise_span, span },
                            )
                        }
                        // No lint on anything other than x86
                        AsmLabelKind::Binary => (),
                    };
                }
            }
        }
    }
}

declare_lint! {
    /// The `special_module_name` lint detects module
    /// declarations for files that have a special meaning.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// mod lib;
    ///
    /// fn main() {
    ///     lib::run();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Cargo recognizes `lib.rs` and `main.rs` as the root of a
    /// library or binary crate, so declaring them as modules
    /// will lead to miscompilation of the crate unless configured
    /// explicitly.
    ///
    /// To access a library from a binary target within the same crate,
    /// use `your_crate_name::` as the path instead of `lib::`:
    ///
    /// ```rust,compile_fail
    /// // bar/src/lib.rs
    /// fn run() {
    ///     // ...
    /// }
    ///
    /// // bar/src/main.rs
    /// fn main() {
    ///     bar::run();
    /// }
    /// ```
    ///
    /// Binary targets cannot be used as libraries and so declaring
    /// one as a module is not allowed.
    pub SPECIAL_MODULE_NAME,
    Warn,
    "module declarations for files with a special meaning",
}

declare_lint_pass!(SpecialModuleName => [SPECIAL_MODULE_NAME]);

impl EarlyLintPass for SpecialModuleName {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &ast::Crate) {
        for item in &krate.items {
            if let ast::ItemKind::Mod(
                _,
                ident,
                ast::ModKind::Unloaded | ast::ModKind::Loaded(_, ast::Inline::No, _, _),
            ) = item.kind
            {
                if item.attrs.iter().any(|a| a.has_name(sym::path)) {
                    continue;
                }

                match ident.name.as_str() {
                    "lib" => cx.emit_span_lint(
                        SPECIAL_MODULE_NAME,
                        item.span,
                        BuiltinSpecialModuleNameUsed::Lib,
                    ),
                    "main" => cx.emit_span_lint(
                        SPECIAL_MODULE_NAME,
                        item.span,
                        BuiltinSpecialModuleNameUsed::Main,
                    ),
                    _ => continue,
                }
            }
        }
    }
}
