use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, FnSig, ForeignItemKind, LanguageItems};
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;
use rustc_trait_selection::infer::TyCtxtInferExt;

use crate::lints::RedefiningRuntimeSymbolsDiag;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `invalid_runtime_symbol_definitions` lint checks the signature of items whose
    /// symbol name is a runtime symbols expected by `core`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #[unsafe(no_mangle)]
    /// pub fn strlen() {} // invalid definition of the `strlen` function
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Up-most care is required when defining runtime symbols assumed and
    /// used by the standard library. They must follow the C specification, not use any
    /// standard-library facility or undefined behavior may occur.
    ///
    /// The symbols currently checked are `memcpy`, `memmove`, `memset`, `memcmp`,
    /// `bcmp` and `strlen`.
    ///
    /// [^1]: https://doc.rust-lang.org/core/index.html#how-to-use-the-core-library
    pub INVALID_RUNTIME_SYMBOL_DEFINITIONS,
    Deny,
    "invalid definition of a symbol used by the standard library"
}

declare_lint_pass!(RuntimeSymbols => [INVALID_RUNTIME_SYMBOL_DEFINITIONS]);

static EXPECTED_SYMBOLS: &[ExpectedSymbol] = &[
    ExpectedSymbol { symbol: "memcpy", lang: LanguageItems::memcpy_fn },
    ExpectedSymbol { symbol: "memmove", lang: LanguageItems::memmove_fn },
    ExpectedSymbol { symbol: "memset", lang: LanguageItems::memset_fn },
    ExpectedSymbol { symbol: "memcmp", lang: LanguageItems::memcmp_fn },
    ExpectedSymbol { symbol: "bcmp", lang: LanguageItems::bcmp_fn },
    ExpectedSymbol { symbol: "strlen", lang: LanguageItems::strlen_fn },
];

#[derive(Copy, Clone, Debug)]
struct ExpectedSymbol {
    symbol: &'static str,
    lang: fn(&LanguageItems) -> Option<DefId>,
}

impl<'tcx> LateLintPass<'tcx> for RuntimeSymbols {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        // Bail-out if the item is not a function/method or static.
        match item.kind {
            hir::ItemKind::Fn { sig, ident: _, generics, body: _, has_body: _ } => {
                // Generic functions cannot have the same runtime symbol as we do not allow
                // any symbol attributes.
                if !generics.params.is_empty() {
                    return;
                }

                // Try to the overridden symbol name of this function (our mangling
                // cannot ever conflict with runtime symbols, so no need to check for those).
                let Some(symbol_name) = rustc_symbol_mangling::symbol_name_from_attrs(
                    cx.tcx,
                    rustc_middle::ty::InstanceKind::Item(item.owner_id.to_def_id()),
                ) else {
                    return;
                };

                check_fn(cx, &symbol_name, sig, item.owner_id.def_id);
            }
            hir::ItemKind::Static(..) => {
                // Compute the symbol name of this static (without mangling, as our mangling
                // cannot ever conflict with runtime symbols).
                let Some(symbol_name) = rustc_symbol_mangling::symbol_name_from_attrs(
                    cx.tcx,
                    rustc_middle::ty::InstanceKind::Item(item.owner_id.to_def_id()),
                ) else {
                    return;
                };

                let def_id = item.owner_id.def_id;
                let static_ty = cx.tcx.type_of(def_id).instantiate_identity();

                check_static(cx, &symbol_name, static_ty, item.span);
            }
            hir::ItemKind::ForeignMod { abi: _, items } => {
                for item in items {
                    let item = cx.tcx.hir_foreign_item(*item);

                    let did = item.owner_id.def_id;
                    let instance = Instance::new_raw(
                        did.to_def_id(),
                        ty::List::identity_for_item(cx.tcx, did),
                    );
                    let symbol_name = cx.tcx.symbol_name(instance);

                    match item.kind {
                        ForeignItemKind::Fn(fn_sig, _idents, _generics) => {
                            check_fn(cx, &symbol_name.name, fn_sig, did);
                        }
                        ForeignItemKind::Static(..) => {
                            let def_id = item.owner_id.def_id;
                            let static_ty = cx.tcx.type_of(def_id).instantiate_identity();
                            check_static(cx, &symbol_name.name, static_ty, item.span);
                        }
                        ForeignItemKind::Type => return,
                    }
                }
            }
            _ => return,
        }
    }
}

fn check_fn(cx: &LateContext<'_>, symbol_name: &str, sig: FnSig<'_>, did: LocalDefId) {
    let Some(expected_symbol) = EXPECTED_SYMBOLS.iter().find(|es| es.symbol == symbol_name) else {
        // The symbol name does not correspond to a runtime symbols, bail out
        return;
    };

    let Some(expected_def_id) = (expected_symbol.lang)(&cx.tcx.lang_items()) else {
        // Can't find the corresponding language item, bail out
        return;
    };

    // Get the two function signatures
    let lang_sig = cx.tcx.normalize_erasing_regions(
        cx.typing_env(),
        cx.tcx.fn_sig(expected_def_id).instantiate_identity(),
    );
    let user_sig = cx
        .tcx
        .normalize_erasing_regions(cx.typing_env(), cx.tcx.fn_sig(did).instantiate_identity());

    // Compare the two signatures with an inference context
    let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
    let cause = rustc_middle::traits::ObligationCause::misc(sig.span, did);
    let result = infcx.at(&cause, cx.param_env).eq(DefineOpaqueTypes::No, lang_sig, user_sig);

    // If they don't match, emit our own mismatch signatures
    if let Err(_terr) = result {
        // Create fn pointers for diagnostics purpose
        let expected = Ty::new_fn_ptr(cx.tcx, lang_sig);
        let actual = Ty::new_fn_ptr(cx.tcx, user_sig);

        cx.emit_span_lint(
            INVALID_RUNTIME_SYMBOL_DEFINITIONS,
            sig.span,
            RedefiningRuntimeSymbolsDiag::FnDef {
                symbol_name: symbol_name.to_string(),
                found_fn_sig: actual,
                expected_fn_sig: expected,
            },
        );
    }
}

fn check_static<'tcx>(cx: &LateContext<'tcx>, symbol_name: &str, static_ty: Ty<'tcx>, sp: Span) {
    let Some(expected_symbol) = EXPECTED_SYMBOLS.iter().find(|es| es.symbol == symbol_name) else {
        // The symbol name does not correspond to a runtime symbols, bail out
        return;
    };

    let Some(expected_def_id) = (expected_symbol.lang)(&cx.tcx.lang_items()) else {
        // Can't find the corresponding language item, bail out
        return;
    };

    // Unconditionally report a mismatch, a static cannot ever be a function definition

    let lang_sig = cx.tcx.normalize_erasing_regions(
        cx.typing_env(),
        cx.tcx.fn_sig(expected_def_id).instantiate_identity(),
    );

    let expected = Ty::new_fn_ptr(cx.tcx, lang_sig);

    cx.emit_span_lint(
        INVALID_RUNTIME_SYMBOL_DEFINITIONS,
        sp,
        RedefiningRuntimeSymbolsDiag::Static {
            static_ty,
            symbol_name: symbol_name.to_string(),
            expected_fn_sig: expected,
        },
    );
}
