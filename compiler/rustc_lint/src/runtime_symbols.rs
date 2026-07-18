use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, CanonicalSymbol, FnSig, ForeignItemKind};
use rustc_infer::infer::DefineOpaqueTypes;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{Span, Symbol};
use rustc_trait_selection::infer::TyCtxtInferExt;

use crate::lints::RedefiningRuntimeSymbolsDiag;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `invalid_runtime_symbol_definitions` lint checks the signature of items whose
    /// symbol name is a runtime symbol expected by `core` or `std` differs significantly from the
    /// expected signature (like mismatch ABI, mismatch C variadics, mismatch argument count,
    /// missing return type, ...).
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
    /// `bcmp`, `strlen`, as well as the following POSIX symbols: `open`, `read`, `write`
    /// `close`, `malloc`, `realloc`, `free` and `exit`.
    ///
    /// [^1]: https://doc.rust-lang.org/core/index.html#how-to-use-the-core-library
    pub INVALID_RUNTIME_SYMBOL_DEFINITIONS,
    Deny,
    "invalid definition of a symbol used by the standard library"
}

declare_lint! {
    /// The `suspicious_runtime_symbol_definitions` lint checks the signature of items whose
    /// symbol name is a runtime symbol expected by `core` or `std`.
    ///
    /// ### Example
    ///
    /// ```rust,no_run,standalone_crate
    /// #[unsafe(no_mangle)]
    /// pub extern "C" fn strlen(ptr: *mut f32) -> usize { 0 }
    /// // suspicious definition of the `strlen` function
    /// // `ptr` should be `*const std::ffi::c_char`
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
    /// `bcmp`, `strlen`, as well as the following POSIX symbols: `open`, `read`, `write`
    /// `close`, `malloc`, `realloc`, `free` and `exit`.
    ///
    /// [^1]: https://doc.rust-lang.org/core/index.html#how-to-use-the-core-library
    pub SUSPICIOUS_RUNTIME_SYMBOL_DEFINITIONS,
    Warn,
    "suspicious definition of a symbol used by the standard library"
}

declare_lint_pass!(RuntimeSymbols => [INVALID_RUNTIME_SYMBOL_DEFINITIONS, SUSPICIOUS_RUNTIME_SYMBOL_DEFINITIONS]);

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

                // Try to get the overridden symbol name of this function (our mangling
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

                check_static(cx, &symbol_name, def_id, item.span);
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
                            check_static(cx, &symbol_name.name, did, item.span);
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
    let s = Symbol::intern(symbol_name);
    let Some(CanonicalSymbol { symbol: _, def_id: expected_def_id }) =
        cx.tcx.all_canonical_symbols(()).iter().find(|cs| cs.symbol == s)
    else {
        // The symbol name does not correspond to a runtime symbols, bail out
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

        if lang_sig.abi() != user_sig.abi()
            || lang_sig.c_variadic() != user_sig.c_variadic()
            || lang_sig.inputs().skip_binder().len() != user_sig.inputs().skip_binder().len()
            || (!lang_sig.output().skip_binder().is_unit()
                && user_sig.output().skip_binder().is_unit())
        {
            cx.emit_span_lint(
                INVALID_RUNTIME_SYMBOL_DEFINITIONS,
                sig.span,
                RedefiningRuntimeSymbolsDiag::FnDefInvalid {
                    symbol_name: symbol_name.to_string(),
                    found_fn_sig: actual,
                    expected_fn_sig: expected,
                },
            );
        } else {
            cx.emit_span_lint(
                SUSPICIOUS_RUNTIME_SYMBOL_DEFINITIONS,
                sig.span,
                RedefiningRuntimeSymbolsDiag::FnDefSuspicious {
                    symbol_name: symbol_name.to_string(),
                    found_fn_sig: actual,
                    expected_fn_sig: expected,
                },
            );
        };
    }
}

fn check_static<'tcx>(cx: &LateContext<'tcx>, symbol_name: &str, did: LocalDefId, sp: Span) {
    let s = Symbol::intern(symbol_name);
    let Some(CanonicalSymbol { symbol: _, def_id: expected_def_id }) =
        cx.tcx.all_canonical_symbols(()).iter().find(|cs| cs.symbol == s)
    else {
        // The symbol name does not correspond to a runtime symbols, bail out
        return;
    };

    // Get the expected symbol function signature
    let lang_sig = cx.tcx.normalize_erasing_regions(
        cx.typing_env(),
        cx.tcx.fn_sig(expected_def_id).instantiate_identity(),
    );

    // Get the static type
    let outer_user_sig = cx.tcx.type_of(did).instantiate_identity().skip_norm_wip();

    // Peel Option<...> and get the inner type (see std weak! macro with #[linkage = "extern_weak"])
    let user_sig: Ty<'_> = match outer_user_sig.kind() {
        ty::Adt(def, args) if Some(def.did()) == cx.tcx.lang_items().option_type() => {
            args.type_at(0)
        }
        _ => outer_user_sig,
    };

    let user_sig = if let ty::FnPtr(sig_tys, hdr) = user_sig.kind() {
        sig_tys.with(*hdr)
    } else {
        // not a function pointer, report an error

        let lang_sig = Ty::new_fn_ptr(cx.tcx, lang_sig);
        cx.emit_span_lint(
            INVALID_RUNTIME_SYMBOL_DEFINITIONS,
            sp,
            RedefiningRuntimeSymbolsDiag::Static {
                static_ty: user_sig,
                symbol_name: symbol_name.to_string(),
                expected_fn_sig: lang_sig,
            },
        );
        return;
    };

    // Compare the two signatures with an inference context
    let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
    let cause = rustc_middle::traits::ObligationCause::misc(sp, did);
    let result = infcx.at(&cause, cx.param_env).eq(DefineOpaqueTypes::No, lang_sig, user_sig);

    // Compare the expected function signature with the static type, report an error if they don't match
    if result.is_err() {
        let user_sig = Ty::new_fn_ptr(cx.tcx, user_sig);
        let lang_sig = Ty::new_fn_ptr(cx.tcx, lang_sig);

        cx.emit_span_lint(
            INVALID_RUNTIME_SYMBOL_DEFINITIONS,
            sp,
            RedefiningRuntimeSymbolsDiag::Static {
                static_ty: user_sig,
                symbol_name: symbol_name.to_string(),
                expected_fn_sig: lang_sig,
            },
        );
    }
}
