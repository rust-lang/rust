use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::{for_each_top_level_late_bound_region, is_copy};
use clippy_utils::{is_self, is_self_ty};
use core::ops::ControlFlow;
use rustc_abi::ExternAbi;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{BindingMode, Body, FnDecl, Impl, ItemKind, MutTy, Mutability, Node, PatKind, find_attr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, PointerCoercion};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, RegionKind, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions taking arguments by reference, where
    /// the argument type is `Copy` and small enough to be more efficient to always
    /// pass by value.
    ///
    /// ### Why is this bad?
    /// In many calling conventions instances of structs will
    /// be passed through registers if they fit into two or less general purpose
    /// registers.
    ///
    /// ### Known problems
    /// This lint is target dependent, some cases will lint on 64-bit targets but
    /// not 32-bit or lower targets.
    ///
    /// The configuration option `trivial_copy_size_limit` can be set to override
    /// this limit for a project.
    ///
    /// This lint attempts to allow passing arguments by reference if a reference
    /// to that argument is returned. This is implemented by comparing the lifetime
    /// of the argument and return value for equality. However, this can cause
    /// false positives in cases involving multiple lifetimes that are bounded by
    /// each other.
    ///
    /// Also, it does not take account of other similar cases where getting memory addresses
    /// matters; namely, returning the pointer to the argument in question,
    /// and passing the argument, as both references and pointers,
    /// to a function that needs the memory address. For further details, refer to
    /// [this issue](https://github.com/rust-lang/rust-clippy/issues/5953)
    /// that explains a real case in which this false positive
    /// led to an **undefined behavior** introduced with unsafe code.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// fn foo(v: &u32) {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn foo(v: u32) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRIVIALLY_COPY_PASS_BY_REF,
    pedantic,
    "functions taking small copyable arguments by reference"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions taking arguments by value, where
    /// the argument type is `Copy` and large enough to be worth considering
    /// passing by reference. Does not trigger if the function is being exported,
    /// because that might induce API breakage, if the parameter is declared as mutable,
    /// or if the argument is a `self`.
    ///
    /// ### Why is this bad?
    /// Arguments passed by value might result in an unnecessary
    /// shallow copy, taking up more space in the stack and requiring a call to
    /// `memcpy`, which can be expensive.
    ///
    /// ### Example
    /// ```no_run
    /// #[derive(Clone, Copy)]
    /// struct TooLarge([u8; 2048]);
    ///
    /// fn foo(v: TooLarge) {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # #[derive(Clone, Copy)]
    /// # struct TooLarge([u8; 2048]);
    /// fn foo(v: &TooLarge) {}
    /// ```
    #[clippy::version = "1.49.0"]
    pub LARGE_TYPES_PASSED_BY_VALUE,
    pedantic,
    "functions taking large arguments by value"
}

pub struct PassByRefOrValue {
    ref_min_size: u64,
    value_max_size: u64,
    avoid_breaking_exported_api: bool,
}

impl PassByRefOrValue {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let ref_min_size = conf
            .trivial_copy_size_limit
            .unwrap_or_else(|| u64::from(tcx.sess.target.pointer_width / 8));

        Self {
            ref_min_size,
            value_max_size: conf.pass_by_value_size_limit,
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
        }
    }

    fn check_poly_fn(&self, cx: &LateContext<'_>, def_id: LocalDefId, decl: &FnDecl<'_>, span: Option<Span>) {
        if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(def_id) {
            return;
        }

        let fn_sig = cx.tcx.fn_sig(def_id).instantiate_identity();
        let fn_body = cx.enclosing_body.map(|id| cx.tcx.hir_body(id));

        // Gather all the lifetimes found in the output type which may affect whether
        // `TRIVIALLY_COPY_PASS_BY_REF` should be linted.
        let mut output_regions = FxHashSet::default();
        let _ = for_each_top_level_late_bound_region(fn_sig.skip_binder().output(), |region| -> ControlFlow<!> {
            output_regions.insert(region);
            ControlFlow::Continue(())
        });

        for (index, (input, ty)) in iter::zip(
            decl.inputs,
            fn_sig.skip_binder().inputs().iter().map(|&ty| fn_sig.rebind(ty)),
        )
        .enumerate()
        {
            // All spans generated from a proc-macro invocation are the same...
            match span {
                Some(s) if s == input.span => continue,
                _ => (),
            }

            match *ty.skip_binder().kind() {
                ty::Ref(lt, ty, Mutability::Not) => {
                    match lt.kind() {
                        RegionKind::ReBound(index, region)
                            if index.as_u32() == 0 && output_regions.contains(&region) =>
                        {
                            continue;
                        },
                        // Early bound regions on functions are either from the containing item, are bounded by another
                        // lifetime, or are used as a bound for a type or lifetime.
                        RegionKind::ReEarlyParam(..) => continue,
                        _ => (),
                    }

                    let ty = cx.tcx.instantiate_bound_regions_with_erased(fn_sig.rebind(ty));
                    if is_copy(cx, ty)
                        && let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes())
                        && size <= self.ref_min_size
                        && let hir::TyKind::Ref(_, MutTy { ty: decl_ty, .. }) = input.kind
                    {
                        if let Some(typeck) = cx.maybe_typeck_results()
                            // Don't lint if a raw pointer is created.
                            // TODO: Limit the check only to raw pointers to the argument (or part of the argument)
                            //       which escape the current function.
                            && (typeck.node_types().items().any(|(_, &ty)| ty.is_raw_ptr())
                                || typeck
                                    .adjustments()
                                    .items()
                                    .flat_map(|(_, a)| a)
                                    .any(|a| matches!(a.kind, Adjust::Pointer(PointerCoercion::UnsafeFnPointer))))
                        {
                            continue;
                        }
                        let value_type = if fn_body.and_then(|body| body.params.get(index)).is_some_and(is_self) {
                            "self".into()
                        } else {
                            snippet(cx, decl_ty.span, "_").into()
                        };
                        span_lint_and_sugg(
                            cx,
                            TRIVIALLY_COPY_PASS_BY_REF,
                            input.span,
                            format!(
                                "this argument ({size} byte) is passed by reference, but would be more efficient if passed by value (limit: {} byte)",
                                self.ref_min_size
                            ),
                            "consider passing by value instead",
                            value_type,
                            Applicability::Unspecified,
                        );
                    }
                },

                ty::Adt(_, _) | ty::Array(_, _) | ty::Tuple(_) => {
                    // if function has a body and parameter is annotated with mut, ignore
                    if let Some(param) = fn_body.and_then(|body| body.params.get(index)) {
                        match param.pat.kind {
                            PatKind::Binding(BindingMode::NONE, _, _, _) => {},
                            _ => continue,
                        }
                    }
                    let ty = cx.tcx.instantiate_bound_regions_with_erased(ty);

                    if is_copy(cx, ty)
                        && !is_self_ty(input)
                        && let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes())
                        && size > self.value_max_size
                    {
                        span_lint_and_sugg(
                            cx,
                            LARGE_TYPES_PASSED_BY_VALUE,
                            input.span,
                            format!(
                                "this argument ({size} byte) is passed by value, but might be more efficient if passed by reference (limit: {} byte)",
                                self.value_max_size
                            ),
                            "consider passing by reference instead",
                            format!("&{}", snippet(cx, input.span, "_")),
                            Applicability::MaybeIncorrect,
                        );
                    }
                },

                _ => {},
            }
        }
    }
}

impl_lint_pass!(PassByRefOrValue => [TRIVIALLY_COPY_PASS_BY_REF, LARGE_TYPES_PASSED_BY_VALUE]);

impl<'tcx> LateLintPass<'tcx> for PassByRefOrValue {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        if item.span.from_expansion() {
            return;
        }

        if let hir::TraitItemKind::Fn(method_sig, _) = &item.kind {
            self.check_poly_fn(cx, item.owner_id.def_id, method_sig.decl, None);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        _body: &'tcx Body<'_>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if span.from_expansion() {
            return;
        }

        let hir_id = cx.tcx.local_def_id_to_hir_id(def_id);
        match kind {
            FnKind::ItemFn(.., header) => {
                if header.abi != ExternAbi::Rust {
                    return;
                }
                let attrs = cx.tcx.hir_attrs(hir_id);
                if find_attr!(attrs, AttributeKind::Inline(InlineAttr::Always, _)) {
                    return;
                }

                for a in attrs {
                    // FIXME(jdonszelmann): make part of the find_attr above
                    if a.has_name(sym::proc_macro_derive) {
                        return;
                    }
                }
            },
            FnKind::Method(..) => (),
            FnKind::Closure => return,
        }

        // Exclude non-inherent impls
        if let Node::Item(item) = cx.tcx.parent_hir_node(hir_id)
            && matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            )
        {
            return;
        }

        self.check_poly_fn(cx, def_id, decl, Some(span));
    }
}
