use std::cmp;

use crate::utils::{is_copy, is_self_ty, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::attr;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{BindingAnnotation, Body, FnDecl, HirId, ItemKind, MutTy, Mutability, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Span};
use rustc_target::abi::LayoutOf;
use rustc_target::spec::abi::Abi;
use rustc_target::spec::Target;

declare_clippy_lint! {
    /// **What it does:** Checks for functions taking arguments by reference, where
    /// the argument type is `Copy` and small enough to be more efficient to always
    /// pass by value.
    ///
    /// **Why is this bad?** In many calling conventions instances of structs will
    /// be passed through registers if they fit into two or less general purpose
    /// registers.
    ///
    /// **Known problems:** This lint is target register size dependent, it is
    /// limited to 32-bit to try and reduce portability problems between 32 and
    /// 64-bit, but if you are compiling for 8 or 16-bit targets then the limit
    /// will be different.
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
    /// **Example:**
    ///
    /// ```rust
    /// // Bad
    /// fn foo(v: &u32) {}
    /// ```
    ///
    /// ```rust
    /// // Better
    /// fn foo(v: u32) {}
    /// ```
    pub TRIVIALLY_COPY_PASS_BY_REF,
    pedantic,
    "functions taking small copyable arguments by reference"
}

declare_clippy_lint! {
    /// **What it does:** Checks for functions taking arguments by value, where
    /// the argument type is `Copy` and large enough to be worth considering
    /// passing by reference. Does not trigger if the function is being exported,
    /// because that might induce API breakage, if the parameter is declared as mutable,
    /// or if the argument is a `self`.
    ///
    /// **Why is this bad?** Arguments passed by value might result in an unnecessary
    /// shallow copy, taking up more space in the stack and requiring a call to
    /// `memcpy`, which which can be expensive.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// #[derive(Clone, Copy)]
    /// struct TooLarge([u8; 2048]);
    ///
    /// // Bad
    /// fn foo(v: TooLarge) {}
    /// ```
    /// ```rust
    /// #[derive(Clone, Copy)]
    /// struct TooLarge([u8; 2048]);
    ///
    /// // Good
    /// fn foo(v: &TooLarge) {}
    /// ```
    pub LARGE_TYPES_PASSED_BY_VALUE,
    pedantic,
    "functions taking large arguments by value"
}

#[derive(Copy, Clone)]
pub struct PassByRefOrValue {
    ref_min_size: u64,
    value_max_size: u64,
}

impl<'tcx> PassByRefOrValue {
    pub fn new(ref_min_size: Option<u64>, value_max_size: u64, target: &Target) -> Self {
        let ref_min_size = ref_min_size.unwrap_or_else(|| {
            let bit_width = u64::from(target.pointer_width);
            // Cap the calculated bit width at 32-bits to reduce
            // portability problems between 32 and 64-bit targets
            let bit_width = cmp::min(bit_width, 32);
            #[allow(clippy::integer_division)]
            let byte_width = bit_width / 8;
            // Use a limit of 2 times the register byte width
            byte_width * 2
        });

        Self {
            ref_min_size,
            value_max_size,
        }
    }

    fn check_poly_fn(&mut self, cx: &LateContext<'tcx>, hir_id: HirId, decl: &FnDecl<'_>, span: Option<Span>) {
        let fn_def_id = cx.tcx.hir().local_def_id(hir_id);

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(fn_sig);

        let fn_body = cx.enclosing_body.map(|id| cx.tcx.hir().body(id));

        for (index, (input, &ty)) in decl.inputs.iter().zip(fn_sig.inputs()).enumerate() {
            // All spans generated from a proc-macro invocation are the same...
            match span {
                Some(s) if s == input.span => return,
                _ => (),
            }

            match ty.kind() {
                ty::Ref(input_lt, ty, Mutability::Not) => {
                    // Use lifetimes to determine if we're returning a reference to the
                    // argument. In that case we can't switch to pass-by-value as the
                    // argument will not live long enough.
                    let output_lts = match *fn_sig.output().kind() {
                        ty::Ref(output_lt, _, _) => vec![output_lt],
                        ty::Adt(_, substs) => substs.regions().collect(),
                        _ => vec![],
                    };

                    if_chain! {
                        if !output_lts.contains(&input_lt);
                        if is_copy(cx, ty);
                        if let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes());
                        if size <= self.ref_min_size;
                        if let hir::TyKind::Rptr(_, MutTy { ty: ref decl_ty, .. }) = input.kind;
                        then {
                            let value_type = if is_self_ty(decl_ty) {
                                "self".into()
                            } else {
                                snippet(cx, decl_ty.span, "_").into()
                            };
                            span_lint_and_sugg(
                                cx,
                                TRIVIALLY_COPY_PASS_BY_REF,
                                input.span,
                                &format!("this argument ({} byte) is passed by reference, but would be more efficient if passed by value (limit: {} byte)", size, self.ref_min_size),
                                "consider passing by value instead",
                                value_type,
                                Applicability::Unspecified,
                            );
                        }
                    }
                },

                ty::Adt(_, _) | ty::Array(_, _) | ty::Tuple(_) => {
                    // if function has a body and parameter is annotated with mut, ignore
                    if let Some(param) = fn_body.and_then(|body| body.params.get(index)) {
                        match param.pat.kind {
                            PatKind::Binding(BindingAnnotation::Unannotated, _, _, _) => {},
                            _ => continue,
                        }
                    }

                    if_chain! {
                        if !cx.access_levels.is_exported(hir_id);
                        if is_copy(cx, ty);
                        if !is_self_ty(input);
                        if let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes());
                        if size > self.value_max_size;
                        then {
                            span_lint_and_sugg(
                                cx,
                                LARGE_TYPES_PASSED_BY_VALUE,
                                input.span,
                                &format!("this argument ({} byte) is passed by value, but might be more efficient if passed by reference (limit: {} byte)", size, self.value_max_size),
                                "consider passing by reference instead",
                                format!("&{}", snippet(cx, input.span, "_")),
                                Applicability::MaybeIncorrect,
                            );
                        }
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
            self.check_poly_fn(cx, item.hir_id, &*method_sig.decl, None);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        _body: &'tcx Body<'_>,
        span: Span,
        hir_id: HirId,
    ) {
        if span.from_expansion() {
            return;
        }

        match kind {
            FnKind::ItemFn(.., header, _, attrs) => {
                if header.abi != Abi::Rust {
                    return;
                }
                for a in attrs {
                    if let Some(meta_items) = a.meta_item_list() {
                        if a.has_name(sym::proc_macro_derive)
                            || (a.has_name(sym::inline) && attr::list_contains_name(&meta_items, sym::always))
                        {
                            return;
                        }
                    }
                }
            },
            FnKind::Method(..) => (),
            FnKind::Closure(..) => return,
        }

        // Exclude non-inherent impls
        if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
            if matches!(
                item.kind,
                ItemKind::Impl { of_trait: Some(_), .. } | ItemKind::Trait(..)
            ) {
                return;
            }
        }

        self.check_poly_fn(cx, hir_id, decl, Some(span));
    }
}
