use std::cmp;

use crate::utils::{is_copy, is_self_ty, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId, ItemKind, MutTy, Mutability, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::config::Config as SessionConfig;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use rustc_target::abi::LayoutOf;
use rustc_target::spec::abi::Abi;

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

#[derive(Copy, Clone)]
pub struct TriviallyCopyPassByRef {
    limit: u64,
}

impl<'a, 'tcx> TriviallyCopyPassByRef {
    pub fn new(limit: Option<u64>, target: &SessionConfig) -> Self {
        let limit = limit.unwrap_or_else(|| {
            let bit_width = u64::from(target.ptr_width);
            // Cap the calculated bit width at 32-bits to reduce
            // portability problems between 32 and 64-bit targets
            let bit_width = cmp::min(bit_width, 32);
            #[allow(clippy::integer_division)]
            let byte_width = bit_width / 8;
            // Use a limit of 2 times the register byte width
            byte_width * 2
        });
        Self { limit }
    }

    fn check_poly_fn(&mut self, cx: &LateContext<'_, 'tcx>, hir_id: HirId, decl: &FnDecl<'_>, span: Option<Span>) {
        let fn_def_id = cx.tcx.hir().local_def_id(hir_id);

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(&fn_sig);

        // Use lifetimes to determine if we're returning a reference to the
        // argument. In that case we can't switch to pass-by-value as the
        // argument will not live long enough.
        let output_lts = match fn_sig.output().kind {
            ty::Ref(output_lt, _, _) => vec![output_lt],
            ty::Adt(_, substs) => substs.regions().collect(),
            _ => vec![],
        };

        for (input, &ty) in decl.inputs.iter().zip(fn_sig.inputs()) {
            // All spans generated from a proc-macro invocation are the same...
            match span {
                Some(s) if s == input.span => return,
                _ => (),
            }

            if_chain! {
                if let ty::Ref(input_lt, ty, Mutability::Not) = ty.kind;
                if !output_lts.contains(&input_lt);
                if is_copy(cx, ty);
                if let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes());
                if size <= self.limit;
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
                        &format!("this argument ({} byte) is passed by reference, but would be more efficient if passed by value (limit: {} byte)", size, self.limit),
                        "consider passing by value instead",
                        value_type,
                        Applicability::Unspecified,
                    );
                }
            }
        }
    }
}

impl_lint_pass!(TriviallyCopyPassByRef => [TRIVIALLY_COPY_PASS_BY_REF]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TriviallyCopyPassByRef {
    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::TraitItem<'_>) {
        if item.span.from_expansion() {
            return;
        }

        if let hir::TraitItemKind::Fn(method_sig, _) = &item.kind {
            self.check_poly_fn(cx, item.hir_id, &*method_sig.decl, None);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
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
                    if a.meta_item_list().is_some() && a.check_name(sym!(proc_macro_derive)) {
                        return;
                    }
                }
            },
            FnKind::Method(..) => (),
            _ => return,
        }

        // Exclude non-inherent impls
        if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
            if matches!(item.kind, ItemKind::Impl{ of_trait: Some(_), .. } |
                ItemKind::Trait(..))
            {
                return;
            }
        }

        self.check_poly_fn(cx, hir_id, decl, Some(span));
    }
}
