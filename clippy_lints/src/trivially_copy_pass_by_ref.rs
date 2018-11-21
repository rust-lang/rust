// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::cmp;

use matches::matches;
use crate::rustc::hir;
use crate::rustc::hir::*;
use crate::rustc::hir::intravisit::FnKind;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use if_chain::if_chain;
use crate::rustc::ty::TyKind;
use crate::rustc::ty::FnSig;
use crate::rustc::session::config::Config as SessionConfig;
use crate::rustc_target::spec::abi::Abi;
use crate::rustc_target::abi::LayoutOf;
use crate::syntax::ast::NodeId;
use crate::syntax_pos::Span;
use crate::utils::{in_macro, is_copy, is_self_ty, span_lint_and_sugg, snippet};

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
/// ```rust
/// fn foo(v: &u32) {
///     assert_eq!(v, 42);
/// }
/// // should be
/// fn foo(v: u32) {
///     assert_eq!(v, 42);
/// }
/// ```
declare_clippy_lint! {
    pub TRIVIALLY_COPY_PASS_BY_REF,
    perf,
    "functions taking small copyable arguments by reference"
}

pub struct TriviallyCopyPassByRef {
    limit: u64,
}

impl<'a, 'tcx> TriviallyCopyPassByRef {
    pub fn new(limit: Option<u64>, target: &SessionConfig) -> Self {
        let limit = limit.unwrap_or_else(|| {
            let bit_width = target.usize_ty.bit_width().expect("usize should have a width") as u64;
            // Cap the calculated bit width at 32-bits to reduce
            // portability problems between 32 and 64-bit targets
            let bit_width = cmp::min(bit_width, 32);
            let byte_width = bit_width / 8;
            // Use a limit of 2 times the register bit width
            byte_width * 2
        });
        Self { limit }
    }

    fn check_trait_method(
        &mut self,
        cx: &LateContext<'_, 'tcx>,
        item: &TraitItemRef
    ) {
        let method_def_id = cx.tcx.hir.local_def_id(item.id.node_id);
        let method_sig = cx.tcx.fn_sig(method_def_id);
        let method_sig = cx.tcx.erase_late_bound_regions(&method_sig);

        let decl = match cx.tcx.hir.fn_decl(item.id.node_id) {
            Some(b) => b,
            None => return,
        };

        self.check_poly_fn(cx, &decl, &method_sig, None);
    }

    fn check_poly_fn(
        &mut self,
        cx: &LateContext<'_, 'tcx>,
        decl: &FnDecl,
        sig: &FnSig<'tcx>,
        span: Option<Span>,
    ) {
        // Use lifetimes to determine if we're returning a reference to the
        // argument. In that case we can't switch to pass-by-value as the
        // argument will not live long enough.
        let output_lts = match sig.output().sty {
            TyKind::Ref(output_lt, _, _) => vec![output_lt],
            TyKind::Adt(_, substs) => substs.regions().collect(),
            _ => vec![],
        };

        for (input, &ty) in decl.inputs.iter().zip(sig.inputs()) {
            // All spans generated from a proc-macro invocation are the same...
            match span {
                Some(s) if s == input.span => return,
                _ => (),
            }

            if_chain! {
                if let TyKind::Ref(input_lt, ty, Mutability::MutImmutable) = ty.sty;
                if !output_lts.contains(&input_lt);
                if is_copy(cx, ty);
                if let Some(size) = cx.layout_of(ty).ok().map(|l| l.size.bytes());
                if size <= self.limit;
                if let hir::TyKind::Rptr(_, MutTy { ty: ref decl_ty, .. }) = input.node;
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
                        "this argument is passed by reference, but would be more efficient if passed by value",
                        "consider passing by value instead",
                        value_type);
                }
            }
        }
    }

    fn check_trait_items(
        &mut self,
        cx: &LateContext<'_, '_>,
        trait_items: &[TraitItemRef]
    ) {
        for item in trait_items {
            if let AssociatedItemKind::Method{..} = item.kind {
                self.check_trait_method(cx, item);
            }
        }
    }
}

impl LintPass for TriviallyCopyPassByRef {
    fn get_lints(&self) -> LintArray {
        lint_array![TRIVIALLY_COPY_PASS_BY_REF]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TriviallyCopyPassByRef {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if in_macro(item.span) {
            return;
        }
        if let ItemKind::Trait(_, _, _, _, ref trait_items) = item.node {
            self.check_trait_items(cx, trait_items);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        _body: &'tcx Body,
        span: Span,
        node_id: NodeId,
    ) {
        if in_macro(span) {
            return;
        }

        match kind {
            FnKind::ItemFn(.., header, _, attrs) => {
                if header.abi != Abi::Rust {
                    return;
                }
                for a in attrs {
                    if a.meta_item_list().is_some() && a.name() == "proc_macro_derive" {
                        return;
                    }
                }
            },
            FnKind::Method(..) => (),
            _ => return,
        }

        // Exclude non-inherent impls
        if let Some(Node::Item(item)) = cx.tcx.hir.find(cx.tcx.hir.get_parent_node(node_id)) {
            if matches!(item.node, ItemKind::Impl(_, _, _, _, Some(_), _, _) |
                ItemKind::Trait(..))
            {
                return;
            }
        }

        let fn_def_id = cx.tcx.hir.local_def_id(node_id);

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(&fn_sig);

        self.check_poly_fn(cx, decl, &fn_sig, Some(span));
    }
}
