//   Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
//   file at the top-level directory of this distribution and at
//   http://rust-lang.org/COPYRIGHT.
//
//   Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
//   http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
//   <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
//   option. This file may not be copied, modified, or distributed
//   except according to those terms.
//

use rustc::hir;
use rustc::lint::*;
use syntax::ast;
use syntax::codemap::Span;

/// **What it does:** it lints if an exported function, method, trait method with default impl,
/// or trait method impl is not `#[inline]`.
///
/// **Why is this bad?** In general, it is not. Functions can be inlined across
/// crates when that's profitable as long as any form of LTO is used. When LTO is disabled,
/// functions that are not `#[inline]` cannot be inlined across crates. Certain types of crates
/// might intend for most of the methods in their public API to be able to be inlined across
/// crates even when LTO is disabled. For these types of crates, enabling this lint might make sense.
/// It allows the crate to require all exported methods to be `#[inline]` by default, and then opt
/// out for specific methods where this might not make sense.
///
/// **Known problems:** None.
declare_clippy_lint! {
    pub MISSING_INLINE_IN_PUBLIC_ITEMS,
    restriction,
    "detects missing #[inline] attribute for public callables (functions, trait methods, methods...)"
}

pub struct MissingInline {}

impl ::std::default::Default for MissingInline {
    fn default() -> Self {
        Self::new()
    }
}

impl MissingInline {
    pub fn new() -> Self {
        Self {}
    }

    fn check_missing_inline_attrs(&self, cx: &LateContext,
                                  attrs: &[ast::Attribute], sp: Span, desc: &'static str) {
        // If we're building a test harness, FIXME: is this relevant?
        // if cx.sess().opts.test {
        //    return;
        // }

        let has_inline = attrs
            .iter()
            .any(|a| a.name() == "inline" );
        if !has_inline {
            cx.span_lint(
                MISSING_INLINE_IN_PUBLIC_ITEMS,
                sp,
                &format!("missing `#[inline]` for {}", desc),
            );
        }
    }
}

impl LintPass for MissingInline {
    fn get_lints(&self) -> LintArray {
        lint_array![MISSING_INLINE_IN_PUBLIC_ITEMS]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MissingInline {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, it: &'tcx hir::Item) {
        if !cx.access_levels.is_exported(it.id) {
            return;
        }
        match it.node {
            hir::ItemFn(..) => {
                // ignore main()
                if it.name == "main" {
                    let def_id = cx.tcx.hir.local_def_id(it.id);
                    let def_key = cx.tcx.hir.def_key(def_id);
                    if def_key.parent == Some(hir::def_id::CRATE_DEF_INDEX) {
                        return;
                    }
                }
                let desc = "a function";
                self.check_missing_inline_attrs(cx, &it.attrs, it.span, desc);
            },
            hir::ItemTrait(ref _is_auto, ref _unsafe, ref _generics,
                           ref _bounds, ref trait_items)  => {
                for tit in trait_items {
                    let tit_ = cx.tcx.hir.trait_item(tit.id);
                    match tit_.node {
                        hir::TraitItemKind::Const(..) |
                        hir::TraitItemKind::Type(..) => {},
                        hir::TraitItemKind::Method(..) => {
                            if tit.defaultness.has_value() {
                                // trait method with default body needs inline in case
                                // an impl is not provided
                                let desc = "a default trait method";
                                let item = cx.tcx.hir.expect_trait_item(tit.id.node_id);
                                self.check_missing_inline_attrs(cx, &item.attrs,
                                                                item.span, desc);
                            }
                        },
                    }
                }
            }
            hir::ItemConst(..) |
            hir::ItemEnum(..) |
            hir::ItemMod(..) |
            hir::ItemStatic(..) |
            hir::ItemStruct(..) |
            hir::ItemTraitAlias(..) |
            hir::ItemGlobalAsm(..) |
            hir::ItemTy(..) |
            hir::ItemUnion(..) |
            hir::ItemExistential(..) |
            hir::ItemExternCrate(..) |
            hir::ItemForeignMod(..) |
            hir::ItemImpl(..) |
            hir::ItemUse(..) => {},
        };
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, impl_item: &'tcx hir::ImplItem) {
        use rustc::ty::{TraitContainer, ImplContainer};

        // If the item being implemented is not exported, then we don't need #[inline]
        if !cx.access_levels.is_exported(impl_item.id) {
            return;
        }

        let def_id = cx.tcx.hir.local_def_id(impl_item.id);
        match cx.tcx.associated_item(def_id).container {
            TraitContainer(cid) => {
                let n = cx.tcx.hir.as_local_node_id(cid);
                if n.is_some() {
                    if !cx.access_levels.is_exported(n.unwrap()) {
                        // If a trait is being implemented for an item, and the
                        // trait is not exported, we don't need #[inline]
                        return;
                    }
                }
            },
            ImplContainer(cid) => {
                if cx.tcx.impl_trait_ref(cid).is_some() {
                    let trait_ref = cx.tcx.impl_trait_ref(cid).unwrap();
                    let n = cx.tcx.hir.as_local_node_id(trait_ref.def_id);
                    if n.is_some() {
                        if !cx.access_levels.is_exported(n.unwrap()) {
                            // If a trait is being implemented for an item, and the
                            // trait is not exported, we don't need #[inline]
                            return;
                        }
                    }
                }
            },
        }

        let desc = match impl_item.node {
            hir::ImplItemKind::Method(..) => "a method",
            hir::ImplItemKind::Const(..) |
            hir::ImplItemKind::Type(_) => return,
        };
        self.check_missing_inline_attrs(cx, &impl_item.attrs, impl_item.span, desc);
    }
}
