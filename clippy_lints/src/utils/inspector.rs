//! checks for attributes

use rustc::lint::*;
use rustc::hir;
use syntax::ast::{Attribute, MetaItemKind};

/// **What it does:** Dumps every ast/hir node which has the `#[inspect]` attribute
///
/// **Why is this bad?** 😈
///
/// **Known problems:** ∅
///
/// **Example:**
/// ```rust
/// #[inspect]
/// extern crate foo;
/// ```
declare_lint! {
    pub DEEP_CODE_INSPECTION,
    Warn,
    "helper to dump info about code"
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DEEP_CODE_INSPECTION)
    }
}

#[allow(print_stdout, use_debug)]
impl LateLintPass for Pass {
    fn check_item(&mut self, cx: &LateContext, item: &hir::Item) {
        if !has_inspect_attr(&item.attrs) {
            return;
        }
        let did = cx.tcx.map.local_def_id(item.id);
        println!("item `{}`", item.name);
        match item.vis {
            hir::Visibility::Public => println!("public"),
            hir::Visibility::Crate => println!("visible crate wide"),
            hir::Visibility::Restricted { ref path, .. } => println!("visible in module `{}`", path),
            hir::Visibility::Inherited => println!("visibility inherited from outer item"),
        }
        match item.node {
            hir::ItemExternCrate(ref _renamed_from) => {
                if let Some(crate_id) = cx.tcx.sess.cstore.extern_mod_stmt_cnum(item.id) {
                    let source = cx.tcx.sess.cstore.used_crate_source(crate_id);
                    if let Some(src) = source.dylib {
                        println!("extern crate dylib source: {:?}", src.0);
                    }
                    if let Some(src) = source.rlib {
                        println!("extern crate rlib source: {:?}", src.0);
                    }
                } else {
                    println!("weird extern crate without a crate id");
                }
            }
            hir::ItemUse(ref path) => println!("{:?}", path.node),
            hir::ItemStatic(..) => println!("static item: {:#?}", cx.tcx.opt_lookup_item_type(did)),
            hir::ItemConst(..) => println!("const item: {:#?}", cx.tcx.opt_lookup_item_type(did)),
            hir::ItemFn(..) => {
                let item_ty = cx.tcx.opt_lookup_item_type(did);
                println!("function: {:#?}", item_ty);
            },
            hir::ItemMod(..) => println!("module"),
            hir::ItemForeignMod(ref fm) => println!("foreign module with abi: {}", fm.abi),
            hir::ItemTy(..) => {
                println!("type alias: {:?}", cx.tcx.opt_lookup_item_type(did));
            },
            hir::ItemEnum(..) => {
                println!("enum definition: {:?}", cx.tcx.opt_lookup_item_type(did));
            },
            hir::ItemStruct(..) => {
                println!("struct definition: {:?}", cx.tcx.opt_lookup_item_type(did));
            },
            hir::ItemUnion(..) => {
                println!("union definition: {:?}", cx.tcx.opt_lookup_item_type(did));
            },
            hir::ItemTrait(..) => {
                println!("trait decl");
                if cx.tcx.trait_has_default_impl(did) {
                    println!("trait has a default impl");
                } else {
                    println!("trait has no default impl");
                }
            },
            hir::ItemDefaultImpl(_, ref trait_ref) => {
                let trait_did = cx.tcx.map.local_def_id(trait_ref.ref_id);
                println!("default impl for `{:?}`", cx.tcx.item_path_str(trait_did));
            },
            hir::ItemImpl(_, _, _, Some(ref trait_ref), _, _) => {
                let trait_did = cx.tcx.map.local_def_id(trait_ref.ref_id);
                println!("impl of trait `{:?}`", cx.tcx.item_path_str(trait_did));
            },
            hir::ItemImpl(_, _, _, None, _, _) => {
                println!("impl");
            },
        }
    }

/*
    fn check_impl_item(&mut self, cx: &LateContext, item: &hir::ImplItem) {
        if !has_inspect_attr(&item.attrs) {
            return;
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &hir::TraitItem) {
        if !has_inspect_attr(&item.attrs) {
            return;
        }
    }

    fn check_variant(&mut self, cx: &LateContext, var: &hir::Variant, _: &hir::Generics) {
        if !has_inspect_attr(&var.node.attrs) {
            return;
        }
    }

    fn check_struct_field(&mut self, cx: &LateContext, field: &hir::StructField) {
        if !has_inspect_attr(&field.attrs) {
            return;
        }
    }
*/

    fn check_expr(&mut self, cx: &LateContext, expr: &hir::Expr) {
        if !has_inspect_attr(&expr.attrs) {
            return;
        }
        println!("expression type: {}", cx.tcx.node_id_to_type(expr.id));
    }

    fn check_decl(&mut self, cx: &LateContext, decl: &hir::Decl) {
        if !has_inspect_attr(decl.node.attrs()) {
            return;
        }
        match decl.node {
            hir::DeclLocal(ref local) => {
                println!("local variable of type {}", cx.tcx.node_id_to_type(local.id));
            },
            hir::DeclItem(_) => println!("item decl"),
        }
    }
/*
    fn check_arm(&mut self, cx: &LateContext, arm: &hir::Arm) {
        if !has_inspect_attr(&arm.attrs) {
            return;
        }
    }

    fn check_stmt(&mut self, cx: &LateContext, stmt: &hir::Stmt) {
        if !has_inspect_attr(stmt.node.attrs()) {
            return;
        }
    }

    fn check_local(&mut self, cx: &LateContext, local: &hir::Local) {
        if !has_inspect_attr(&local.attrs) {
            return;
        }
    }

    fn check_foreign_item(&mut self, cx: &LateContext, item: &hir::ForeignItem) {
        if !has_inspect_attr(&item.attrs) {
            return;
        }
    }
*/
}

fn has_inspect_attr(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| match attr.node.value.node {
        MetaItemKind::Word(ref word) => word == "inspect",
        _ => false,
    })
}
