#![allow(print_stdout, use_debug)]

//! checks for attributes

use rustc::lint::*;
use rustc::hir;
use syntax::ast::{Attribute, MetaItemKind};

/// **What it does:** Dumps every ast/hir node which has the `#[clippy_dump]` attribute
///
/// **Example:**
/// ```rust
/// #[clippy_dump]
/// extern crate foo;
/// ```
///
/// prints
///
/// ```
/// item `foo`
/// visibility inherited from outer item
/// extern crate dylib source: "/path/to/foo.so"
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

impl LateLintPass for Pass {
    fn check_item(&mut self, cx: &LateContext, item: &hir::Item) {
        if !has_attr(&item.attrs) {
            return;
        }
        print_item(cx, item);
    }

    fn check_impl_item(&mut self, cx: &LateContext, item: &hir::ImplItem) {
        if !has_attr(&item.attrs) {
            return;
        }
        println!("impl item `{}`", item.name);
        match item.vis {
            hir::Visibility::Public => println!("public"),
            hir::Visibility::Crate => println!("visible crate wide"),
            hir::Visibility::Restricted { ref path, .. } => println!("visible in module `{}`", path),
            hir::Visibility::Inherited => println!("visibility inherited from outer item"),
        }
        if item.defaultness.is_default() {
            println!("default");
        }
        match item.node {
            hir::ImplItemKind::Const(_, ref e) => {
                println!("associated constant");
                print_expr(cx, e, 1);
            },
            hir::ImplItemKind::Method(..) => println!("method"),
            hir::ImplItemKind::Type(_) => println!("associated type"),
        }
    }
/*
    fn check_trait_item(&mut self, cx: &LateContext, item: &hir::TraitItem) {
        if !has_attr(&item.attrs) {
            return;
        }
    }

    fn check_variant(&mut self, cx: &LateContext, var: &hir::Variant, _: &hir::Generics) {
        if !has_attr(&var.node.attrs) {
            return;
        }
    }

    fn check_struct_field(&mut self, cx: &LateContext, field: &hir::StructField) {
        if !has_attr(&field.attrs) {
            return;
        }
    }
*/

    fn check_expr(&mut self, cx: &LateContext, expr: &hir::Expr) {
        if !has_attr(&expr.attrs) {
            return;
        }
        print_expr(cx, expr, 0);
    }

    fn check_arm(&mut self, cx: &LateContext, arm: &hir::Arm) {
        if !has_attr(&arm.attrs) {
            return;
        }
        for pat in &arm.pats {
            print_pat(cx, pat, 1);
        }
        if let Some(ref guard) = arm.guard {
            println!("guard:");
            print_expr(cx, guard, 1);
        }
        println!("body:");
        print_expr(cx, &arm.body, 1);
    }

    fn check_stmt(&mut self, cx: &LateContext, stmt: &hir::Stmt) {
        if !has_attr(stmt.node.attrs()) {
            return;
        }
        match stmt.node {
            hir::StmtDecl(ref decl, _) => print_decl(cx, decl),
            hir::StmtExpr(ref e, _) | hir::StmtSemi(ref e, _) => print_expr(cx, e, 0),
        }
    }
/*

    fn check_foreign_item(&mut self, cx: &LateContext, item: &hir::ForeignItem) {
        if !has_attr(&item.attrs) {
            return;
        }
    }
*/
}

fn has_attr(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| match attr.node.value.node {
        MetaItemKind::Word(ref word) => word == "clippy_dump",
        _ => false,
    })
}

fn print_decl(cx: &LateContext, decl: &hir::Decl) {
    match decl.node {
        hir::DeclLocal(ref local) => {
            println!("local variable of type {}", cx.tcx.node_id_to_type(local.id));
            println!("pattern:");
            print_pat(cx, &local.pat, 0);
            if let Some(ref e) = local.init {
                println!("init expression:");
                print_expr(cx, e, 0);
            }
        },
        hir::DeclItem(_) => println!("item decl"),
    }
}

fn print_expr(cx: &LateContext, expr: &hir::Expr, indent: usize) {
    let ind = "  ".repeat(indent);
    let ty = cx.tcx.node_id_to_type(expr.id);
    println!("{}+", ind);
    match expr.node {
        hir::ExprBox(ref e) => {
            println!("{}Box, {}", ind, ty);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprArray(ref v) => {
            println!("{}Array, {}", ind, ty);
            for e in v {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprCall(ref func, ref args) => {
            println!("{}Call, {}", ind, ty);
            println!("{}function:", ind);
            print_expr(cx, func, indent + 1);
            println!("{}arguments:", ind);
            for arg in args {
                print_expr(cx, arg, indent + 1);
            }
        },
        hir::ExprMethodCall(ref name, _, ref args) => {
            println!("{}MethodCall, {}", ind, ty);
            println!("{}method name: {}", ind, name.node);
            for arg in args {
                print_expr(cx, arg, indent + 1);
            }
        },
        hir::ExprTup(ref v) => {
            println!("{}Tup, {}", ind, ty);
            for e in v {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprBinary(op, ref lhs, ref rhs) => {
            println!("{}Binary, {}", ind, ty);
            println!("{}op: {:?}", ind, op.node);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprUnary(op, ref inner) => {
            println!("{}Unary, {}", ind, ty);
            println!("{}op: {:?}", ind, op);
            print_expr(cx, inner, indent + 1);
        },
        hir::ExprLit(ref lit) => {
            println!("{}Lit, {}", ind, ty);
            println!("{}{:?}", ind, lit);
        },
        hir::ExprCast(ref e, ref target) => {
            println!("{}Cast, {}", ind, ty);
            print_expr(cx, e, indent + 1);
            println!("{}target type: {:?}", ind, target);
        },
        hir::ExprType(ref e, ref target) => {
            println!("{}Type, {}", ind, ty);
            print_expr(cx, e, indent + 1);
            println!("{}target type: {:?}", ind, target);
        },
        hir::ExprIf(ref e, _, ref els) => {
            println!("{}If, {}", ind, ty);
            println!("{}condition:", ind);
            print_expr(cx, e, indent + 1);
            if let Some(ref els) = *els {
                println!("{}else:", ind);
                print_expr(cx, els, indent + 1);
            }
        },
        hir::ExprWhile(ref cond, _, _) => {
            println!("{}While, {}", ind, ty);
            println!("{}condition:", ind);
            print_expr(cx, cond, indent + 1);
        },
        hir::ExprLoop(..) => {
            println!("{}Loop, {}", ind, ty);
        },
        hir::ExprMatch(ref cond, _, ref source) => {
            println!("{}Match, {}", ind, ty);
            println!("{}condition:", ind);
            print_expr(cx, cond, indent + 1);
            println!("{}source: {:?}", ind, source);
        },
        hir::ExprClosure(ref clause, _, _, _) => {
            println!("{}Closure, {}", ind, ty);
            println!("{}clause: {:?}", ind, clause);
        },
        hir::ExprBlock(_) => {
            println!("{}Block, {}", ind, ty);
        },
        hir::ExprAssign(ref lhs, ref rhs) => {
            println!("{}Assign, {}", ind, ty);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprAssignOp(ref binop, ref lhs, ref rhs) => {
            println!("{}AssignOp, {}", ind, ty);
            println!("{}op: {:?}", ind, binop.node);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprField(ref e, ref name) => {
            println!("{}Field, {}", ind, ty);
            println!("{}field name: {}", ind, name.node);
            println!("{}struct expr:", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprTupField(ref e, ref idx) => {
            println!("{}TupField, {}", ind, ty);
            println!("{}field index: {}", ind, idx.node);
            println!("{}tuple expr:", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprIndex(ref arr, ref idx) => {
            println!("{}Index, {}", ind, ty);
            println!("{}array expr:", ind);
            print_expr(cx, arr, indent + 1);
            println!("{}index expr:", ind);
            print_expr(cx, idx, indent + 1);
        },
        hir::ExprPath(ref sel, ref path) => {
            println!("{}Path, {}", ind, ty);
            println!("{}self: {:?}", ind, sel);
            println!("{}path: {:?}", ind, path);
        },
        hir::ExprAddrOf(ref muta, ref e) => {
            println!("{}AddrOf, {}", ind, ty);
            println!("mutability: {:?}", muta);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprBreak(_) => println!("{}Break, {}", ind, ty),
        hir::ExprAgain(_) => println!("{}Again, {}", ind, ty),
        hir::ExprRet(ref e) => {
            println!("{}Ret, {}", ind, ty);
            if let Some(ref e) = *e {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprInlineAsm(_, ref input, ref output) => {
            println!("{}InlineAsm, {}", ind, ty);
            println!("{}inputs:", ind);
            for e in input {
                print_expr(cx, e, indent + 1);
            }
            println!("{}outputs:", ind);
            for e in output {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprStruct(ref path, ref fields, ref base) => {
            println!("{}Struct, {}", ind, ty);
            println!("{}path: {:?}", ind, path);
            for field in fields {
                println!("{}field \"{}\":", ind, field.name.node);
                print_expr(cx, &field.expr, indent + 1);
            }
            if let Some(ref base) = *base {
                println!("{}base:", ind);
                print_expr(cx, base, indent + 1);
            }
        },
        hir::ExprRepeat(ref val, ref n) => {
            println!("{}Repeat, {}", ind, ty);
            println!("{}value:", ind);
            print_expr(cx, val, indent + 1);
            println!("{}repeat count:", ind);
            print_expr(cx, n, indent + 1);
        },
    }
}

fn print_item(cx: &LateContext, item: &hir::Item) {
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

fn print_pat(cx: &LateContext, pat: &hir::Pat, indent: usize) {
    let ind = "  ".repeat(indent);
    println!("{}+", ind);
    match pat.node {
        hir::PatKind::Wild => println!("{}Wild", ind),
        hir::PatKind::Binding(ref mode, ref name, ref inner) => {
            println!("{}Binding", ind);
            println!("{}mode: {:?}", ind, mode);
            println!("{}name: {}", ind, name.node);
            if let Some(ref inner) = *inner {
                println!("{}inner:", ind);
                print_pat(cx, inner, indent + 1);
            }
        },
        hir::PatKind::Struct(ref path, ref fields, ignore) => {
            println!("{}Struct", ind);
            println!("{}name: {}", ind, path);
            println!("{}ignore leftover fields: {}", ind, ignore);
            println!("{}fields:", ind);
            for field in fields {
                println!("{}  field name: {}", ind, field.node.name);
                if field.node.is_shorthand {
                    println!("{}  in shorthand notation", ind);
                }
                print_pat(cx, &field.node.pat, indent + 1);
            }
        },
        hir::PatKind::TupleStruct(ref path, ref fields, opt_dots_position) => {
            println!("{}TupleStruct", ind);
            println!("{}path: {}", ind, path);
            if let Some(dot_position) = opt_dots_position {
                println!("{}dot position: {}", ind, dot_position);
            }
            for field in fields {
                print_pat(cx, field, indent + 1);
            }
        },
        hir::PatKind::Path(ref sel, ref path) => {
            println!("{}Path", ind);
            println!("{}self: {:?}", ind, sel);
            println!("{}path: {:?}", ind, path);
        },
        hir::PatKind::Tuple(ref pats, opt_dots_position) => {
            println!("{}Tuple", ind);
            if let Some(dot_position) = opt_dots_position {
                println!("{}dot position: {}", ind, dot_position);
            }
            for field in pats {
                print_pat(cx, field, indent + 1);
            }
        },
        hir::PatKind::Box(ref inner) => {
            println!("{}Box", ind);
            print_pat(cx, inner, indent + 1);
        },
        hir::PatKind::Ref(ref inner, ref muta) => {
            println!("{}Ref", ind);
            println!("{}mutability: {:?}", ind, muta);
            print_pat(cx, inner, indent + 1);
        },
        hir::PatKind::Lit(ref e) => {
            println!("{}Lit", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::PatKind::Range(ref l, ref r) => {
            println!("{}Range", ind);
            print_expr(cx, l, indent + 1);
            print_expr(cx, r, indent + 1);
        },
        hir::PatKind::Slice(ref first_pats, ref range, ref last_pats) => {
            println!("{}Slice [a, b, ..i, y, z]", ind);
            println!("[a, b]:");
            for pat in first_pats {
                print_pat(cx, pat, indent + 1);
            }
            println!("i:");
            if let Some(ref pat) = *range {
                print_pat(cx, pat, indent + 1);
            }
            println!("[y, z]:");
            for pat in last_pats {
                print_pat(cx, pat, indent + 1);
            }
        },
    }
}
