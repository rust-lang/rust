#![allow(print_stdout, use_debug)]

//! checks for attributes

use rustc::lint::*;
use rustc::hir;
use rustc::hir::print;
use syntax::ast::Attribute;
use syntax::attr;

/// **What it does:** Dumps every ast/hir node which has the `#[clippy_dump]`
/// attribute
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::Item) {
        if !has_attr(&item.attrs) {
            return;
        }
        print_item(cx, item);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::ImplItem) {
        if !has_attr(&item.attrs) {
            return;
        }
        println!("impl item `{}`", item.name);
        match item.vis {
            hir::Visibility::Public => println!("public"),
            hir::Visibility::Crate => println!("visible crate wide"),
            hir::Visibility::Restricted { ref path, .. } => println!(
                "visible in module `{}`",
                print::to_string(print::NO_ANN, |s| s.print_path(path, false))
            ),
            hir::Visibility::Inherited => println!("visibility inherited from outer item"),
        }
        if item.defaultness.is_default() {
            println!("default");
        }
        match item.node {
            hir::ImplItemKind::Const(_, body_id) => {
                println!("associated constant");
                print_expr(cx, &cx.tcx.hir.body(body_id).value, 1);
            },
            hir::ImplItemKind::Method(..) => println!("method"),
            hir::ImplItemKind::Type(_) => println!("associated type"),
        }
    }
    // fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx
    // hir::TraitItem) {
    // if !has_attr(&item.attrs) {
    // return;
    // }
    // }
    //
    // fn check_variant(&mut self, cx: &LateContext<'a, 'tcx>, var: &'tcx
    // hir::Variant, _:
    // &hir::Generics) {
    // if !has_attr(&var.node.attrs) {
    // return;
    // }
    // }
    //
    // fn check_struct_field(&mut self, cx: &LateContext<'a, 'tcx>, field: &'tcx
    // hir::StructField) {
    // if !has_attr(&field.attrs) {
    // return;
    // }
    // }
    //

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if !has_attr(&expr.attrs) {
            return;
        }
        print_expr(cx, expr, 0);
    }

    fn check_arm(&mut self, cx: &LateContext<'a, 'tcx>, arm: &'tcx hir::Arm) {
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

    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx hir::Stmt) {
        if !has_attr(stmt.node.attrs()) {
            return;
        }
        match stmt.node {
            hir::StmtDecl(ref decl, _) => print_decl(cx, decl),
            hir::StmtExpr(ref e, _) | hir::StmtSemi(ref e, _) => print_expr(cx, e, 0),
        }
    }
    // fn check_foreign_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx
    // hir::ForeignItem) {
    // if !has_attr(&item.attrs) {
    // return;
    // }
    // }
    //
}

fn has_attr(attrs: &[Attribute]) -> bool {
    attr::contains_name(attrs, "clippy_dump")
}

fn print_decl(cx: &LateContext, decl: &hir::Decl) {
    match decl.node {
        hir::DeclLocal(ref local) => {
            println!("local variable of type {}", cx.tables.node_id_to_type(local.hir_id));
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
    println!("{}+", ind);
    println!("{}ty: {}", ind, cx.tables.expr_ty(expr));
    println!("{}adjustments: {:?}", ind, cx.tables.adjustments().get(expr.hir_id));
    match expr.node {
        hir::ExprBox(ref e) => {
            println!("{}Box", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprArray(ref v) => {
            println!("{}Array", ind);
            for e in v {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprCall(ref func, ref args) => {
            println!("{}Call", ind);
            println!("{}function:", ind);
            print_expr(cx, func, indent + 1);
            println!("{}arguments:", ind);
            for arg in args {
                print_expr(cx, arg, indent + 1);
            }
        },
        hir::ExprMethodCall(ref path, _, ref args) => {
            println!("{}MethodCall", ind);
            println!("{}method name: {}", ind, path.name);
            for arg in args {
                print_expr(cx, arg, indent + 1);
            }
        },
        hir::ExprTup(ref v) => {
            println!("{}Tup", ind);
            for e in v {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprBinary(op, ref lhs, ref rhs) => {
            println!("{}Binary", ind);
            println!("{}op: {:?}", ind, op.node);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprUnary(op, ref inner) => {
            println!("{}Unary", ind);
            println!("{}op: {:?}", ind, op);
            print_expr(cx, inner, indent + 1);
        },
        hir::ExprLit(ref lit) => {
            println!("{}Lit", ind);
            println!("{}{:?}", ind, lit);
        },
        hir::ExprCast(ref e, ref target) => {
            println!("{}Cast", ind);
            print_expr(cx, e, indent + 1);
            println!("{}target type: {:?}", ind, target);
        },
        hir::ExprType(ref e, ref target) => {
            println!("{}Type", ind);
            print_expr(cx, e, indent + 1);
            println!("{}target type: {:?}", ind, target);
        },
        hir::ExprIf(ref e, _, ref els) => {
            println!("{}If", ind);
            println!("{}condition:", ind);
            print_expr(cx, e, indent + 1);
            if let Some(ref els) = *els {
                println!("{}else:", ind);
                print_expr(cx, els, indent + 1);
            }
        },
        hir::ExprWhile(ref cond, _, _) => {
            println!("{}While", ind);
            println!("{}condition:", ind);
            print_expr(cx, cond, indent + 1);
        },
        hir::ExprLoop(..) => {
            println!("{}Loop", ind);
        },
        hir::ExprMatch(ref cond, _, ref source) => {
            println!("{}Match", ind);
            println!("{}condition:", ind);
            print_expr(cx, cond, indent + 1);
            println!("{}source: {:?}", ind, source);
        },
        hir::ExprClosure(ref clause, _, _, _, _) => {
            println!("{}Closure", ind);
            println!("{}clause: {:?}", ind, clause);
        },
        hir::ExprYield(ref sub) => {
            println!("{}Yield", ind);
            print_expr(cx, sub, indent + 1);
        },
        hir::ExprBlock(_) => {
            println!("{}Block", ind);
        },
        hir::ExprAssign(ref lhs, ref rhs) => {
            println!("{}Assign", ind);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprAssignOp(ref binop, ref lhs, ref rhs) => {
            println!("{}AssignOp", ind);
            println!("{}op: {:?}", ind, binop.node);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprField(ref e, ref name) => {
            println!("{}Field", ind);
            println!("{}field name: {}", ind, name.node);
            println!("{}struct expr:", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprTupField(ref e, ref idx) => {
            println!("{}TupField", ind);
            println!("{}field index: {}", ind, idx.node);
            println!("{}tuple expr:", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprIndex(ref arr, ref idx) => {
            println!("{}Index", ind);
            println!("{}array expr:", ind);
            print_expr(cx, arr, indent + 1);
            println!("{}index expr:", ind);
            print_expr(cx, idx, indent + 1);
        },
        hir::ExprPath(hir::QPath::Resolved(ref ty, ref path)) => {
            println!("{}Resolved Path, {:?}", ind, ty);
            println!("{}path: {:?}", ind, path);
        },
        hir::ExprPath(hir::QPath::TypeRelative(ref ty, ref seg)) => {
            println!("{}Relative Path, {:?}", ind, ty);
            println!("{}seg: {:?}", ind, seg);
        },
        hir::ExprAddrOf(ref muta, ref e) => {
            println!("{}AddrOf", ind);
            println!("mutability: {:?}", muta);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprBreak(_, ref e) => {
            println!("{}Break", ind);
            if let Some(ref e) = *e {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprAgain(_) => println!("{}Again", ind),
        hir::ExprRet(ref e) => {
            println!("{}Ret", ind);
            if let Some(ref e) = *e {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprInlineAsm(_, ref input, ref output) => {
            println!("{}InlineAsm", ind);
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
            println!("{}Struct", ind);
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
        hir::ExprRepeat(ref val, body_id) => {
            println!("{}Repeat", ind);
            println!("{}value:", ind);
            print_expr(cx, val, indent + 1);
            println!("{}repeat count:", ind);
            print_expr(cx, &cx.tcx.hir.body(body_id).value, indent + 1);
        },
    }
}

fn print_item(cx: &LateContext, item: &hir::Item) {
    let did = cx.tcx.hir.local_def_id(item.id);
    println!("item `{}`", item.name);
    match item.vis {
        hir::Visibility::Public => println!("public"),
        hir::Visibility::Crate => println!("visible crate wide"),
        hir::Visibility::Restricted { ref path, .. } => println!(
            "visible in module `{}`",
            print::to_string(print::NO_ANN, |s| s.print_path(path, false))
        ),
        hir::Visibility::Inherited => println!("visibility inherited from outer item"),
    }
    match item.node {
        hir::ItemExternCrate(ref _renamed_from) => {
            if let Some(crate_id) = cx.tcx.extern_mod_stmt_cnum(cx.tcx.hir.node_to_hir_id(item.id)) {
                let source = cx.tcx.used_crate_source(crate_id);
                if let Some(ref src) = source.dylib {
                    println!("extern crate dylib source: {:?}", src.0);
                }
                if let Some(ref src) = source.rlib {
                    println!("extern crate rlib source: {:?}", src.0);
                }
            } else {
                println!("weird extern crate without a crate id");
            }
        },
        hir::ItemUse(ref path, ref kind) => println!("{:?}, {:?}", path, kind),
        hir::ItemStatic(..) => println!("static item of type {:#?}", cx.tcx.type_of(did)),
        hir::ItemConst(..) => println!("const item of type {:#?}", cx.tcx.type_of(did)),
        hir::ItemFn(..) => {
            let item_ty = cx.tcx.type_of(did);
            println!("function of type {:#?}", item_ty);
        },
        hir::ItemMod(..) => println!("module"),
        hir::ItemForeignMod(ref fm) => println!("foreign module with abi: {}", fm.abi),
        hir::ItemGlobalAsm(ref asm) => println!("global asm: {:?}", asm),
        hir::ItemTy(..) => {
            println!("type alias for {:?}", cx.tcx.type_of(did));
        },
        hir::ItemEnum(..) => {
            println!("enum definition of type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemStruct(..) => {
            println!("struct definition of type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemUnion(..) => {
            println!("union definition of type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemTrait(..) => {
            println!("trait decl");
            if cx.tcx.trait_has_default_impl(did) {
                println!("trait has a default impl");
            } else {
                println!("trait has no default impl");
            }
        },
        hir::ItemDefaultImpl(_, ref _trait_ref) => {
            println!("default impl");
        },
        hir::ItemImpl(_, _, _, _, Some(ref _trait_ref), _, _) => {
            println!("trait impl");
        },
        hir::ItemImpl(_, _, _, _, None, _, _) => {
            println!("impl");
        },
    }
}

fn print_pat(cx: &LateContext, pat: &hir::Pat, indent: usize) {
    let ind = "  ".repeat(indent);
    println!("{}+", ind);
    match pat.node {
        hir::PatKind::Wild => println!("{}Wild", ind),
        hir::PatKind::Binding(ref mode, _, ref name, ref inner) => {
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
            println!(
                "{}name: {}",
                ind,
                print::to_string(print::NO_ANN, |s| s.print_qpath(path, false))
            );
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
            println!(
                "{}path: {}",
                ind,
                print::to_string(print::NO_ANN, |s| s.print_qpath(path, false))
            );
            if let Some(dot_position) = opt_dots_position {
                println!("{}dot position: {}", ind, dot_position);
            }
            for field in fields {
                print_pat(cx, field, indent + 1);
            }
        },
        hir::PatKind::Path(hir::QPath::Resolved(ref ty, ref path)) => {
            println!("{}Resolved Path, {:?}", ind, ty);
            println!("{}path: {:?}", ind, path);
        },
        hir::PatKind::Path(hir::QPath::TypeRelative(ref ty, ref seg)) => {
            println!("{}Relative Path, {:?}", ind, ty);
            println!("{}seg: {:?}", ind, seg);
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
        hir::PatKind::Range(ref l, ref r, ref range_end) => {
            println!("{}Range", ind);
            print_expr(cx, l, indent + 1);
            print_expr(cx, r, indent + 1);
            match *range_end {
                hir::RangeEnd::Included => println!("{} end included", ind),
                hir::RangeEnd::Excluded => println!("{} end excluded", ind),
            }
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
