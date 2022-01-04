//! checks for attributes

use clippy_utils::get_attr;
use rustc_ast::ast::{Attribute, InlineAsmTemplatePiece};
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::Session;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Dumps every ast/hir node which has the `#[clippy::dump]`
    /// attribute
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[clippy::dump]
    /// extern crate foo;
    /// ```
    ///
    /// prints
    ///
    /// ```text
    /// item `foo`
    /// visibility inherited from outer item
    /// extern crate dylib source: "/path/to/foo.so"
    /// ```
    pub DEEP_CODE_INSPECTION,
    internal_warn,
    "helper to dump info about code"
}

declare_lint_pass!(DeepCodeInspector => [DEEP_CODE_INSPECTION]);

impl<'tcx> LateLintPass<'tcx> for DeepCodeInspector {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if !has_attr(cx.sess(), cx.tcx.hir().attrs(item.hir_id())) {
            return;
        }
        print_item(cx, item);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        if !has_attr(cx.sess(), cx.tcx.hir().attrs(item.hir_id())) {
            return;
        }
        println!("impl item `{}`", item.ident.name);
        match item.vis.node {
            hir::VisibilityKind::Public => println!("public"),
            hir::VisibilityKind::Crate(_) => println!("visible crate wide"),
            hir::VisibilityKind::Restricted { path, .. } => println!(
                "visible in module `{}`",
                rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_path(path, false))
            ),
            hir::VisibilityKind::Inherited => println!("visibility inherited from outer item"),
        }
        if item.defaultness.is_default() {
            println!("default");
        }
        match item.kind {
            hir::ImplItemKind::Const(_, body_id) => {
                println!("associated constant");
                print_expr(cx, &cx.tcx.hir().body(body_id).value, 1);
            },
            hir::ImplItemKind::Fn(..) => println!("method"),
            hir::ImplItemKind::TyAlias(_) => println!("associated type"),
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if !has_attr(cx.sess(), cx.tcx.hir().attrs(expr.hir_id)) {
            return;
        }
        print_expr(cx, expr, 0);
    }

    fn check_arm(&mut self, cx: &LateContext<'tcx>, arm: &'tcx hir::Arm<'_>) {
        if !has_attr(cx.sess(), cx.tcx.hir().attrs(arm.hir_id)) {
            return;
        }
        print_pat(cx, arm.pat, 1);
        if let Some(ref guard) = arm.guard {
            println!("guard:");
            print_guard(cx, guard, 1);
        }
        println!("body:");
        print_expr(cx, arm.body, 1);
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx hir::Stmt<'_>) {
        if !has_attr(cx.sess(), cx.tcx.hir().attrs(stmt.hir_id)) {
            return;
        }
        match stmt.kind {
            hir::StmtKind::Local(local) => {
                println!("local variable of type {}", cx.typeck_results().node_type(local.hir_id));
                println!("pattern:");
                print_pat(cx, local.pat, 0);
                if let Some(e) = local.init {
                    println!("init expression:");
                    print_expr(cx, e, 0);
                }
            },
            hir::StmtKind::Item(_) => println!("item decl"),
            hir::StmtKind::Expr(e) | hir::StmtKind::Semi(e) => print_expr(cx, e, 0),
        }
    }
}

fn has_attr(sess: &Session, attrs: &[Attribute]) -> bool {
    get_attr(sess, attrs, "dump").count() > 0
}

#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
fn print_expr(cx: &LateContext<'_>, expr: &hir::Expr<'_>, indent: usize) {
    let ind = "  ".repeat(indent);
    println!("{}+", ind);
    println!("{}ty: {}", ind, cx.typeck_results().expr_ty(expr));
    println!(
        "{}adjustments: {:?}",
        ind,
        cx.typeck_results().adjustments().get(expr.hir_id)
    );
    match expr.kind {
        hir::ExprKind::Box(e) => {
            println!("{}Box", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprKind::Array(v) => {
            println!("{}Array", ind);
            for e in v {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprKind::Call(func, args) => {
            println!("{}Call", ind);
            println!("{}function:", ind);
            print_expr(cx, func, indent + 1);
            println!("{}arguments:", ind);
            for arg in args {
                print_expr(cx, arg, indent + 1);
            }
        },
        hir::ExprKind::Let(hir::Let { pat, init, ty, .. }) => {
            print_pat(cx, pat, indent + 1);
            if let Some(ty) = ty {
                println!("{}  type annotation: {:?}", ind, ty);
            }
            print_expr(cx, init, indent + 1);
        },
        hir::ExprKind::MethodCall(path, _, args, _) => {
            println!("{}MethodCall", ind);
            println!("{}method name: {}", ind, path.ident.name);
            for arg in args {
                print_expr(cx, arg, indent + 1);
            }
        },
        hir::ExprKind::Tup(v) => {
            println!("{}Tup", ind);
            for e in v {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprKind::Binary(op, lhs, rhs) => {
            println!("{}Binary", ind);
            println!("{}op: {:?}", ind, op.node);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprKind::Unary(op, inner) => {
            println!("{}Unary", ind);
            println!("{}op: {:?}", ind, op);
            print_expr(cx, inner, indent + 1);
        },
        hir::ExprKind::Lit(ref lit) => {
            println!("{}Lit", ind);
            println!("{}{:?}", ind, lit);
        },
        hir::ExprKind::Cast(e, target) => {
            println!("{}Cast", ind);
            print_expr(cx, e, indent + 1);
            println!("{}target type: {:?}", ind, target);
        },
        hir::ExprKind::Type(e, target) => {
            println!("{}Type", ind);
            print_expr(cx, e, indent + 1);
            println!("{}target type: {:?}", ind, target);
        },
        hir::ExprKind::Loop(..) => {
            println!("{}Loop", ind);
        },
        hir::ExprKind::If(cond, _, ref else_opt) => {
            println!("{}If", ind);
            println!("{}condition:", ind);
            print_expr(cx, cond, indent + 1);
            if let Some(els) = *else_opt {
                println!("{}else:", ind);
                print_expr(cx, els, indent + 1);
            }
        },
        hir::ExprKind::Match(cond, _, ref source) => {
            println!("{}Match", ind);
            println!("{}condition:", ind);
            print_expr(cx, cond, indent + 1);
            println!("{}source: {:?}", ind, source);
        },
        hir::ExprKind::Closure(ref clause, _, _, _, _) => {
            println!("{}Closure", ind);
            println!("{}clause: {:?}", ind, clause);
        },
        hir::ExprKind::Yield(sub, _) => {
            println!("{}Yield", ind);
            print_expr(cx, sub, indent + 1);
        },
        hir::ExprKind::Block(_, _) => {
            println!("{}Block", ind);
        },
        hir::ExprKind::Assign(lhs, rhs, _) => {
            println!("{}Assign", ind);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprKind::AssignOp(ref binop, lhs, rhs) => {
            println!("{}AssignOp", ind);
            println!("{}op: {:?}", ind, binop.node);
            println!("{}lhs:", ind);
            print_expr(cx, lhs, indent + 1);
            println!("{}rhs:", ind);
            print_expr(cx, rhs, indent + 1);
        },
        hir::ExprKind::Field(e, ident) => {
            println!("{}Field", ind);
            println!("{}field name: {}", ind, ident.name);
            println!("{}struct expr:", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprKind::Index(arr, idx) => {
            println!("{}Index", ind);
            println!("{}array expr:", ind);
            print_expr(cx, arr, indent + 1);
            println!("{}index expr:", ind);
            print_expr(cx, idx, indent + 1);
        },
        hir::ExprKind::Path(hir::QPath::Resolved(ref ty, path)) => {
            println!("{}Resolved Path, {:?}", ind, ty);
            println!("{}path: {:?}", ind, path);
        },
        hir::ExprKind::Path(hir::QPath::TypeRelative(ty, seg)) => {
            println!("{}Relative Path, {:?}", ind, ty);
            println!("{}seg: {:?}", ind, seg);
        },
        hir::ExprKind::Path(hir::QPath::LangItem(lang_item, ..)) => {
            println!("{}Lang Item Path, {:?}", ind, lang_item.name());
        },
        hir::ExprKind::AddrOf(kind, ref muta, e) => {
            println!("{}AddrOf", ind);
            println!("kind: {:?}", kind);
            println!("mutability: {:?}", muta);
            print_expr(cx, e, indent + 1);
        },
        hir::ExprKind::Break(_, ref e) => {
            println!("{}Break", ind);
            if let Some(e) = *e {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprKind::Continue(_) => println!("{}Again", ind),
        hir::ExprKind::Ret(ref e) => {
            println!("{}Ret", ind);
            if let Some(e) = *e {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprKind::InlineAsm(asm) => {
            println!("{}InlineAsm", ind);
            println!("{}template: {}", ind, InlineAsmTemplatePiece::to_string(asm.template));
            println!("{}options: {:?}", ind, asm.options);
            println!("{}operands:", ind);
            for (op, _op_sp) in asm.operands {
                match op {
                    hir::InlineAsmOperand::In { expr, .. }
                    | hir::InlineAsmOperand::InOut { expr, .. }
                    | hir::InlineAsmOperand::Sym { expr } => print_expr(cx, expr, indent + 1),
                    hir::InlineAsmOperand::Out { expr, .. } => {
                        if let Some(expr) = expr {
                            print_expr(cx, expr, indent + 1);
                        }
                    },
                    hir::InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                        print_expr(cx, in_expr, indent + 1);
                        if let Some(out_expr) = out_expr {
                            print_expr(cx, out_expr, indent + 1);
                        }
                    },
                    hir::InlineAsmOperand::Const { anon_const } => {
                        println!("{}anon_const:", ind);
                        print_expr(cx, &cx.tcx.hir().body(anon_const.body).value, indent + 1);
                    },
                }
            }
        },
        hir::ExprKind::LlvmInlineAsm(asm) => {
            let inputs = &asm.inputs_exprs;
            let outputs = &asm.outputs_exprs;
            println!("{}LlvmInlineAsm", ind);
            println!("{}inputs:", ind);
            for e in inputs.iter() {
                print_expr(cx, e, indent + 1);
            }
            println!("{}outputs:", ind);
            for e in outputs.iter() {
                print_expr(cx, e, indent + 1);
            }
        },
        hir::ExprKind::Struct(path, fields, ref base) => {
            println!("{}Struct", ind);
            println!("{}path: {:?}", ind, path);
            for field in fields {
                println!("{}field \"{}\":", ind, field.ident.name);
                print_expr(cx, field.expr, indent + 1);
            }
            if let Some(base) = *base {
                println!("{}base:", ind);
                print_expr(cx, base, indent + 1);
            }
        },
        hir::ExprKind::ConstBlock(ref anon_const) => {
            println!("{}ConstBlock", ind);
            println!("{}anon_const:", ind);
            print_expr(cx, &cx.tcx.hir().body(anon_const.body).value, indent + 1);
        },
        hir::ExprKind::Repeat(val, length) => {
            println!("{}Repeat", ind);
            println!("{}value:", ind);
            print_expr(cx, val, indent + 1);
            println!("{}repeat count:", ind);
            match length {
                hir::ArrayLen::Infer(_, _) => println!("{}repeat count: _", ind),
                hir::ArrayLen::Body(anon_const) => {
                    print_expr(cx, &cx.tcx.hir().body(anon_const.body).value, indent + 1)
                }
            }
        },
        hir::ExprKind::Err => {
            println!("{}Err", ind);
        },
        hir::ExprKind::DropTemps(e) => {
            println!("{}DropTemps", ind);
            print_expr(cx, e, indent + 1);
        },
    }
}

fn print_item(cx: &LateContext<'_>, item: &hir::Item<'_>) {
    let did = item.def_id;
    println!("item `{}`", item.ident.name);
    match item.vis.node {
        hir::VisibilityKind::Public => println!("public"),
        hir::VisibilityKind::Crate(_) => println!("visible crate wide"),
        hir::VisibilityKind::Restricted { path, .. } => println!(
            "visible in module `{}`",
            rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_path(path, false))
        ),
        hir::VisibilityKind::Inherited => println!("visibility inherited from outer item"),
    }
    match item.kind {
        hir::ItemKind::ExternCrate(ref _renamed_from) => {
            if let Some(crate_id) = cx.tcx.extern_mod_stmt_cnum(did) {
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
        hir::ItemKind::Use(path, ref kind) => println!("{:?}, {:?}", path, kind),
        hir::ItemKind::Static(..) => println!("static item of type {:#?}", cx.tcx.type_of(did)),
        hir::ItemKind::Const(..) => println!("const item of type {:#?}", cx.tcx.type_of(did)),
        hir::ItemKind::Fn(..) => {
            let item_ty = cx.tcx.type_of(did);
            println!("function of type {:#?}", item_ty);
        },
        hir::ItemKind::Macro(ref macro_def) => {
            if macro_def.macro_rules {
                println!("macro introduced by `macro_rules!`");
            } else {
                println!("macro introduced by `macro`");
            }
        },
        hir::ItemKind::Mod(..) => println!("module"),
        hir::ItemKind::ForeignMod { abi, .. } => println!("foreign module with abi: {}", abi),
        hir::ItemKind::GlobalAsm(asm) => println!("global asm: {:?}", asm),
        hir::ItemKind::TyAlias(..) => {
            println!("type alias for {:?}", cx.tcx.type_of(did));
        },
        hir::ItemKind::OpaqueTy(..) => {
            println!("existential type with real type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemKind::Enum(..) => {
            println!("enum definition of type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemKind::Struct(..) => {
            println!("struct definition of type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemKind::Union(..) => {
            println!("union definition of type {:?}", cx.tcx.type_of(did));
        },
        hir::ItemKind::Trait(..) => {
            println!("trait decl");
            if cx.tcx.trait_is_auto(did.to_def_id()) {
                println!("trait is auto");
            } else {
                println!("trait is not auto");
            }
        },
        hir::ItemKind::TraitAlias(..) => {
            println!("trait alias");
        },
        hir::ItemKind::Impl(hir::Impl {
            of_trait: Some(ref _trait_ref),
            ..
        }) => {
            println!("trait impl");
        },
        hir::ItemKind::Impl(hir::Impl { of_trait: None, .. }) => {
            println!("impl");
        },
    }
}

#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
fn print_pat(cx: &LateContext<'_>, pat: &hir::Pat<'_>, indent: usize) {
    let ind = "  ".repeat(indent);
    println!("{}+", ind);
    match pat.kind {
        hir::PatKind::Wild => println!("{}Wild", ind),
        hir::PatKind::Binding(ref mode, .., ident, ref inner) => {
            println!("{}Binding", ind);
            println!("{}mode: {:?}", ind, mode);
            println!("{}name: {}", ind, ident.name);
            if let Some(inner) = *inner {
                println!("{}inner:", ind);
                print_pat(cx, inner, indent + 1);
            }
        },
        hir::PatKind::Or(fields) => {
            println!("{}Or", ind);
            for field in fields {
                print_pat(cx, field, indent + 1);
            }
        },
        hir::PatKind::Struct(ref path, fields, ignore) => {
            println!("{}Struct", ind);
            println!(
                "{}name: {}",
                ind,
                rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false))
            );
            println!("{}ignore leftover fields: {}", ind, ignore);
            println!("{}fields:", ind);
            for field in fields {
                println!("{}  field name: {}", ind, field.ident.name);
                if field.is_shorthand {
                    println!("{}  in shorthand notation", ind);
                }
                print_pat(cx, field.pat, indent + 1);
            }
        },
        hir::PatKind::TupleStruct(ref path, fields, opt_dots_position) => {
            println!("{}TupleStruct", ind);
            println!(
                "{}path: {}",
                ind,
                rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false))
            );
            if let Some(dot_position) = opt_dots_position {
                println!("{}dot position: {}", ind, dot_position);
            }
            for field in fields {
                print_pat(cx, field, indent + 1);
            }
        },
        hir::PatKind::Path(hir::QPath::Resolved(ref ty, path)) => {
            println!("{}Resolved Path, {:?}", ind, ty);
            println!("{}path: {:?}", ind, path);
        },
        hir::PatKind::Path(hir::QPath::TypeRelative(ty, seg)) => {
            println!("{}Relative Path, {:?}", ind, ty);
            println!("{}seg: {:?}", ind, seg);
        },
        hir::PatKind::Path(hir::QPath::LangItem(lang_item, ..)) => {
            println!("{}Lang Item Path, {:?}", ind, lang_item.name());
        },
        hir::PatKind::Tuple(pats, opt_dots_position) => {
            println!("{}Tuple", ind);
            if let Some(dot_position) = opt_dots_position {
                println!("{}dot position: {}", ind, dot_position);
            }
            for field in pats {
                print_pat(cx, field, indent + 1);
            }
        },
        hir::PatKind::Box(inner) => {
            println!("{}Box", ind);
            print_pat(cx, inner, indent + 1);
        },
        hir::PatKind::Ref(inner, ref muta) => {
            println!("{}Ref", ind);
            println!("{}mutability: {:?}", ind, muta);
            print_pat(cx, inner, indent + 1);
        },
        hir::PatKind::Lit(e) => {
            println!("{}Lit", ind);
            print_expr(cx, e, indent + 1);
        },
        hir::PatKind::Range(ref l, ref r, ref range_end) => {
            println!("{}Range", ind);
            if let Some(expr) = l {
                print_expr(cx, expr, indent + 1);
            }
            if let Some(expr) = r {
                print_expr(cx, expr, indent + 1);
            }
            match *range_end {
                hir::RangeEnd::Included => println!("{} end included", ind),
                hir::RangeEnd::Excluded => println!("{} end excluded", ind),
            }
        },
        hir::PatKind::Slice(first_pats, ref range, last_pats) => {
            println!("{}Slice [a, b, ..i, y, z]", ind);
            println!("[a, b]:");
            for pat in first_pats {
                print_pat(cx, pat, indent + 1);
            }
            println!("i:");
            if let Some(pat) = *range {
                print_pat(cx, pat, indent + 1);
            }
            println!("[y, z]:");
            for pat in last_pats {
                print_pat(cx, pat, indent + 1);
            }
        },
    }
}

fn print_guard(cx: &LateContext<'_>, guard: &hir::Guard<'_>, indent: usize) {
    let ind = "  ".repeat(indent);
    println!("{}+", ind);
    match guard {
        hir::Guard::If(expr) => {
            println!("{}If", ind);
            print_expr(cx, expr, indent + 1);
        },
        hir::Guard::IfLet(pat, expr) => {
            println!("{}IfLet", ind);
            print_pat(cx, pat, indent + 1);
            print_expr(cx, expr, indent + 1);
        },
    }
}
