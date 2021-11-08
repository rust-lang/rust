//! A group of attributes that can be attached to Rust code in order
//! to generate a clippy lint detecting said code automatically.

use clippy_utils::{get_attr, higher};
use rustc_ast::ast::{LitFloatType, LitKind};
use rustc_ast::{walk_list, Label, LitIntType};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::intravisit::{NestedVisitorMap, Visitor};
use rustc_hir::{Arm, Block, Expr, ExprKind, FnRetTy, Lit, MatchSource, Pat, PatKind, QPath, Stmt, StmtKind, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Generates clippy code that detects the offending pattern
    ///
    /// ### Example
    /// ```rust,ignore
    /// // ./tests/ui/my_lint.rs
    /// fn foo() {
    ///     // detect the following pattern
    ///     #[clippy::author]
    ///     if x == 42 {
    ///         // but ignore everything from here on
    ///         #![clippy::author = "ignore"]
    ///     }
    ///     ()
    /// }
    /// ```
    ///
    /// Running `TESTNAME=ui/my_lint cargo uitest` will produce
    /// a `./tests/ui/new_lint.stdout` file with the generated code:
    ///
    /// ```rust,ignore
    /// // ./tests/ui/new_lint.stdout
    /// if_chain! {
    ///     if let ExprKind::If(ref cond, ref then, None) = item.kind,
    ///     if let ExprKind::Binary(BinOp::Eq, ref left, ref right) = cond.kind,
    ///     if let ExprKind::Path(ref path) = left.kind,
    ///     if let ExprKind::Lit(ref lit) = right.kind,
    ///     if let LitKind::Int(42, _) = lit.node,
    ///     then {
    ///         // report your lint here
    ///     }
    /// }
    /// ```
    pub LINT_AUTHOR,
    internal_warn,
    "helper for writing lints"
}

declare_lint_pass!(Author => [LINT_AUTHOR]);

fn prelude() {
    println!("if_chain! {{");
}

fn done() {
    println!("    then {{");
    println!("        // report your lint here");
    println!("    }}");
    println!("}}");
}

impl<'tcx> LateLintPass<'tcx> for Author {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        if !has_attr(cx, item.hir_id()) {
            return;
        }
        prelude();
        PrintVisitor::new("item", cx).visit_item(item);
        done();
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        if !has_attr(cx, item.hir_id()) {
            return;
        }
        prelude();
        PrintVisitor::new("item", cx).visit_impl_item(item);
        done();
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        if !has_attr(cx, item.hir_id()) {
            return;
        }
        prelude();
        PrintVisitor::new("item", cx).visit_trait_item(item);
        done();
    }

    fn check_variant(&mut self, cx: &LateContext<'tcx>, var: &'tcx hir::Variant<'_>) {
        if !has_attr(cx, var.id) {
            return;
        }
        prelude();
        let parent_hir_id = cx.tcx.hir().get_parent_node(var.id);
        PrintVisitor::new("var", cx).visit_variant(var, &hir::Generics::empty(), parent_hir_id);
        done();
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &'tcx hir::FieldDef<'_>) {
        if !has_attr(cx, field.hir_id) {
            return;
        }
        prelude();
        PrintVisitor::new("field", cx).visit_field_def(field);
        done();
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if !has_attr(cx, expr.hir_id) {
            return;
        }
        prelude();
        PrintVisitor::new("expr", cx).visit_expr(expr);
        done();
    }

    fn check_arm(&mut self, cx: &LateContext<'tcx>, arm: &'tcx hir::Arm<'_>) {
        if !has_attr(cx, arm.hir_id) {
            return;
        }
        prelude();
        PrintVisitor::new("arm", cx).visit_arm(arm);
        done();
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx hir::Stmt<'_>) {
        if !has_attr(cx, stmt.hir_id) {
            return;
        }
        match stmt.kind {
            StmtKind::Expr(e) | StmtKind::Semi(e) if has_attr(cx, e.hir_id) => return,
            _ => {},
        }
        prelude();
        PrintVisitor::new("stmt", cx).visit_stmt(stmt);
        done();
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ForeignItem<'_>) {
        if !has_attr(cx, item.hir_id()) {
            return;
        }
        prelude();
        PrintVisitor::new("item", cx).visit_foreign_item(item);
        done();
    }
}

impl<'a, 'tcx> PrintVisitor<'a, 'tcx> {
    #[must_use]
    fn new(s: &'static str, cx: &'a LateContext<'tcx>) -> Self {
        Self {
            ids: FxHashMap::default(),
            current: s.to_owned(),
            cx,
        }
    }

    fn next(&mut self, s: &'static str) -> String {
        use std::collections::hash_map::Entry::{Occupied, Vacant};
        match self.ids.entry(s) {
            // already there: start numbering from `1`
            Occupied(mut occ) => {
                let val = occ.get_mut();
                *val += 1;
                format!("{}{}", s, *val)
            },
            // not there: insert and return name as given
            Vacant(vac) => {
                vac.insert(0);
                s.to_owned()
            },
        }
    }

    fn print_qpath(&mut self, path: &QPath<'_>) {
        if let QPath::LangItem(lang_item, _) = *path {
            println!(
                "    if matches!({}, QPath::LangItem(LangItem::{:?}, _));",
                self.current, lang_item,
            );
        } else {
            print!("    if match_qpath({}, &[", self.current);
            print_path(path, &mut true);
            println!("]);");
        }
    }

    fn print_label(&mut self, label: Option<Label>) {
        if let Some(label) = label {
            let label_bind = self.next("label");

            println!("    if let Some(ref {}) = {}", label_bind, self.current);

            let label_name_bind = self.next("label_name");
            let label_name = label.ident.name;

            println!(
                "    if {}.ident.name.as_str() == {:?};",
                label_name_bind,
                label_name.as_str()
            );
        }
    }

    fn print_lit_expr(&mut self, lit: &Lit, current: &str) {
        let lit_pat = self.next("lit");

        println!("Lit(ref {}) = {};", lit_pat, current);

        match lit.node {
            LitKind::Bool(val) => println!("    if let LitKind::Bool({:?}) = {}.node;", val, lit_pat),
            LitKind::Char(c) => println!("    if let LitKind::Char({:?}) = {}.node;", c, lit_pat),
            LitKind::Err(val) => println!("    if let LitKind::Err({}) = {}.node;", val, lit_pat),
            LitKind::Byte(b) => println!("    if let LitKind::Byte({}) = {}.node;", b, lit_pat),
            LitKind::Int(i, suffix) => {
                let int_ty = match suffix {
                    LitIntType::Signed(int_ty) => format!("LitIntType::Signed(IntTy::{:?})", int_ty),
                    LitIntType::Unsigned(uint_ty) => format!("LitIntType::Unsigned(UintTy::{:?})", uint_ty),
                    LitIntType::Unsuffixed => String::from("LitIntType::Unsuffixed"),
                };

                println!("    if let LitKind::Int({}, {}) = {}.node;", i, int_ty, lit_pat);
            },
            LitKind::Float(_, suffix) => {
                let float_ty = match suffix {
                    LitFloatType::Suffixed(suffix_ty) => format!("LitFloatType::Suffixed(FloatTy::{:?})", suffix_ty),
                    LitFloatType::Unsuffixed => String::from("LitFloatType::Unsuffixed"),
                };

                println!("    if let LitKind::Float(_, {}) = {}.node;", float_ty, lit_pat);
            },
            LitKind::ByteStr(ref vec) => {
                let vec_pat = self.next("vec");

                println!("    if let LitKind::ByteStr(ref {}) = {}.node;", vec_pat, lit_pat);
                println!("    if let [{:?}] = **{};", vec, vec_pat);
            },
            LitKind::Str(ref text, _) => {
                let str_pat = self.next("s");

                println!("    if let LitKind::Str(ref {}, _) = {}.node;", str_pat, lit_pat);
                println!("    if {}.as_str() == {:?}", str_pat, &*text.as_str());
            },
        }
    }

    fn print_match_expr(&mut self, expr: &Expr<'_>, arms: &[Arm<'_>], des: MatchSource, current: &str) {
        let expr_pat = self.next("expr");
        let arms_pat = self.next("arms");

        println!(
            "Match(ref {}, ref {}, MatchSource::{:?}) = {};",
            expr_pat, arms_pat, des, current
        );

        self.current = expr_pat;
        self.visit_expr(expr);

        println!("    if {}.len() == {};", arms_pat, arms.len());

        for (i, arm) in arms.iter().enumerate() {
            self.current = format!("{}[{}].body", arms_pat, i);
            self.visit_expr(arm.body);

            if let Some(ref guard) = arm.guard {
                let guard_pat = self.next("guard");

                println!("    if let Some(ref {}) = {}[{}].guard;", guard_pat, arms_pat, i);

                match guard {
                    hir::Guard::If(if_expr) => {
                        let if_expr_pat = self.next("expr");

                        println!("    if let Guard::If(ref {}) = {};", if_expr_pat, guard_pat);

                        self.current = if_expr_pat;
                        self.visit_expr(if_expr);
                    },
                    hir::Guard::IfLet(if_let_pat, if_let_expr) => {
                        let if_let_pat_pat = self.next("pat");
                        let if_let_expr_pat = self.next("expr");

                        println!(
                            "    if let Guard::IfLet(ref {}, ref {}) = {};",
                            if_let_pat_pat, if_let_expr_pat, guard_pat
                        );

                        self.current = if_let_expr_pat;
                        self.visit_expr(if_let_expr);

                        self.current = if_let_pat_pat;
                        self.visit_pat(if_let_pat);
                    },
                }
            }
            self.current = format!("{}[{}].pat", arms_pat, i);
            self.visit_pat(arm.pat);
        }
    }

    fn check_higher(&mut self, expr: &Expr<'_>) -> bool {
        if let Some(higher::While { condition, body }) = higher::While::hir(expr) {
            let condition_pat = self.next("condition");
            let body_pat = self.next("body");

            println!(
                "    if let Some(higher::While {{ condition: {}, body: {} }}) = higher::While::hir({})",
                condition_pat, body_pat, self.current
            );

            self.current = condition_pat;
            self.visit_expr(condition);

            self.current = body_pat;
            self.visit_expr(body);

            return true;
        }

        if let Some(higher::WhileLet {
            let_pat,
            let_expr,
            if_then,
        }) = higher::WhileLet::hir(expr)
        {
            let let_pat_ = self.next("let_pat");
            let let_expr_pat = self.next("let_expr");
            let if_then_pat = self.next("if_then");

            println!(
                "    if let Some(higher::WhileLet {{ let_pat: {}, let_expr: {}, if_then: {} }}) = higher::WhileLet::hir({})",
                let_pat_, let_expr_pat, if_then_pat, self.current
            );

            self.current = let_pat_;
            self.visit_pat(let_pat);

            self.current = let_expr_pat;
            self.visit_expr(let_expr);

            self.current = if_then_pat;
            self.visit_expr(if_then);

            return true;
        }

        if let Some(higher::IfLet {
            let_pat,
            let_expr,
            if_then,
            if_else,
        }) = higher::IfLet::hir(self.cx, expr)
        {
            let let_pat_ = self.next("let_pat");
            let let_expr_pat = self.next("let_expr");
            let if_then_pat = self.next("if_then");
            let else_pat = self.next("else_expr");

            println!(
                "    if let Some(higher::IfLet {{ let_pat: {}, let_expr: {}, if_then: {}, if_else: {}}}) = higher::IfLet::hir({})",
                let_pat_, let_expr_pat, if_then_pat, else_pat, self.current
            );

            self.current = let_pat_;
            self.visit_pat(let_pat);

            self.current = let_expr_pat;
            self.visit_expr(let_expr);

            self.current = if_then_pat;
            self.visit_expr(if_then);

            if let Some(else_expr) = if_else {
                self.current = else_pat;
                self.visit_expr(else_expr);
            }

            return true;
        }

        if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr) {
            let cond_pat = self.next("cond");
            let then_pat = self.next("then");
            let else_pat = self.next("else_expr");

            println!(
                "    if let Some(higher::If {{ cond: {}, then: {}, r#else: {}}}) = higher::If::hir({})",
                cond_pat, then_pat, else_pat, self.current
            );

            self.current = cond_pat;
            self.visit_expr(cond);

            self.current = then_pat;
            self.visit_expr(then);

            if let Some(else_expr) = r#else {
                self.current = else_pat;
                self.visit_expr(else_expr);
            }

            return true;
        }

        if let Some(higher::ForLoop { pat, arg, body, .. }) = higher::ForLoop::hir(expr) {
            let pat_ = self.next("pat");
            let arg_pat = self.next("arg");
            let body_pat = self.next("body");

            println!(
                "    if let Some(higher::ForLoop {{ pat: {}, arg: {}, body: {}, ..}}) = higher::ForLoop::hir({})",
                pat_, arg_pat, body_pat, self.current
            );

            self.current = pat_;
            self.visit_pat(pat);

            self.current = arg_pat;
            self.visit_expr(arg);

            self.current = body_pat;
            self.visit_expr(body);

            return true;
        }

        false
    }
}

struct PrintVisitor<'a, 'tcx> {
    /// Fields are the current index that needs to be appended to pattern
    /// binding names
    ids: FxHashMap<&'static str, usize>,
    /// the name that needs to be destructured
    current: String,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for PrintVisitor<'a, '_> {
    type Map = Map<'tcx>;

    #[allow(clippy::too_many_lines)]
    fn visit_expr(&mut self, expr: &Expr<'_>) {
        if self.check_higher(expr) {
            return;
        }

        print!("    if let ExprKind::");
        let current = format!("{}.kind", self.current);

        match expr.kind {
            ExprKind::Let(pat, expr, _) => {
                let let_pat = self.next("pat");
                let let_expr = self.next("expr");

                println!("Let(ref {}, ref {}, _) = {};", let_pat, let_expr, current);

                self.current = let_expr;
                self.visit_expr(expr);

                self.current = let_pat;
                self.visit_pat(pat);
            },
            ExprKind::Box(inner) => {
                let inner_pat = self.next("inner");

                println!("Box(ref {}) = {};", inner_pat, current);

                self.current = inner_pat;
                self.visit_expr(inner);
            },
            ExprKind::Array(elements) => {
                let elements_pat = self.next("elements");

                println!("Array(ref {}) = {};", elements_pat, current);

                println!("    if {}.len() == {};", elements_pat, elements.len());

                for (i, element) in elements.iter().enumerate() {
                    self.current = format!("{}[{}]", elements_pat, i);
                    self.visit_expr(element);
                }
            },
            ExprKind::Call(func, args) => {
                let func_pat = self.next("func");
                let args_pat = self.next("args");

                println!("Call(ref {}, ref {}) = {};", func_pat, args_pat, current);

                self.current = func_pat;
                self.visit_expr(func);

                println!("    if {}.len() == {};", args_pat, args.len());

                for (i, arg) in args.iter().enumerate() {
                    self.current = format!("{}[{}]", args_pat, i);
                    self.visit_expr(arg);
                }
            },
            ExprKind::MethodCall(method_name, _, args, _) => {
                let method_name_pat = self.next("method_name");
                let args_pat = self.next("args");

                println!(
                    "MethodCall(ref {}, ref {}, _) = {};",
                    method_name_pat, args_pat, current
                );

                println!(
                    "    if {}.ident.name.as_str() == {};",
                    method_name_pat,
                    method_name.ident.name.as_str()
                );

                println!("    if {}.len() == {};", args_pat, args.len());

                for (i, arg) in args.iter().enumerate() {
                    self.current = format!("{}[{}]", args_pat, i);
                    self.visit_expr(arg);
                }
            },
            ExprKind::Tup(elements) => {
                let elements_pat = self.next("elements");

                println!("Tup(ref {}) = {};", elements_pat, current);

                println!("    if {}.len() == {};", elements_pat, elements.len());

                for (i, element) in elements.iter().enumerate() {
                    self.current = format!("{}[{}]", elements_pat, i);
                    self.visit_expr(element);
                }
            },
            ExprKind::Binary(ref op, left, right) => {
                let op_pat = self.next("op");
                let left_pat = self.next("left");
                let right_pat = self.next("right");

                println!(
                    "Binary(ref {}, ref {}, ref {}) = {};",
                    op_pat, left_pat, right_pat, current
                );

                println!("    if BinOpKind::{:?} == {}.node;", op.node, op_pat);

                self.current = left_pat;
                self.visit_expr(left);

                self.current = right_pat;
                self.visit_expr(right);
            },
            ExprKind::Unary(ref op, inner) => {
                let inner_pat = self.next("inner");

                println!("Unary(UnOp::{:?}, ref {}) = {};", op, inner_pat, current);

                self.current = inner_pat;
                self.visit_expr(inner);
            },
            ExprKind::Lit(ref lit) => self.print_lit_expr(lit, &current),
            ExprKind::Cast(expr, ty) => {
                let cast_pat = self.next("expr");
                let cast_ty = self.next("cast_ty");
                let qp_label = self.next("qpath");

                println!("Cast(ref {}, ref {}) = {};", cast_pat, cast_ty, current);

                if let TyKind::Path(ref qp) = ty.kind {
                    println!("    if let TyKind::Path(ref {}) = {}.kind;", qp_label, cast_ty);

                    self.current = qp_label;
                    self.print_qpath(qp);
                }

                self.current = cast_pat;
                self.visit_expr(expr);
            },
            ExprKind::Type(expr, _ty) => {
                let cast_pat = self.next("expr");

                println!("Type(ref {}, _) = {};", cast_pat, current);

                self.current = cast_pat;
                self.visit_expr(expr);
            },
            ExprKind::Loop(body, label, des, _) => {
                let body_pat = self.next("body");
                let label_pat = self.next("label");

                println!(
                    "Loop(ref {}, ref {}, LoopSource::{:?}) = {};",
                    body_pat, label_pat, des, current
                );

                self.current = body_pat;
                self.visit_block(body);

                self.current = label_pat;
                self.print_label(label);
            },
            ExprKind::If(_, _, _) => {}, // Covered by check_higher
            ExprKind::Match(match_expr, arms, des) => self.print_match_expr(match_expr, arms, des, &current),
            ExprKind::Closure(capture_clause, fn_decl, body_id, _, movability) => {
                let capture_by = format!("CaptureBy::{:?}", capture_clause);

                let movability = if let Some(movability) = movability {
                    format!("Some(Movability::{:?})", movability)
                } else {
                    String::from("None")
                };

                let ret_ty = match fn_decl.output {
                    FnRetTy::DefaultReturn(_) => "FnRetTy::DefaultReturn(_)",
                    FnRetTy::Return(_) => "FnRetTy::Return(_ty)",
                };

                let fn_decl_pat = self.next("fn_decl");
                let body_id_pat = self.next("body_id");

                println!(
                    "Closure({}, ref {}, ref {}, _, {}) = {}",
                    capture_by, fn_decl_pat, body_id_pat, movability, current
                );
                println!("    if let {} = {}.output", ret_ty, fn_decl_pat);

                let hir = self.cx.tcx.hir();
                let body = hir.body(body_id);

                let body_pat = self.next("body");

                println!("    let {} = cx.tcx.hir().body({});", body_pat, body_id_pat);

                self.current = format!("{}.value", body_pat);
                self.visit_expr(&body.value);
            },
            ExprKind::Yield(sub, source) => {
                let sub_pat = self.next("sub");

                println!("Yield(ref sub, YieldSource::{:?}) = {};", source, current);

                self.current = sub_pat;
                self.visit_expr(sub);
            },
            ExprKind::Block(block, label) => {
                let block_pat = self.next("block");
                let label_pat = self.next("label");

                println!("Block(ref {}, ref {}) = {};", block_pat, label_pat, current);

                self.current = block_pat;
                self.visit_block(block);

                self.current = label_pat;
                self.print_label(label);
            },
            ExprKind::Assign(target, value, _) => {
                let target_pat = self.next("target");
                let value_pat = self.next("value");

                println!(
                    "Assign(ref {}, ref {}, ref _span) = {};",
                    target_pat, value_pat, current
                );

                self.current = target_pat;
                self.visit_expr(target);

                self.current = value_pat;
                self.visit_expr(value);
            },
            ExprKind::AssignOp(ref op, target, value) => {
                let op_pat = self.next("op");
                let target_pat = self.next("target");
                let value_pat = self.next("value");

                println!(
                    "AssignOp(ref {}, ref {}, ref {}) = {};",
                    op_pat, target_pat, value_pat, current
                );

                println!("    if BinOpKind::{:?} == {}.node;", op.node, op_pat);

                self.current = target_pat;
                self.visit_expr(target);

                self.current = value_pat;
                self.visit_expr(value);
            },
            ExprKind::Field(object, ref field_ident) => {
                let obj_pat = self.next("object");
                let field_name_pat = self.next("field_name");

                println!("Field(ref {}, ref {}) = {};", obj_pat, field_name_pat, current);
                println!("    if {}.as_str() == {:?}", field_name_pat, field_ident.as_str());

                self.current = obj_pat;
                self.visit_expr(object);
            },
            ExprKind::Index(object, index) => {
                let object_pat = self.next("object");
                let index_pat = self.next("index");

                println!("Index(ref {}, ref {}) = {};", object_pat, index_pat, current);

                self.current = object_pat;
                self.visit_expr(object);

                self.current = index_pat;
                self.visit_expr(index);
            },
            ExprKind::Path(ref path) => {
                let path_pat = self.next("qpath");

                println!("Path(ref {}) = {};", path_pat, current);

                self.current = path_pat;
                self.print_qpath(path);
            },
            ExprKind::AddrOf(kind, mutability, inner) => {
                let inner_pat = self.next("inner");

                println!(
                    "AddrOf(BorrowKind::{:?}, Mutability::{:?}, ref {}) = {};",
                    kind, mutability, inner_pat, current
                );

                self.current = inner_pat;
                self.visit_expr(inner);
            },
            ExprKind::Break(ref destination, ref opt_value) => {
                let destination_pat = self.next("destination");

                if let Some(value) = *opt_value {
                    let value_pat = self.next("value");

                    println!("Break(ref {}, Some(ref {})) = {};", destination_pat, value_pat, current);

                    self.current = value_pat;
                    self.visit_expr(value);
                } else {
                    println!("Break(ref {}, None) = {};", destination_pat, current);
                }

                self.current = format!("{}.label", destination_pat);
                self.print_label(destination.label);
            },
            ExprKind::Continue(ref destination) => {
                let destination_pat = self.next("destination");
                println!("Continue(ref {}) = {};", destination_pat, current);

                self.current = format!("{}.label", destination_pat);
                self.print_label(destination.label);
            },
            ExprKind::Ret(ref opt_value) => {
                if let Some(value) = *opt_value {
                    let value_pat = self.next("value");

                    println!("Ret(Some(ref {})) = {};", value_pat, current);

                    self.current = value_pat;
                    self.visit_expr(value);
                } else {
                    println!("Ret(None) = {};", current);
                }
            },
            ExprKind::InlineAsm(_) => {
                println!("InlineAsm(_) = {};", current);
                println!("    // unimplemented: `ExprKind::InlineAsm` is not further destructured at the moment");
            },
            ExprKind::LlvmInlineAsm(_) => {
                println!("LlvmInlineAsm(_) = {};", current);
                println!("    // unimplemented: `ExprKind::LlvmInlineAsm` is not further destructured at the moment");
            },
            ExprKind::Struct(path, fields, ref opt_base) => {
                let path_pat = self.next("qpath");
                let fields_pat = self.next("fields");

                if let Some(base) = *opt_base {
                    let base_pat = self.next("base");

                    println!(
                        "Struct(ref {}, ref {}, Some(ref {})) = {};",
                        path_pat, fields_pat, base_pat, current
                    );

                    self.current = base_pat;
                    self.visit_expr(base);
                } else {
                    println!("Struct(ref {}, ref {}, None) = {};", path_pat, fields_pat, current);
                }

                self.current = path_pat;
                self.print_qpath(path);

                println!("    if {}.len() == {};", fields_pat, fields.len());

                for (i, field) in fields.iter().enumerate() {
                    println!(
                        "    if {}[{}].ident.name.as_str() == {:?}",
                        fields_pat,
                        i,
                        &*field.ident.name.as_str()
                    );

                    self.current = format!("{}[{}]", fields_pat, i);
                    self.visit_expr(field.expr);
                }
            },
            ExprKind::ConstBlock(_) => {
                let value_pat = self.next("value");
                println!("Const({}) = {}", value_pat, current);
                self.current = value_pat;
            },
            ExprKind::Repeat(value, length) => {
                let value_pat = self.next("value");
                let length_pat = self.next("length");

                println!("Repeat(ref {}, ref {}) = {};", value_pat, length_pat, current);

                self.current = value_pat;
                self.visit_expr(value);

                let hir = self.cx.tcx.hir();
                let body = hir.body(length.body);

                self.current = format!("{}.value", length_pat);
                self.visit_expr(&body.value);
            },
            ExprKind::Err => {
                println!("Err = {}", current);
            },
            ExprKind::DropTemps(expr) => {
                let expr_pat = self.next("expr");

                println!("DropTemps(ref {}) = {};", expr_pat, current);

                self.current = expr_pat;
                self.visit_expr(expr);
            },
        }
    }

    fn visit_block(&mut self, block: &Block<'_>) {
        println!("    if {}.stmts.len() == {};", self.current, block.stmts.len());

        let block_name = self.current.clone();

        for (i, stmt) in block.stmts.iter().enumerate() {
            self.current = format!("{}.stmts[{}]", block_name, i);
            self.visit_stmt(stmt);
        }

        if let Some(expr) = block.expr {
            self.current = self.next("trailing_expr");
            println!("    if let Some({}) = &{}.expr;", self.current, block_name);
            self.visit_expr(expr);
        } else {
            println!("    if {}.expr.is_none();", block_name);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn visit_pat(&mut self, pat: &Pat<'_>) {
        print!("    if let PatKind::");
        let current = format!("{}.kind", self.current);

        match pat.kind {
            PatKind::Wild => println!("Wild = {};", current),
            PatKind::Binding(anno, .., ident, ref sub) => {
                let anno_pat = &format!("BindingAnnotation::{:?}", anno);
                let name_pat = self.next("name");

                if let Some(sub) = *sub {
                    let sub_pat = self.next("sub");

                    println!(
                        "Binding({}, _, {}, Some(ref {})) = {};",
                        anno_pat, name_pat, sub_pat, current
                    );

                    self.current = sub_pat;
                    self.visit_pat(sub);
                } else {
                    println!("Binding({}, _, {}, None) = {};", anno_pat, name_pat, current);
                }

                println!("    if {}.as_str() == \"{}\";", name_pat, ident.as_str());
            },
            PatKind::Struct(ref path, fields, ignore) => {
                let path_pat = self.next("qpath");
                let fields_pat = self.next("fields");
                println!(
                    "Struct(ref {}, ref {}, {}) = {};",
                    path_pat, fields_pat, ignore, current
                );

                self.current = path_pat;
                self.print_qpath(path);

                println!("    if {}.len() == {};", fields_pat, fields.len());

                for (i, field) in fields.iter().enumerate() {
                    println!(
                        "    if {}[{}].ident.name.as_str() == {:?}",
                        fields_pat,
                        i,
                        &*field.ident.name.as_str()
                    );

                    self.current = format!("{}[{}]", fields_pat, i);
                    self.visit_pat(field.pat);
                }
            },
            PatKind::Or(fields) => {
                let fields_pat = self.next("fields");
                println!("Or(ref {}) = {};", fields_pat, current);
                println!("    if {}.len() == {};", fields_pat, fields.len());

                for (i, field) in fields.iter().enumerate() {
                    self.current = format!("{}[{}]", fields_pat, i);
                    self.visit_pat(field);
                }
            },
            PatKind::TupleStruct(ref path, fields, skip_pos) => {
                let path_pat = self.next("qpath");
                let fields_pat = self.next("fields");

                println!(
                    "TupleStruct(ref {}, ref {}, {:?}) = {};",
                    path_pat, fields_pat, skip_pos, current
                );

                self.current = path_pat;
                self.print_qpath(path);

                println!("    if {}.len() == {};", fields_pat, fields.len());

                for (i, field) in fields.iter().enumerate() {
                    self.current = format!("{}[{}]", fields_pat, i);
                    self.visit_pat(field);
                }
            },
            PatKind::Path(ref path) => {
                let path_pat = self.next("qpath");
                println!("Path(ref {}) = {};", path_pat, current);

                self.current = path_pat;
                self.print_qpath(path);
            },
            PatKind::Tuple(fields, skip_pos) => {
                let fields_pat = self.next("fields");
                println!("Tuple(ref {}, {:?}) = {};", fields_pat, skip_pos, current);
                println!("    if {}.len() == {};", fields_pat, fields.len());

                for (i, field) in fields.iter().enumerate() {
                    self.current = format!("{}[{}]", fields_pat, i);
                    self.visit_pat(field);
                }
            },
            PatKind::Box(pat) => {
                let pat_pat = self.next("pat");
                println!("Box(ref {}) = {};", pat_pat, current);

                self.current = pat_pat;
                self.visit_pat(pat);
            },
            PatKind::Ref(pat, muta) => {
                let pat_pat = self.next("pat");
                println!("Ref(ref {}, Mutability::{:?}) = {};", pat_pat, muta, current);

                self.current = pat_pat;
                self.visit_pat(pat);
            },
            PatKind::Lit(lit_expr) => {
                let lit_expr_pat = self.next("lit_expr");
                println!("Lit(ref {}) = {}", lit_expr_pat, current);

                self.current = lit_expr_pat;
                self.visit_expr(lit_expr);
            },
            PatKind::Range(ref start, ref end, end_kind) => {
                let start_pat = self.next("start");
                let end_pat = self.next("end");

                println!(
                    "Range(ref {}, ref {}, RangeEnd::{:?}) = {};",
                    start_pat, end_pat, end_kind, current
                );

                self.current = start_pat;
                walk_list!(self, visit_expr, start);

                self.current = end_pat;
                walk_list!(self, visit_expr, end);
            },
            PatKind::Slice(start, ref middle, end) => {
                let start_pat = self.next("start");
                let end_pat = self.next("end");

                if let Some(middle) = middle {
                    let middle_pat = self.next("middle");
                    println!(
                        "Slice(ref {}, Some(ref {}), ref {}) = {};",
                        start_pat, middle_pat, end_pat, current
                    );
                    self.current = middle_pat;
                    self.visit_pat(middle);
                } else {
                    println!("Slice(ref {}, None, ref {}) = {};", start_pat, end_pat, current);
                }

                println!("    if {}.len() == {};", start_pat, start.len());

                for (i, pat) in start.iter().enumerate() {
                    self.current = format!("{}[{}]", start_pat, i);
                    self.visit_pat(pat);
                }

                println!("    if {}.len() == {};", end_pat, end.len());

                for (i, pat) in end.iter().enumerate() {
                    self.current = format!("{}[{}]", end_pat, i);
                    self.visit_pat(pat);
                }
            },
        }
    }

    fn visit_stmt(&mut self, s: &Stmt<'_>) {
        print!("    if let StmtKind::");
        let current = format!("{}.kind", self.current);

        match s.kind {
            // A local (let) binding:
            StmtKind::Local(local) => {
                let local_pat = self.next("local");
                println!("Local(ref {}) = {};", local_pat, current);

                if let Some(init) = local.init {
                    let init_pat = self.next("init");
                    println!("    if let Some(ref {}) = {}.init;", init_pat, local_pat);

                    self.current = init_pat;
                    self.visit_expr(init);
                }

                self.current = format!("{}.pat", local_pat);
                self.visit_pat(local.pat);
            },
            // An item binding:
            StmtKind::Item(_) => {
                println!("Item(item_id) = {};", current);
            },

            // Expr without trailing semi-colon (must have unit type):
            StmtKind::Expr(e) => {
                let e_pat = self.next("e");
                println!("Expr(ref {}, _) = {}", e_pat, current);

                self.current = e_pat;
                self.visit_expr(e);
            },

            // Expr with trailing semi-colon (may have any type):
            StmtKind::Semi(e) => {
                let e_pat = self.next("e");
                println!("Semi(ref {}, _) = {}", e_pat, current);

                self.current = e_pat;
                self.visit_expr(e);
            },
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

fn has_attr(cx: &LateContext<'_>, hir_id: hir::HirId) -> bool {
    let attrs = cx.tcx.hir().attrs(hir_id);
    get_attr(cx.sess(), attrs, "author").count() > 0
}

fn print_path(path: &QPath<'_>, first: &mut bool) {
    match *path {
        QPath::Resolved(_, path) => {
            for segment in path.segments {
                if *first {
                    *first = false;
                } else {
                    print!(", ");
                }
                print!("{:?}", segment.ident.as_str());
            }
        },
        QPath::TypeRelative(ty, segment) => match ty.kind {
            hir::TyKind::Path(ref inner_path) => {
                print_path(inner_path, first);
                if *first {
                    *first = false;
                } else {
                    print!(", ");
                }
                print!("{:?}", segment.ident.as_str());
            },
            ref other => print!("/* unimplemented: {:?}*/", other),
        },
        QPath::LangItem(..) => panic!("print_path: called for lang item qpath"),
    }
}
