//! A group of attributes that can be attached to Rust code in order
//! to generate a clippy lint detecting said code automatically.

use clippy_utils::{get_attr, higher};
use rustc_ast::ast::{LitFloatType, LitKind};
use rustc_ast::LitIntType;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::{ArrayLen, Closure, ExprKind, FnRetTy, HirId, Lit, PatKind, QPath, StmtKind, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::{Ident, Symbol};
use std::fmt::{Display, Formatter, Write as _};

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

/// Writes a line of output with indentation added
macro_rules! out {
    ($($t:tt)*) => {
        println!("    {}", format_args!($($t)*))
    };
}

/// The variables passed in are replaced with `&Binding`s where the `value` field is set
/// to the original value of the variable. The `name` field is set to the name of the variable
/// (using `stringify!`) and is adjusted to avoid duplicate names.
/// Note that the `Binding` may be printed directly to output the `name`.
macro_rules! bind {
    ($self:ident $(, $name:ident)+) => {
        $(let $name = & $self.bind(stringify!($name), $name);)+
    };
}

/// Transforms the given `Option<T>` variables into `OptionPat<Binding<T>>`.
/// This displays as `Some($name)` or `None` when printed. The name of the inner binding
/// is set to the name of the variable passed to the macro.
macro_rules! opt_bind {
    ($self:ident $(, $name:ident)+) => {
        $(let $name = OptionPat::new($name.map(|o| $self.bind(stringify!($name), o)));)+
    };
}

/// Creates a `Binding` that accesses the field of an existing `Binding`
macro_rules! field {
    ($binding:ident.$field:ident) => {
        &Binding {
            name: $binding.name.to_string() + stringify!(.$field),
            value: $binding.value.$field,
        }
    };
}

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
        check_item(cx, item.hir_id());
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        check_item(cx, item.hir_id());
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        check_item(cx, item.hir_id());
    }

    fn check_arm(&mut self, cx: &LateContext<'tcx>, arm: &'tcx hir::Arm<'_>) {
        check_node(cx, arm.hir_id, |v| {
            v.arm(&v.bind("arm", arm));
        });
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        check_node(cx, expr.hir_id, |v| {
            v.expr(&v.bind("expr", expr));
        });
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx hir::Stmt<'_>) {
        match stmt.kind {
            StmtKind::Expr(e) | StmtKind::Semi(e) if has_attr(cx, e.hir_id) => return,
            _ => {},
        }
        check_node(cx, stmt.hir_id, |v| {
            v.stmt(&v.bind("stmt", stmt));
        });
    }
}

fn check_item(cx: &LateContext<'_>, hir_id: HirId) {
    let hir = cx.tcx.hir();
    if let Some(body_id) = hir.maybe_body_owned_by(hir_id.expect_owner()) {
        check_node(cx, hir_id, |v| {
            v.expr(&v.bind("expr", &hir.body(body_id).value));
        });
    }
}

fn check_node(cx: &LateContext<'_>, hir_id: HirId, f: impl Fn(&PrintVisitor<'_, '_>)) {
    if has_attr(cx, hir_id) {
        prelude();
        f(&PrintVisitor::new(cx));
        done();
    }
}

struct Binding<T> {
    name: String,
    value: T,
}

impl<T> Display for Binding<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

struct OptionPat<T> {
    pub opt: Option<T>,
}

impl<T> OptionPat<T> {
    fn new(opt: Option<T>) -> Self {
        Self { opt }
    }

    fn if_some(&self, f: impl Fn(&T)) {
        if let Some(t) = &self.opt {
            f(t);
        }
    }
}

impl<T: Display> Display for OptionPat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.opt {
            None => f.write_str("None"),
            Some(node) => write!(f, "Some({node})"),
        }
    }
}

struct PrintVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    /// Fields are the current index that needs to be appended to pattern
    /// binding names
    ids: std::cell::Cell<FxHashMap<&'static str, u32>>,
}

#[allow(clippy::unused_self)]
impl<'a, 'tcx> PrintVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            ids: std::cell::Cell::default(),
        }
    }

    fn next(&self, s: &'static str) -> String {
        let mut ids = self.ids.take();
        let out = match *ids.entry(s).and_modify(|n| *n += 1).or_default() {
            // first usage of the name, use it as is
            0 => s.to_string(),
            // append a number starting with 1
            n => format!("{s}{n}"),
        };
        self.ids.set(ids);
        out
    }

    fn bind<T>(&self, name: &'static str, value: T) -> Binding<T> {
        let name = self.next(name);
        Binding { name, value }
    }

    fn option<T: Copy>(&self, option: &Binding<Option<T>>, name: &'static str, f: impl Fn(&Binding<T>)) {
        match option.value {
            None => out!("if {option}.is_none();"),
            Some(value) => {
                let value = &self.bind(name, value);
                out!("if let Some({value}) = {option};");
                f(value);
            },
        }
    }

    fn slice<T>(&self, slice: &Binding<&[T]>, f: impl Fn(&Binding<&T>)) {
        if slice.value.is_empty() {
            out!("if {slice}.is_empty();");
        } else {
            out!("if {slice}.len() == {};", slice.value.len());
            for (i, value) in slice.value.iter().enumerate() {
                let name = format!("{slice}[{i}]");
                f(&Binding { name, value });
            }
        }
    }

    fn destination(&self, destination: &Binding<hir::Destination>) {
        self.option(field!(destination.label), "label", |label| {
            self.ident(field!(label.ident));
        });
    }

    fn ident(&self, ident: &Binding<Ident>) {
        out!("if {ident}.as_str() == {:?};", ident.value.as_str());
    }

    fn symbol(&self, symbol: &Binding<Symbol>) {
        out!("if {symbol}.as_str() == {:?};", symbol.value.as_str());
    }

    fn qpath(&self, qpath: &Binding<&QPath<'_>>) {
        if let QPath::LangItem(lang_item, ..) = *qpath.value {
            out!("if matches!({qpath}, QPath::LangItem(LangItem::{lang_item:?}, _));");
        } else {
            out!("if match_qpath({qpath}, &[{}]);", path_to_string(qpath.value));
        }
    }

    fn lit(&self, lit: &Binding<&Lit>) {
        let kind = |kind| out!("if let LitKind::{kind} = {lit}.node;");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match lit.value.node {
            LitKind::Bool(val) => kind!("Bool({val:?})"),
            LitKind::Char(c) => kind!("Char({c:?})"),
            LitKind::Err => kind!("Err"),
            LitKind::Byte(b) => kind!("Byte({b})"),
            LitKind::Int(i, suffix) => {
                let int_ty = match suffix {
                    LitIntType::Signed(int_ty) => format!("LitIntType::Signed(IntTy::{int_ty:?})"),
                    LitIntType::Unsigned(uint_ty) => format!("LitIntType::Unsigned(UintTy::{uint_ty:?})"),
                    LitIntType::Unsuffixed => String::from("LitIntType::Unsuffixed"),
                };
                kind!("Int({i}, {int_ty})");
            },
            LitKind::Float(_, suffix) => {
                let float_ty = match suffix {
                    LitFloatType::Suffixed(suffix_ty) => format!("LitFloatType::Suffixed(FloatTy::{suffix_ty:?})"),
                    LitFloatType::Unsuffixed => String::from("LitFloatType::Unsuffixed"),
                };
                kind!("Float(_, {float_ty})");
            },
            LitKind::ByteStr(ref vec) => {
                bind!(self, vec);
                kind!("ByteStr(ref {vec})");
                out!("if let [{:?}] = **{vec};", vec.value);
            },
            LitKind::Str(s, _) => {
                bind!(self, s);
                kind!("Str({s}, _)");
                self.symbol(s);
            },
        }
    }

    fn arm(&self, arm: &Binding<&hir::Arm<'_>>) {
        self.pat(field!(arm.pat));
        match arm.value.guard {
            None => out!("if {arm}.guard.is_none();"),
            Some(hir::Guard::If(expr)) => {
                bind!(self, expr);
                out!("if let Some(Guard::If({expr})) = {arm}.guard;");
                self.expr(expr);
            },
            Some(hir::Guard::IfLet(let_expr)) => {
                bind!(self, let_expr);
                out!("if let Some(Guard::IfLet({let_expr}) = {arm}.guard;");
                self.pat(field!(let_expr.pat));
                self.expr(field!(let_expr.init));
            },
        }
        self.expr(field!(arm.body));
    }

    #[allow(clippy::too_many_lines)]
    fn expr(&self, expr: &Binding<&hir::Expr<'_>>) {
        if let Some(higher::While { condition, body }) = higher::While::hir(expr.value) {
            bind!(self, condition, body);
            out!(
                "if let Some(higher::While {{ condition: {condition}, body: {body} }}) \
                = higher::While::hir({expr});"
            );
            self.expr(condition);
            self.expr(body);
            return;
        }

        if let Some(higher::WhileLet {
            let_pat,
            let_expr,
            if_then,
        }) = higher::WhileLet::hir(expr.value)
        {
            bind!(self, let_pat, let_expr, if_then);
            out!(
                "if let Some(higher::WhileLet {{ let_pat: {let_pat}, let_expr: {let_expr}, if_then: {if_then} }}) \
                = higher::WhileLet::hir({expr});"
            );
            self.pat(let_pat);
            self.expr(let_expr);
            self.expr(if_then);
            return;
        }

        if let Some(higher::ForLoop { pat, arg, body, .. }) = higher::ForLoop::hir(expr.value) {
            bind!(self, pat, arg, body);
            out!(
                "if let Some(higher::ForLoop {{ pat: {pat}, arg: {arg}, body: {body}, .. }}) \
                = higher::ForLoop::hir({expr});"
            );
            self.pat(pat);
            self.expr(arg);
            self.expr(body);
            return;
        }

        let kind = |kind| out!("if let ExprKind::{kind} = {expr}.kind;");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match expr.value.kind {
            ExprKind::Let(let_expr) => {
                bind!(self, let_expr);
                kind!("Let({let_expr})");
                self.pat(field!(let_expr.pat));
                // Does what ExprKind::Cast does, only adds a clause for the type
                // if it's a path
                if let Some(TyKind::Path(ref qpath)) = let_expr.value.ty.as_ref().map(|ty| &ty.kind) {
                    bind!(self, qpath);
                    out!("if let TyKind::Path(ref {qpath}) = {let_expr}.ty.kind;");
                    self.qpath(qpath);
                }
                self.expr(field!(let_expr.init));
            },
            ExprKind::Box(inner) => {
                bind!(self, inner);
                kind!("Box({inner})");
                self.expr(inner);
            },
            ExprKind::Array(elements) => {
                bind!(self, elements);
                kind!("Array({elements})");
                self.slice(elements, |e| self.expr(e));
            },
            ExprKind::Call(func, args) => {
                bind!(self, func, args);
                kind!("Call({func}, {args})");
                self.expr(func);
                self.slice(args, |e| self.expr(e));
            },
            ExprKind::MethodCall(method_name, receiver, args, _) => {
                bind!(self, method_name, receiver, args);
                kind!("MethodCall({method_name}, {receiver}, {args}, _)");
                self.ident(field!(method_name.ident));
                self.expr(receiver);
                self.slice(args, |e| self.expr(e));
            },
            ExprKind::Tup(elements) => {
                bind!(self, elements);
                kind!("Tup({elements})");
                self.slice(elements, |e| self.expr(e));
            },
            ExprKind::Binary(op, left, right) => {
                bind!(self, op, left, right);
                kind!("Binary({op}, {left}, {right})");
                out!("if BinOpKind::{:?} == {op}.node;", op.value.node);
                self.expr(left);
                self.expr(right);
            },
            ExprKind::Unary(op, inner) => {
                bind!(self, inner);
                kind!("Unary(UnOp::{op:?}, {inner})");
                self.expr(inner);
            },
            ExprKind::Lit(ref lit) => {
                bind!(self, lit);
                kind!("Lit(ref {lit})");
                self.lit(lit);
            },
            ExprKind::Cast(expr, cast_ty) => {
                bind!(self, expr, cast_ty);
                kind!("Cast({expr}, {cast_ty})");
                if let TyKind::Path(ref qpath) = cast_ty.value.kind {
                    bind!(self, qpath);
                    out!("if let TyKind::Path(ref {qpath}) = {cast_ty}.kind;");
                    self.qpath(qpath);
                }
                self.expr(expr);
            },
            ExprKind::Type(expr, _ty) => {
                bind!(self, expr);
                kind!("Type({expr}, _)");
                self.expr(expr);
            },
            ExprKind::Loop(body, label, des, _) => {
                bind!(self, body);
                opt_bind!(self, label);
                kind!("Loop({body}, {label}, LoopSource::{des:?}, _)");
                self.block(body);
                label.if_some(|l| self.ident(field!(l.ident)));
            },
            ExprKind::If(cond, then, else_expr) => {
                bind!(self, cond, then);
                opt_bind!(self, else_expr);
                kind!("If({cond}, {then}, {else_expr})");
                self.expr(cond);
                self.expr(then);
                else_expr.if_some(|e| self.expr(e));
            },
            ExprKind::Match(scrutinee, arms, des) => {
                bind!(self, scrutinee, arms);
                kind!("Match({scrutinee}, {arms}, MatchSource::{des:?})");
                self.expr(scrutinee);
                self.slice(arms, |arm| self.arm(arm));
            },
            ExprKind::Closure(&Closure {
                capture_clause,
                fn_decl,
                body: body_id,
                movability,
                ..
            }) => {
                let movability = OptionPat::new(movability.map(|m| format!("Movability::{m:?}")));

                let ret_ty = match fn_decl.output {
                    FnRetTy::DefaultReturn(_) => "FnRetTy::DefaultReturn(_)",
                    FnRetTy::Return(_) => "FnRetTy::Return(_ty)",
                };

                bind!(self, fn_decl, body_id);
                kind!("Closure(CaptureBy::{capture_clause:?}, {fn_decl}, {body_id}, _, {movability})");
                out!("if let {ret_ty} = {fn_decl}.output;");
                self.body(body_id);
            },
            ExprKind::Yield(sub, source) => {
                bind!(self, sub);
                kind!("Yield(sub, YieldSource::{source:?})");
                self.expr(sub);
            },
            ExprKind::Block(block, label) => {
                bind!(self, block);
                opt_bind!(self, label);
                kind!("Block({block}, {label})");
                self.block(block);
                label.if_some(|l| self.ident(field!(l.ident)));
            },
            ExprKind::Assign(target, value, _) => {
                bind!(self, target, value);
                kind!("Assign({target}, {value}, _span)");
                self.expr(target);
                self.expr(value);
            },
            ExprKind::AssignOp(op, target, value) => {
                bind!(self, op, target, value);
                kind!("AssignOp({op}, {target}, {value})");
                out!("if BinOpKind::{:?} == {op}.node;", op.value.node);
                self.expr(target);
                self.expr(value);
            },
            ExprKind::Field(object, field_name) => {
                bind!(self, object, field_name);
                kind!("Field({object}, {field_name})");
                self.ident(field_name);
                self.expr(object);
            },
            ExprKind::Index(object, index) => {
                bind!(self, object, index);
                kind!("Index({object}, {index})");
                self.expr(object);
                self.expr(index);
            },
            ExprKind::Path(ref qpath) => {
                bind!(self, qpath);
                kind!("Path(ref {qpath})");
                self.qpath(qpath);
            },
            ExprKind::AddrOf(kind, mutability, inner) => {
                bind!(self, inner);
                kind!("AddrOf(BorrowKind::{kind:?}, Mutability::{mutability:?}, {inner})");
                self.expr(inner);
            },
            ExprKind::Break(destination, value) => {
                bind!(self, destination);
                opt_bind!(self, value);
                kind!("Break({destination}, {value})");
                self.destination(destination);
                value.if_some(|e| self.expr(e));
            },
            ExprKind::Continue(destination) => {
                bind!(self, destination);
                kind!("Continue({destination})");
                self.destination(destination);
            },
            ExprKind::Ret(value) => {
                opt_bind!(self, value);
                kind!("Ret({value})");
                value.if_some(|e| self.expr(e));
            },
            ExprKind::InlineAsm(_) => {
                kind!("InlineAsm(_)");
                out!("// unimplemented: `ExprKind::InlineAsm` is not further destructured at the moment");
            },
            ExprKind::Struct(qpath, fields, base) => {
                bind!(self, qpath, fields);
                opt_bind!(self, base);
                kind!("Struct({qpath}, {fields}, {base})");
                self.qpath(qpath);
                self.slice(fields, |field| {
                    self.ident(field!(field.ident));
                    self.expr(field!(field.expr));
                });
                base.if_some(|e| self.expr(e));
            },
            ExprKind::ConstBlock(_) => kind!("ConstBlock(_)"),
            ExprKind::Repeat(value, length) => {
                bind!(self, value, length);
                kind!("Repeat({value}, {length})");
                self.expr(value);
                match length.value {
                    ArrayLen::Infer(..) => out!("if let ArrayLen::Infer(..) = length;"),
                    ArrayLen::Body(anon_const) => {
                        bind!(self, anon_const);
                        out!("if let ArrayLen::Body({anon_const}) = {length};");
                        self.body(field!(anon_const.body));
                    },
                }
            },
            ExprKind::Err => kind!("Err"),
            ExprKind::DropTemps(expr) => {
                bind!(self, expr);
                kind!("DropTemps({expr})");
                self.expr(expr);
            },
        }
    }

    fn block(&self, block: &Binding<&hir::Block<'_>>) {
        self.slice(field!(block.stmts), |stmt| self.stmt(stmt));
        self.option(field!(block.expr), "trailing_expr", |expr| {
            self.expr(expr);
        });
    }

    fn body(&self, body_id: &Binding<hir::BodyId>) {
        let expr = self.cx.tcx.hir().body(body_id.value).value;
        bind!(self, expr);
        out!("let {expr} = &cx.tcx.hir().body({body_id}).value;");
        self.expr(expr);
    }

    fn pat(&self, pat: &Binding<&hir::Pat<'_>>) {
        let kind = |kind| out!("if let PatKind::{kind} = {pat}.kind;");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match pat.value.kind {
            PatKind::Wild => kind!("Wild"),
            PatKind::Binding(anno, .., name, sub) => {
                bind!(self, name);
                opt_bind!(self, sub);
                kind!("Binding(BindingAnnotation::{anno:?}, _, {name}, {sub})");
                self.ident(name);
                sub.if_some(|p| self.pat(p));
            },
            PatKind::Struct(ref qpath, fields, ignore) => {
                bind!(self, qpath, fields);
                kind!("Struct(ref {qpath}, {fields}, {ignore})");
                self.qpath(qpath);
                self.slice(fields, |field| {
                    self.ident(field!(field.ident));
                    self.pat(field!(field.pat));
                });
            },
            PatKind::Or(fields) => {
                bind!(self, fields);
                kind!("Or({fields})");
                self.slice(fields, |pat| self.pat(pat));
            },
            PatKind::TupleStruct(ref qpath, fields, skip_pos) => {
                bind!(self, qpath, fields);
                kind!("TupleStruct(ref {qpath}, {fields}, {skip_pos:?})");
                self.qpath(qpath);
                self.slice(fields, |pat| self.pat(pat));
            },
            PatKind::Path(ref qpath) => {
                bind!(self, qpath);
                kind!("Path(ref {qpath})");
                self.qpath(qpath);
            },
            PatKind::Tuple(fields, skip_pos) => {
                bind!(self, fields);
                kind!("Tuple({fields}, {skip_pos:?})");
                self.slice(fields, |field| self.pat(field));
            },
            PatKind::Box(pat) => {
                bind!(self, pat);
                kind!("Box({pat})");
                self.pat(pat);
            },
            PatKind::Ref(pat, muta) => {
                bind!(self, pat);
                kind!("Ref({pat}, Mutability::{muta:?})");
                self.pat(pat);
            },
            PatKind::Lit(lit_expr) => {
                bind!(self, lit_expr);
                kind!("Lit({lit_expr})");
                self.expr(lit_expr);
            },
            PatKind::Range(start, end, end_kind) => {
                opt_bind!(self, start, end);
                kind!("Range({start}, {end}, RangeEnd::{end_kind:?})");
                start.if_some(|e| self.expr(e));
                end.if_some(|e| self.expr(e));
            },
            PatKind::Slice(start, middle, end) => {
                bind!(self, start, end);
                opt_bind!(self, middle);
                kind!("Slice({start}, {middle}, {end})");
                middle.if_some(|p| self.pat(p));
                self.slice(start, |pat| self.pat(pat));
                self.slice(end, |pat| self.pat(pat));
            },
        }
    }

    fn stmt(&self, stmt: &Binding<&hir::Stmt<'_>>) {
        let kind = |kind| out!("if let StmtKind::{kind} = {stmt}.kind;");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match stmt.value.kind {
            StmtKind::Local(local) => {
                bind!(self, local);
                kind!("Local({local})");
                self.option(field!(local.init), "init", |init| {
                    self.expr(init);
                });
                self.pat(field!(local.pat));
            },
            StmtKind::Item(_) => kind!("Item(item_id)"),
            StmtKind::Expr(e) => {
                bind!(self, e);
                kind!("Expr({e})");
                self.expr(e);
            },
            StmtKind::Semi(e) => {
                bind!(self, e);
                kind!("Semi({e})");
                self.expr(e);
            },
        }
    }
}

fn has_attr(cx: &LateContext<'_>, hir_id: hir::HirId) -> bool {
    let attrs = cx.tcx.hir().attrs(hir_id);
    get_attr(cx.sess(), attrs, "author").count() > 0
}

fn path_to_string(path: &QPath<'_>) -> String {
    fn inner(s: &mut String, path: &QPath<'_>) {
        match *path {
            QPath::Resolved(_, path) => {
                for (i, segment) in path.segments.iter().enumerate() {
                    if i > 0 {
                        *s += ", ";
                    }
                    write!(s, "{:?}", segment.ident.as_str()).unwrap();
                }
            },
            QPath::TypeRelative(ty, segment) => match &ty.kind {
                hir::TyKind::Path(inner_path) => {
                    inner(s, inner_path);
                    *s += ", ";
                    write!(s, "{:?}", segment.ident.as_str()).unwrap();
                },
                other => write!(s, "/* unimplemented: {:?}*/", other).unwrap(),
            },
            QPath::LangItem(..) => panic!("path_to_string: called for lang item qpath"),
        }
    }
    let mut s = String::new();
    inner(&mut s, path);
    s
}
