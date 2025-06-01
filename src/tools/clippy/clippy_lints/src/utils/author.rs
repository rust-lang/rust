use clippy_utils::{MaybePath, get_attr, higher, path_def_id, sym};
use itertools::Itertools;
use rustc_ast::LitIntType;
use rustc_ast::ast::{LitFloatType, LitKind};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_hir::{
    self as hir, BindingMode, CaptureBy, Closure, ClosureKind, ConstArg, ConstArgKind, CoroutineKind, ExprKind,
    FnRetTy, HirId, Lit, PatExprKind, PatKind, QPath, StmtKind, StructTailExpr,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::{Ident, Symbol};
use std::cell::Cell;
use std::fmt::{Display, Formatter};

declare_lint_pass!(
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
    /// if ExprKind::If(ref cond, ref then, None) = item.kind
    ///     && let ExprKind::Binary(BinOp::Eq, ref left, ref right) = cond.kind
    ///     && let ExprKind::Path(ref path) = left.kind
    ///     && let ExprKind::Lit(ref lit) = right.kind
    ///     && let LitKind::Int(42, _) = lit.node
    /// {
    ///     // report your lint here
    /// }
    /// ```
    Author => []
);

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

/// Print a condition of a let chain, `chain!(self, "let Some(x) = y")` will print
/// `if let Some(x) = y` on the first call and `    && let Some(x) = y` thereafter
macro_rules! chain {
    ($self:ident, $($t:tt)*) => {
        if $self.first.take() {
            println!("if {}", format_args!($($t)*));
        } else {
            println!("    && {}", format_args!($($t)*));
        }
    }
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
    if let Some(body) = cx.tcx.hir_maybe_body_owned_by(hir_id.expect_owner().def_id) {
        check_node(cx, hir_id, |v| {
            v.expr(&v.bind("expr", body.value));
        });
    }
}

fn check_node(cx: &LateContext<'_>, hir_id: HirId, f: impl Fn(&PrintVisitor<'_, '_>)) {
    if has_attr(cx, hir_id) {
        f(&PrintVisitor::new(cx));
        println!("{{");
        println!("    // report your lint here");
        println!("}}");
    }
}

fn paths_static_name(cx: &LateContext<'_>, id: DefId) -> String {
    cx.get_def_path(id)
        .iter()
        .map(Symbol::as_str)
        .filter(|s| !s.starts_with('<'))
        .join("_")
        .to_uppercase()
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
    ids: Cell<FxHashMap<&'static str, u32>>,
    /// Currently at the first condition in the if chain
    first: Cell<bool>,
}

#[allow(clippy::unused_self)]
impl<'a, 'tcx> PrintVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            ids: Cell::default(),
            first: Cell::new(true),
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
            None => chain!(self, "{option}.is_none()"),
            Some(value) => {
                let value = &self.bind(name, value);
                chain!(self, "let Some({value}) = {option}");
                f(value);
            },
        }
    }

    fn slice<T>(&self, slice: &Binding<&[T]>, f: impl Fn(&Binding<&T>)) {
        if slice.value.is_empty() {
            chain!(self, "{slice}.is_empty()");
        } else {
            chain!(self, "{slice}.len() == {}", slice.value.len());
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
        chain!(self, "{ident}.as_str() == {:?}", ident.value.as_str());
    }

    fn symbol(&self, symbol: &Binding<Symbol>) {
        chain!(self, "{symbol}.as_str() == {:?}", symbol.value.as_str());
    }

    fn qpath<'p>(&self, qpath: &Binding<&QPath<'_>>, has_hir_id: &Binding<&impl MaybePath<'p>>) {
        if let QPath::LangItem(lang_item, ..) = *qpath.value {
            chain!(self, "matches!({qpath}, QPath::LangItem(LangItem::{lang_item:?}, _))");
        } else if let Some(def_id) = self.cx.qpath_res(qpath.value, has_hir_id.value.hir_id()).opt_def_id()
            && !def_id.is_local()
        {
            bind!(self, def_id);
            chain!(
                self,
                "let Some({def_id}) = cx.qpath_res({qpath}, {has_hir_id}.hir_id).opt_def_id()"
            );
            if let Some(name) = self.cx.tcx.get_diagnostic_name(def_id.value) {
                chain!(self, "cx.tcx.is_diagnostic_item(sym::{name}, {def_id})");
            } else {
                chain!(
                    self,
                    "paths::{}.matches(cx, {def_id}) // Add the path to `clippy_utils::paths` if needed",
                    paths_static_name(self.cx, def_id.value)
                );
            }
        }
    }

    fn maybe_path<'p>(&self, path: &Binding<&impl MaybePath<'p>>) {
        if let Some(id) = path_def_id(self.cx, path.value)
            && !id.is_local()
        {
            if let Some(lang) = self.cx.tcx.lang_items().from_def_id(id) {
                chain!(self, "is_path_lang_item(cx, {path}, LangItem::{}", lang.name());
            } else if let Some(name) = self.cx.tcx.get_diagnostic_name(id) {
                chain!(self, "is_path_diagnostic_item(cx, {path}, sym::{name})");
            } else {
                chain!(
                    self,
                    "paths::{}.matches_path(cx, {path}) // Add the path to `clippy_utils::paths` if needed",
                    paths_static_name(self.cx, id)
                );
            }
        }
    }

    fn const_arg(&self, const_arg: &Binding<&ConstArg<'_>>) {
        match const_arg.value.kind {
            ConstArgKind::Path(ref qpath) => {
                bind!(self, qpath);
                chain!(self, "let ConstArgKind::Path(ref {qpath}) = {const_arg}.kind");
            },
            ConstArgKind::Anon(anon_const) => {
                bind!(self, anon_const);
                chain!(self, "let ConstArgKind::Anon({anon_const}) = {const_arg}.kind");
                self.body(field!(anon_const.body));
            },
            ConstArgKind::Infer(..) => chain!(self, "let ConstArgKind::Infer(..) = {const_arg}.kind"),
        }
    }

    fn lit(&self, lit: &Binding<Lit>) {
        let kind = |kind| chain!(self, "let LitKind::{kind} = {lit}.node");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match lit.value.node {
            LitKind::Bool(val) => kind!("Bool({val:?})"),
            LitKind::Char(c) => kind!("Char({c:?})"),
            LitKind::Err(_) => kind!("Err"),
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
            LitKind::ByteStr(ref vec, _) => {
                bind!(self, vec);
                kind!("ByteStr(ref {vec})");
                chain!(self, "let [{:?}] = **{vec}", vec.value);
            },
            LitKind::CStr(ref vec, _) => {
                bind!(self, vec);
                kind!("CStr(ref {vec})");
                chain!(self, "let [{:?}] = **{vec}", vec.value);
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
            None => chain!(self, "{arm}.guard.is_none()"),
            Some(expr) => {
                bind!(self, expr);
                chain!(self, "let Some({expr}) = {arm}.guard");
                self.expr(expr);
            },
        }
        self.expr(field!(arm.body));
    }

    #[allow(clippy::too_many_lines)]
    fn expr(&self, expr: &Binding<&hir::Expr<'_>>) {
        if let Some(higher::While { condition, body, .. }) = higher::While::hir(expr.value) {
            bind!(self, condition, body);
            chain!(
                self,
                "let Some(higher::While {{ condition: {condition}, body: {body} }}) \
                = higher::While::hir({expr})"
            );
            self.expr(condition);
            self.expr(body);
            return;
        }

        if let Some(higher::WhileLet {
            let_pat,
            let_expr,
            if_then,
            ..
        }) = higher::WhileLet::hir(expr.value)
        {
            bind!(self, let_pat, let_expr, if_then);
            chain!(
                self,
                "let Some(higher::WhileLet {{ let_pat: {let_pat}, let_expr: {let_expr}, if_then: {if_then} }}) \
                = higher::WhileLet::hir({expr})"
            );
            self.pat(let_pat);
            self.expr(let_expr);
            self.expr(if_then);
            return;
        }

        if let Some(higher::ForLoop { pat, arg, body, .. }) = higher::ForLoop::hir(expr.value) {
            bind!(self, pat, arg, body);
            chain!(
                self,
                "let Some(higher::ForLoop {{ pat: {pat}, arg: {arg}, body: {body}, .. }}) \
                = higher::ForLoop::hir({expr})"
            );
            self.pat(pat);
            self.expr(arg);
            self.expr(body);
            return;
        }

        let kind = |kind| chain!(self, "let ExprKind::{kind} = {expr}.kind");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match expr.value.kind {
            ExprKind::Let(let_expr) => {
                bind!(self, let_expr);
                kind!("Let({let_expr})");
                self.pat(field!(let_expr.pat));
                if let Some(ty) = let_expr.value.ty {
                    bind!(self, ty);
                    chain!(self, "let Some({ty}) = {let_expr}.ty");
                    self.maybe_path(ty);
                }
                self.expr(field!(let_expr.init));
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
            ExprKind::Use(expr, _) => {
                bind!(self, expr);
                kind!("Use({expr})");
                self.expr(expr);
            },
            ExprKind::Binary(op, left, right) => {
                bind!(self, op, left, right);
                kind!("Binary({op}, {left}, {right})");
                chain!(self, "BinOpKind::{:?} == {op}.node", op.value.node);
                self.expr(left);
                self.expr(right);
            },
            ExprKind::Unary(op, inner) => {
                bind!(self, inner);
                kind!("Unary(UnOp::{op:?}, {inner})");
                self.expr(inner);
            },
            ExprKind::Lit(lit) => {
                bind!(self, lit);
                kind!("Lit(ref {lit})");
                self.lit(lit);
            },
            ExprKind::Cast(expr, cast_ty) => {
                bind!(self, expr, cast_ty);
                kind!("Cast({expr}, {cast_ty})");
                self.maybe_path(cast_ty);
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
                kind,
                ..
            }) => {
                let capture_clause = match capture_clause {
                    CaptureBy::Value { .. } => "Value { .. }",
                    CaptureBy::Use { .. } => "Use { .. }",
                    CaptureBy::Ref => "Ref",
                };

                let closure_kind = match kind {
                    ClosureKind::Closure => "ClosureKind::Closure".to_string(),
                    ClosureKind::Coroutine(coroutine_kind) => match coroutine_kind {
                        CoroutineKind::Desugared(desugaring, source) => format!(
                            "ClosureKind::Coroutine(CoroutineKind::Desugared(CoroutineDesugaring::{desugaring:?}, CoroutineSource::{source:?}))"
                        ),
                        CoroutineKind::Coroutine(movability) => {
                            format!("ClosureKind::Coroutine(CoroutineKind::Coroutine(Movability::{movability:?})")
                        },
                    },
                    ClosureKind::CoroutineClosure(desugaring) => {
                        format!("ClosureKind::CoroutineClosure(CoroutineDesugaring::{desugaring:?})")
                    },
                };

                let ret_ty = match fn_decl.output {
                    FnRetTy::DefaultReturn(_) => "FnRetTy::DefaultReturn(_)",
                    FnRetTy::Return(_) => "FnRetTy::Return(_ty)",
                };

                bind!(self, fn_decl, body_id);
                kind!(
                    "Closure {{ capture_clause: CaptureBy::{capture_clause}, fn_decl: {fn_decl}, body: {body_id}, closure_kind: {closure_kind}, .. }}"
                );
                chain!(self, "let {ret_ty} = {fn_decl}.output");
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
                chain!(self, "BinOpKind::{:?} == {op}.node", op.value.node);
                self.expr(target);
                self.expr(value);
            },
            ExprKind::Field(object, field_name) => {
                bind!(self, object, field_name);
                kind!("Field({object}, {field_name})");
                self.ident(field_name);
                self.expr(object);
            },
            ExprKind::Index(object, index, _) => {
                bind!(self, object, index);
                kind!("Index({object}, {index})");
                self.expr(object);
                self.expr(index);
            },
            ExprKind::Path(_) => {
                self.maybe_path(expr);
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
            ExprKind::Become(value) => {
                bind!(self, value);
                kind!("Become({value})");
                self.expr(value);
            },
            ExprKind::InlineAsm(_) => {
                kind!("InlineAsm(_)");
                out!("// unimplemented: `ExprKind::InlineAsm` is not further destructured at the moment");
            },
            ExprKind::OffsetOf(container, ref fields) => {
                bind!(self, container, fields);
                kind!("OffsetOf({container}, {fields})");
            },
            ExprKind::Struct(qpath, fields, base) => {
                bind!(self, qpath, fields);
                let base = OptionPat::new(match base {
                    StructTailExpr::Base(base) => Some(self.bind("base", base)),
                    StructTailExpr::None | StructTailExpr::DefaultFields(_) => None,
                });
                kind!("Struct({qpath}, {fields}, {base})");
                self.qpath(qpath, expr);
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
                self.const_arg(length);
            },
            ExprKind::Err(_) => kind!("Err(_)"),
            ExprKind::DropTemps(expr) => {
                bind!(self, expr);
                kind!("DropTemps({expr})");
                self.expr(expr);
            },
            ExprKind::UnsafeBinderCast(..) => {
                unimplemented!("unsafe binders are not implemented yet");
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
        let expr = self.cx.tcx.hir_body(body_id.value).value;
        bind!(self, expr);
        chain!(self, "{expr} = &cx.tcx.hir_body({body_id}).value");
        self.expr(expr);
    }

    fn pat_expr(&self, lit: &Binding<&hir::PatExpr<'_>>, pat: &Binding<&hir::Pat<'_>>) {
        let kind = |kind| chain!(self, "let PatExprKind::{kind} = {lit}.kind");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }
        match lit.value.kind {
            PatExprKind::Lit { lit, negated } => {
                bind!(self, lit);
                bind!(self, negated);
                kind!("Lit {{ ref {lit}, {negated} }}");
                self.lit(lit);
            },
            PatExprKind::ConstBlock(_) => kind!("ConstBlock(_)"),
            PatExprKind::Path(_) => self.maybe_path(pat),
        }
    }

    fn pat(&self, pat: &Binding<&hir::Pat<'_>>) {
        let kind = |kind| chain!(self, "let PatKind::{kind} = {pat}.kind");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match pat.value.kind {
            PatKind::Missing => unreachable!(),
            PatKind::Wild => kind!("Wild"),
            PatKind::Never => kind!("Never"),
            PatKind::Binding(ann, _, name, sub) => {
                bind!(self, name);
                opt_bind!(self, sub);
                let ann = match ann {
                    BindingMode::NONE => "NONE",
                    BindingMode::REF => "REF",
                    BindingMode::MUT => "MUT",
                    BindingMode::REF_MUT => "REF_MUT",
                    BindingMode::MUT_REF => "MUT_REF",
                    BindingMode::MUT_REF_MUT => "MUT_REF_MUT",
                };
                kind!("Binding(BindingMode::{ann}, _, {name}, {sub})");
                self.ident(name);
                sub.if_some(|p| self.pat(p));
            },
            PatKind::Struct(ref qpath, fields, ignore) => {
                bind!(self, qpath, fields);
                kind!("Struct(ref {qpath}, {fields}, {ignore})");
                self.qpath(qpath, pat);
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
                self.qpath(qpath, pat);
                self.slice(fields, |pat| self.pat(pat));
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
            PatKind::Deref(pat) => {
                bind!(self, pat);
                kind!("Deref({pat})");
                self.pat(pat);
            },
            PatKind::Ref(pat, muta) => {
                bind!(self, pat);
                kind!("Ref({pat}, Mutability::{muta:?})");
                self.pat(pat);
            },
            PatKind::Guard(pat, cond) => {
                bind!(self, pat, cond);
                kind!("Guard({pat}, {cond})");
                self.pat(pat);
                self.expr(cond);
            },
            PatKind::Expr(lit_expr) => {
                bind!(self, lit_expr);
                kind!("Expr({lit_expr})");
                self.pat_expr(lit_expr, pat);
            },
            PatKind::Range(start, end, end_kind) => {
                opt_bind!(self, start, end);
                kind!("Range({start}, {end}, RangeEnd::{end_kind:?})");
                start.if_some(|e| self.pat_expr(e, pat));
                end.if_some(|e| self.pat_expr(e, pat));
            },
            PatKind::Slice(start, middle, end) => {
                bind!(self, start, end);
                opt_bind!(self, middle);
                kind!("Slice({start}, {middle}, {end})");
                middle.if_some(|p| self.pat(p));
                self.slice(start, |pat| self.pat(pat));
                self.slice(end, |pat| self.pat(pat));
            },
            PatKind::Err(_) => kind!("Err"),
        }
    }

    fn stmt(&self, stmt: &Binding<&hir::Stmt<'_>>) {
        let kind = |kind| chain!(self, "let StmtKind::{kind} = {stmt}.kind");
        macro_rules! kind {
            ($($t:tt)*) => (kind(format_args!($($t)*)));
        }

        match stmt.value.kind {
            StmtKind::Let(local) => {
                bind!(self, local);
                kind!("Let({local})");
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

fn has_attr(cx: &LateContext<'_>, hir_id: HirId) -> bool {
    let attrs = cx.tcx.hir_attrs(hir_id);
    get_attr(cx.sess(), attrs, sym::author).count() > 0
}
