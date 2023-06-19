//! A pretty-printer for HIR.

use std::fmt::{self, Write};

use hir_expand::db::ExpandDatabase;
use syntax::ast::HasName;

use crate::{
    hir::{
        Array, BindingAnnotation, BindingId, CaptureBy, ClosureKind, Literal, LiteralOrConst,
        Movability, Statement,
    },
    pretty::{print_generic_args, print_path, print_type_ref},
    type_ref::TypeRef,
};

use super::*;

pub(super) fn print_body_hir(db: &dyn DefDatabase, body: &Body, owner: DefWithBodyId) -> String {
    let header = match owner {
        DefWithBodyId::FunctionId(it) => {
            let item_tree_id = it.lookup(db).id;
            format!(
                "fn {}",
                item_tree_id.item_tree(db)[item_tree_id.value].name.display(db.upcast())
            )
        }
        DefWithBodyId::StaticId(it) => {
            let item_tree_id = it.lookup(db).id;
            format!(
                "static {} = ",
                item_tree_id.item_tree(db)[item_tree_id.value].name.display(db.upcast())
            )
        }
        DefWithBodyId::ConstId(it) => {
            let item_tree_id = it.lookup(db).id;
            let name = match &item_tree_id.item_tree(db)[item_tree_id.value].name {
                Some(name) => name.display(db.upcast()).to_string(),
                None => "_".to_string(),
            };
            format!("const {name} = ")
        }
        DefWithBodyId::InTypeConstId(_) => format!("In type const = "),
        DefWithBodyId::VariantId(it) => {
            let src = it.parent.child_source(db);
            let variant = &src.value[it.local_id];
            match &variant.name() {
                Some(name) => name.to_string(),
                None => "_".to_string(),
            }
        }
    };

    let mut p =
        Printer { db: db.upcast(), body, buf: header, indent_level: 0, needs_indent: false };
    if let DefWithBodyId::FunctionId(it) = owner {
        p.buf.push('(');
        body.params.iter().zip(&db.function_data(it).params).for_each(|(&param, ty)| {
            p.print_pat(param);
            p.buf.push(':');
            p.print_type_ref(ty);
        });
        p.buf.push(')');
        p.buf.push(' ');
    }
    p.print_expr(body.body_expr);
    if matches!(owner, DefWithBodyId::StaticId(_) | DefWithBodyId::ConstId(_)) {
        p.buf.push(';');
    }
    p.buf
}

pub(super) fn print_expr_hir(
    db: &dyn DefDatabase,
    body: &Body,
    _owner: DefWithBodyId,
    expr: ExprId,
) -> String {
    let mut p =
        Printer { db: db.upcast(), body, buf: String::new(), indent_level: 0, needs_indent: false };
    p.print_expr(expr);
    p.buf
}

macro_rules! w {
    ($dst:expr, $($arg:tt)*) => {
        { let _ = write!($dst, $($arg)*); }
    };
}

macro_rules! wln {
    ($dst:expr) => {
        { let _ = writeln!($dst); }
    };
    ($dst:expr, $($arg:tt)*) => {
        { let _ = writeln!($dst, $($arg)*); }
    };
}

struct Printer<'a> {
    db: &'a dyn ExpandDatabase,
    body: &'a Body,
    buf: String,
    indent_level: usize,
    needs_indent: bool,
}

impl<'a> Write for Printer<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for line in s.split_inclusive('\n') {
            if self.needs_indent {
                match self.buf.chars().rev().find(|ch| *ch != ' ') {
                    Some('\n') | None => {}
                    _ => self.buf.push('\n'),
                }
                self.buf.push_str(&"    ".repeat(self.indent_level));
                self.needs_indent = false;
            }

            self.buf.push_str(line);
            self.needs_indent = line.ends_with('\n');
        }

        Ok(())
    }
}

impl<'a> Printer<'a> {
    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent_level += 1;
        wln!(self);
        f(self);
        self.indent_level -= 1;
        self.buf = self.buf.trim_end_matches('\n').to_string();
    }

    fn whitespace(&mut self) {
        match self.buf.chars().next_back() {
            None | Some('\n' | ' ') => {}
            _ => self.buf.push(' '),
        }
    }

    fn newline(&mut self) {
        match self.buf.chars().rev().find(|ch| *ch != ' ') {
            Some('\n') | None => {}
            _ => writeln!(self).unwrap(),
        }
    }

    fn print_expr(&mut self, expr: ExprId) {
        let expr = &self.body[expr];

        match expr {
            Expr::Missing => w!(self, "�"),
            Expr::Underscore => w!(self, "_"),
            Expr::Path(path) => self.print_path(path),
            Expr::If { condition, then_branch, else_branch } => {
                w!(self, "if ");
                self.print_expr(*condition);
                w!(self, " ");
                self.print_expr(*then_branch);
                if let Some(els) = *else_branch {
                    w!(self, " else ");
                    self.print_expr(els);
                }
            }
            Expr::Let { pat, expr } => {
                w!(self, "let ");
                self.print_pat(*pat);
                w!(self, " = ");
                self.print_expr(*expr);
            }
            Expr::Loop { body, label } => {
                if let Some(lbl) = label {
                    w!(self, "{}: ", self.body[*lbl].name.display(self.db));
                }
                w!(self, "loop ");
                self.print_expr(*body);
            }
            Expr::While { condition, body, label } => {
                if let Some(lbl) = label {
                    w!(self, "{}: ", self.body[*lbl].name.display(self.db));
                }
                w!(self, "while ");
                self.print_expr(*condition);
                self.print_expr(*body);
            }
            Expr::Call { callee, args, is_assignee_expr: _ } => {
                self.print_expr(*callee);
                w!(self, "(");
                if !args.is_empty() {
                    self.indented(|p| {
                        for arg in &**args {
                            p.print_expr(*arg);
                            wln!(p, ",");
                        }
                    });
                }
                w!(self, ")");
            }
            Expr::MethodCall { receiver, method_name, args, generic_args } => {
                self.print_expr(*receiver);
                w!(self, ".{}", method_name.display(self.db));
                if let Some(args) = generic_args {
                    w!(self, "::<");
                    print_generic_args(self.db, args, self).unwrap();
                    w!(self, ">");
                }
                w!(self, "(");
                if !args.is_empty() {
                    self.indented(|p| {
                        for arg in &**args {
                            p.print_expr(*arg);
                            wln!(p, ",");
                        }
                    });
                }
                w!(self, ")");
            }
            Expr::Match { expr, arms } => {
                w!(self, "match ");
                self.print_expr(*expr);
                w!(self, " {{");
                self.indented(|p| {
                    for arm in &**arms {
                        p.print_pat(arm.pat);
                        if let Some(guard) = arm.guard {
                            w!(p, " if ");
                            p.print_expr(guard);
                        }
                        w!(p, " => ");
                        p.print_expr(arm.expr);
                        wln!(p, ",");
                    }
                });
                wln!(self, "}}");
            }
            Expr::Continue { label } => {
                w!(self, "continue");
                if let Some(lbl) = label {
                    w!(self, " {}", self.body[*lbl].name.display(self.db));
                }
            }
            Expr::Break { expr, label } => {
                w!(self, "break");
                if let Some(lbl) = label {
                    w!(self, " {}", self.body[*lbl].name.display(self.db));
                }
                if let Some(expr) = expr {
                    self.whitespace();
                    self.print_expr(*expr);
                }
            }
            Expr::Return { expr } => {
                w!(self, "return");
                if let Some(expr) = expr {
                    self.whitespace();
                    self.print_expr(*expr);
                }
            }
            Expr::Yield { expr } => {
                w!(self, "yield");
                if let Some(expr) = expr {
                    self.whitespace();
                    self.print_expr(*expr);
                }
            }
            Expr::Yeet { expr } => {
                w!(self, "do");
                self.whitespace();
                w!(self, "yeet");
                if let Some(expr) = expr {
                    self.whitespace();
                    self.print_expr(*expr);
                }
            }
            Expr::RecordLit { path, fields, spread, ellipsis, is_assignee_expr: _ } => {
                match path {
                    Some(path) => self.print_path(path),
                    None => w!(self, "�"),
                }

                w!(self, "{{");
                self.indented(|p| {
                    for field in &**fields {
                        w!(p, "{}: ", field.name.display(self.db));
                        p.print_expr(field.expr);
                        wln!(p, ",");
                    }
                    if let Some(spread) = spread {
                        w!(p, "..");
                        p.print_expr(*spread);
                        wln!(p);
                    }
                    if *ellipsis {
                        wln!(p, "..");
                    }
                });
                w!(self, "}}");
            }
            Expr::Field { expr, name } => {
                self.print_expr(*expr);
                w!(self, ".{}", name.display(self.db));
            }
            Expr::Await { expr } => {
                self.print_expr(*expr);
                w!(self, ".await");
            }
            Expr::Cast { expr, type_ref } => {
                self.print_expr(*expr);
                w!(self, " as ");
                self.print_type_ref(type_ref);
            }
            Expr::Ref { expr, rawness, mutability } => {
                w!(self, "&");
                if rawness.is_raw() {
                    w!(self, "raw ");
                }
                if mutability.is_mut() {
                    w!(self, "mut ");
                }
                self.print_expr(*expr);
            }
            Expr::Box { expr } => {
                w!(self, "box ");
                self.print_expr(*expr);
            }
            Expr::UnaryOp { expr, op } => {
                let op = match op {
                    ast::UnaryOp::Deref => "*",
                    ast::UnaryOp::Not => "!",
                    ast::UnaryOp::Neg => "-",
                };
                w!(self, "{}", op);
                self.print_expr(*expr);
            }
            Expr::BinaryOp { lhs, rhs, op } => {
                let (bra, ket) = match op {
                    None | Some(ast::BinaryOp::Assignment { .. }) => ("", ""),
                    _ => ("(", ")"),
                };
                w!(self, "{}", bra);
                self.print_expr(*lhs);
                w!(self, "{} ", ket);
                match op {
                    Some(op) => w!(self, "{}", op),
                    None => w!(self, "�"), // :)
                }
                w!(self, " {}", bra);
                self.print_expr(*rhs);
                w!(self, "{}", ket);
            }
            Expr::Range { lhs, rhs, range_type } => {
                if let Some(lhs) = lhs {
                    w!(self, "(");
                    self.print_expr(*lhs);
                    w!(self, ") ");
                }
                let range = match range_type {
                    ast::RangeOp::Exclusive => "..",
                    ast::RangeOp::Inclusive => "..=",
                };
                w!(self, "{}", range);
                if let Some(rhs) = rhs {
                    w!(self, "(");
                    self.print_expr(*rhs);
                    w!(self, ") ");
                }
            }
            Expr::Index { base, index } => {
                self.print_expr(*base);
                w!(self, "[");
                self.print_expr(*index);
                w!(self, "]");
            }
            Expr::Closure { args, arg_types, ret_type, body, closure_kind, capture_by } => {
                match closure_kind {
                    ClosureKind::Generator(Movability::Static) => {
                        w!(self, "static ");
                    }
                    ClosureKind::Async => {
                        w!(self, "async ");
                    }
                    _ => (),
                }
                match capture_by {
                    CaptureBy::Value => {
                        w!(self, "move ");
                    }
                    CaptureBy::Ref => (),
                }
                w!(self, "|");
                for (i, (pat, ty)) in args.iter().zip(arg_types.iter()).enumerate() {
                    if i != 0 {
                        w!(self, ", ");
                    }
                    self.print_pat(*pat);
                    if let Some(ty) = ty {
                        w!(self, ": ");
                        self.print_type_ref(ty);
                    }
                }
                w!(self, "|");
                if let Some(ret_ty) = ret_type {
                    w!(self, " -> ");
                    self.print_type_ref(ret_ty);
                }
                self.whitespace();
                self.print_expr(*body);
            }
            Expr::Tuple { exprs, is_assignee_expr: _ } => {
                w!(self, "(");
                for expr in exprs.iter() {
                    self.print_expr(*expr);
                    w!(self, ", ");
                }
                w!(self, ")");
            }
            Expr::Array(arr) => {
                w!(self, "[");
                if !matches!(arr, Array::ElementList { elements, .. } if elements.is_empty()) {
                    self.indented(|p| match arr {
                        Array::ElementList { elements, is_assignee_expr: _ } => {
                            for elem in elements.iter() {
                                p.print_expr(*elem);
                                w!(p, ", ");
                            }
                        }
                        Array::Repeat { initializer, repeat } => {
                            p.print_expr(*initializer);
                            w!(p, "; ");
                            p.print_expr(*repeat);
                        }
                    });
                    self.newline();
                }
                w!(self, "]");
            }
            Expr::Literal(lit) => self.print_literal(lit),
            Expr::Block { id: _, statements, tail, label } => {
                let label = label.map(|lbl| format!("{}: ", self.body[lbl].name.display(self.db)));
                self.print_block(label.as_deref(), statements, tail);
            }
            Expr::Unsafe { id: _, statements, tail } => {
                self.print_block(Some("unsafe "), statements, tail);
            }
            Expr::Async { id: _, statements, tail } => {
                self.print_block(Some("async "), statements, tail);
            }
            Expr::Const(id) => {
                w!(self, "const {{ /* {id:?} */ }}");
            }
        }
    }

    fn print_block(
        &mut self,
        label: Option<&str>,
        statements: &[Statement],
        tail: &Option<la_arena::Idx<Expr>>,
    ) {
        self.whitespace();
        if let Some(lbl) = label {
            w!(self, "{}", lbl);
        }
        w!(self, "{{");
        if !statements.is_empty() || tail.is_some() {
            self.indented(|p| {
                for stmt in statements {
                    p.print_stmt(stmt);
                }
                if let Some(tail) = tail {
                    p.print_expr(*tail);
                }
                p.newline();
            });
        }
        w!(self, "}}");
    }

    fn print_pat(&mut self, pat: PatId) {
        let pat = &self.body[pat];

        match pat {
            Pat::Missing => w!(self, "�"),
            Pat::Wild => w!(self, "_"),
            Pat::Tuple { args, ellipsis } => {
                w!(self, "(");
                for (i, pat) in args.iter().enumerate() {
                    if i != 0 {
                        w!(self, ", ");
                    }
                    if *ellipsis == Some(i) {
                        w!(self, ".., ");
                    }
                    self.print_pat(*pat);
                }
                w!(self, ")");
            }
            Pat::Or(pats) => {
                for (i, pat) in pats.iter().enumerate() {
                    if i != 0 {
                        w!(self, " | ");
                    }
                    self.print_pat(*pat);
                }
            }
            Pat::Record { path, args, ellipsis } => {
                match path {
                    Some(path) => self.print_path(path),
                    None => w!(self, "�"),
                }

                w!(self, " {{");
                self.indented(|p| {
                    for arg in args.iter() {
                        w!(p, "{}: ", arg.name.display(self.db));
                        p.print_pat(arg.pat);
                        wln!(p, ",");
                    }
                    if *ellipsis {
                        wln!(p, "..");
                    }
                });
                w!(self, "}}");
            }
            Pat::Range { start, end } => {
                if let Some(start) = start {
                    self.print_literal_or_const(start);
                }
                w!(self, "..=");
                if let Some(end) = end {
                    self.print_literal_or_const(end);
                }
            }
            Pat::Slice { prefix, slice, suffix } => {
                w!(self, "[");
                for pat in prefix.iter() {
                    self.print_pat(*pat);
                    w!(self, ", ");
                }
                if let Some(pat) = slice {
                    self.print_pat(*pat);
                    w!(self, ", ");
                }
                for pat in suffix.iter() {
                    self.print_pat(*pat);
                    w!(self, ", ");
                }
                w!(self, "]");
            }
            Pat::Path(path) => self.print_path(path),
            Pat::Lit(expr) => self.print_expr(*expr),
            Pat::Bind { id, subpat } => {
                self.print_binding(*id);
                if let Some(pat) = subpat {
                    self.whitespace();
                    self.print_pat(*pat);
                }
            }
            Pat::TupleStruct { path, args, ellipsis } => {
                match path {
                    Some(path) => self.print_path(path),
                    None => w!(self, "�"),
                }
                w!(self, "(");
                for (i, arg) in args.iter().enumerate() {
                    if i != 0 {
                        w!(self, ", ");
                    }
                    if *ellipsis == Some(i) {
                        w!(self, ", ..");
                    }
                    self.print_pat(*arg);
                }
                w!(self, ")");
            }
            Pat::Ref { pat, mutability } => {
                w!(self, "&");
                if mutability.is_mut() {
                    w!(self, "mut ");
                }
                self.print_pat(*pat);
            }
            Pat::Box { inner } => {
                w!(self, "box ");
                self.print_pat(*inner);
            }
            Pat::ConstBlock(c) => {
                w!(self, "const ");
                self.print_expr(*c);
            }
        }
    }

    fn print_stmt(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Let { pat, type_ref, initializer, else_branch } => {
                w!(self, "let ");
                self.print_pat(*pat);
                if let Some(ty) = type_ref {
                    w!(self, ": ");
                    self.print_type_ref(ty);
                }
                if let Some(init) = initializer {
                    w!(self, " = ");
                    self.print_expr(*init);
                }
                if let Some(els) = else_branch {
                    w!(self, " else ");
                    self.print_expr(*els);
                }
                wln!(self, ";");
            }
            Statement::Expr { expr, has_semi } => {
                self.print_expr(*expr);
                if *has_semi {
                    w!(self, ";");
                }
                wln!(self);
            }
        }
    }

    fn print_literal_or_const(&mut self, literal_or_const: &LiteralOrConst) {
        match literal_or_const {
            LiteralOrConst::Literal(l) => self.print_literal(l),
            LiteralOrConst::Const(c) => self.print_path(c),
        }
    }

    fn print_literal(&mut self, literal: &Literal) {
        match literal {
            Literal::String(it) => w!(self, "{:?}", it),
            Literal::ByteString(it) => w!(self, "\"{}\"", it.escape_ascii()),
            Literal::CString(it) => w!(self, "\"{}\\0\"", it),
            Literal::Char(it) => w!(self, "'{}'", it.escape_debug()),
            Literal::Bool(it) => w!(self, "{}", it),
            Literal::Int(i, suffix) => {
                w!(self, "{}", i);
                if let Some(suffix) = suffix {
                    w!(self, "{}", suffix);
                }
            }
            Literal::Uint(i, suffix) => {
                w!(self, "{}", i);
                if let Some(suffix) = suffix {
                    w!(self, "{}", suffix);
                }
            }
            Literal::Float(f, suffix) => {
                w!(self, "{}", f);
                if let Some(suffix) = suffix {
                    w!(self, "{}", suffix);
                }
            }
        }
    }

    fn print_type_ref(&mut self, ty: &TypeRef) {
        print_type_ref(self.db, ty, self).unwrap();
    }

    fn print_path(&mut self, path: &Path) {
        print_path(self.db, path, self).unwrap();
    }

    fn print_binding(&mut self, id: BindingId) {
        let Binding { name, mode, .. } = &self.body.bindings[id];
        let mode = match mode {
            BindingAnnotation::Unannotated => "",
            BindingAnnotation::Mutable => "mut ",
            BindingAnnotation::Ref => "ref ",
            BindingAnnotation::RefMut => "ref mut ",
        };
        w!(self, "{}{}", mode, name.display(self.db));
    }
}
