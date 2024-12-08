//! A pretty-printer for HIR.

use std::fmt::{self, Write};

use itertools::Itertools;
use span::Edition;

use crate::{
    hir::{
        Array, BindingAnnotation, CaptureBy, ClosureKind, Literal, LiteralOrConst, Movability,
        Statement,
    },
    pretty::{print_generic_args, print_path, print_type_ref},
};

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LineFormat {
    Oneline,
    Newline,
    Indentation,
}

pub(super) fn print_body_hir(
    db: &dyn DefDatabase,
    body: &Body,
    owner: DefWithBodyId,
    edition: Edition,
) -> String {
    let header = match owner {
        DefWithBodyId::FunctionId(it) => it
            .lookup(db)
            .id
            .resolved(db, |it| format!("fn {}", it.name.display(db.upcast(), edition))),
        DefWithBodyId::StaticId(it) => it
            .lookup(db)
            .id
            .resolved(db, |it| format!("static {} = ", it.name.display(db.upcast(), edition))),
        DefWithBodyId::ConstId(it) => it.lookup(db).id.resolved(db, |it| {
            format!(
                "const {} = ",
                match &it.name {
                    Some(name) => name.display(db.upcast(), edition).to_string(),
                    None => "_".to_owned(),
                }
            )
        }),
        DefWithBodyId::InTypeConstId(_) => "In type const = ".to_owned(),
        DefWithBodyId::VariantId(it) => {
            let loc = it.lookup(db);
            let enum_loc = loc.parent.lookup(db);
            format!(
                "enum {}::{}",
                enum_loc.id.item_tree(db)[enum_loc.id.value].name.display(db.upcast(), edition),
                loc.id.item_tree(db)[loc.id.value].name.display(db.upcast(), edition),
            )
        }
    };

    let mut p = Printer {
        db,
        body,
        buf: header,
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    if let DefWithBodyId::FunctionId(it) = owner {
        p.buf.push('(');
        let function_data = db.function_data(it);
        let (mut params, ret_type) = (function_data.params.iter(), &function_data.ret_type);
        if let Some(self_param) = body.self_param {
            p.print_binding(self_param);
            p.buf.push_str(": ");
            if let Some(ty) = params.next() {
                p.print_type_ref(*ty, &function_data.types_map);
                p.buf.push_str(", ");
            }
        }
        body.params.iter().zip(params).for_each(|(&param, ty)| {
            p.print_pat(param);
            p.buf.push_str(": ");
            p.print_type_ref(*ty, &function_data.types_map);
            p.buf.push_str(", ");
        });
        // remove the last ", " in param list
        if body.params.len() > 0 {
            p.buf.truncate(p.buf.len() - 2);
        }
        p.buf.push(')');
        // return type
        p.buf.push_str(" -> ");
        p.print_type_ref(*ret_type, &function_data.types_map);
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
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        body,
        buf: String::new(),
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    p.print_expr(expr);
    p.buf
}

pub(super) fn print_pat_hir(
    db: &dyn DefDatabase,
    body: &Body,
    _owner: DefWithBodyId,
    pat: PatId,
    oneline: bool,
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        body,
        buf: String::new(),
        indent_level: 0,
        line_format: if oneline { LineFormat::Oneline } else { LineFormat::Newline },
        edition,
    };
    p.print_pat(pat);
    p.buf
}

macro_rules! w {
    ($dst:expr, $($arg:tt)*) => {
        { let _ = write!($dst, $($arg)*); }
    };
}

macro_rules! wln {
    ($dst:expr) => {
        { $dst.newline(); }
    };
    ($dst:expr, $($arg:tt)*) => {
        { let _ = w!($dst, $($arg)*); $dst.newline(); }
    };
}

struct Printer<'a> {
    db: &'a dyn DefDatabase,
    body: &'a Body,
    buf: String,
    indent_level: usize,
    line_format: LineFormat,
    edition: Edition,
}

impl Write for Printer<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for line in s.split_inclusive('\n') {
            if matches!(self.line_format, LineFormat::Indentation) {
                match self.buf.chars().rev().find(|ch| *ch != ' ') {
                    Some('\n') | None => {}
                    _ => self.buf.push('\n'),
                }
                self.buf.push_str(&"    ".repeat(self.indent_level));
            }

            self.buf.push_str(line);

            if matches!(self.line_format, LineFormat::Newline | LineFormat::Indentation) {
                self.line_format = if line.ends_with('\n') {
                    LineFormat::Indentation
                } else {
                    LineFormat::Newline
                };
            }
        }

        Ok(())
    }
}

impl Printer<'_> {
    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent_level += 1;
        wln!(self);
        f(self);
        self.indent_level -= 1;
        self.buf = self.buf.trim_end_matches('\n').to_owned();
    }

    fn whitespace(&mut self) {
        match self.buf.chars().next_back() {
            None | Some('\n' | ' ') => {}
            _ => self.buf.push(' '),
        }
    }

    // Add a newline if the current line is not empty.
    // If the current line is empty, add a space instead.
    //
    // Do not use [`writeln!()`] or [`wln!()`] here, which will result in
    // infinite recursive calls to this function.
    fn newline(&mut self) {
        if matches!(self.line_format, LineFormat::Oneline) {
            match self.buf.chars().last() {
                Some(' ') | None => {}
                Some(_) => {
                    w!(self, " ");
                }
            }
        } else {
            match self.buf.chars().rev().find_position(|ch| *ch != ' ') {
                Some((_, '\n')) | None => {}
                Some((idx, _)) => {
                    if idx != 0 {
                        self.buf.drain(self.buf.len() - idx..);
                    }
                    w!(self, "\n");
                }
            }
        }
    }

    fn print_expr(&mut self, expr: ExprId) {
        let expr = &self.body[expr];

        match expr {
            Expr::Missing => w!(self, "�"),
            Expr::Underscore => w!(self, "_"),
            Expr::InlineAsm(_) => w!(self, "builtin#asm(_)"),
            Expr::OffsetOf(offset_of) => {
                w!(self, "builtin#offset_of(");
                self.print_type_ref(offset_of.container, &self.body.types);
                let edition = self.edition;
                w!(
                    self,
                    ", {})",
                    offset_of
                        .fields
                        .iter()
                        .format_with(".", |field, f| f(&field.display(self.db.upcast(), edition)))
                );
            }
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
                    w!(self, "{}: ", self.body[*lbl].name.display(self.db.upcast(), self.edition));
                }
                w!(self, "loop ");
                self.print_expr(*body);
            }
            Expr::Call { callee, args } => {
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
                w!(self, ".{}", method_name.display(self.db.upcast(), self.edition));
                if let Some(args) = generic_args {
                    w!(self, "::<");
                    let edition = self.edition;
                    print_generic_args(self.db, args, &self.body.types, self, edition).unwrap();
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
                    w!(self, " {}", self.body[*lbl].name.display(self.db.upcast(), self.edition));
                }
            }
            Expr::Break { expr, label } => {
                w!(self, "break");
                if let Some(lbl) = label {
                    w!(self, " {}", self.body[*lbl].name.display(self.db.upcast(), self.edition));
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
            Expr::Become { expr } => {
                w!(self, "become");
                self.whitespace();
                self.print_expr(*expr);
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
            Expr::RecordLit { path, fields, spread } => {
                match path {
                    Some(path) => self.print_path(path),
                    None => w!(self, "�"),
                }

                w!(self, "{{");
                let edition = self.edition;
                self.indented(|p| {
                    for field in &**fields {
                        w!(p, "{}: ", field.name.display(self.db.upcast(), edition));
                        p.print_expr(field.expr);
                        wln!(p, ",");
                    }
                    if let Some(spread) = spread {
                        w!(p, "..");
                        p.print_expr(*spread);
                        wln!(p);
                    }
                });
                w!(self, "}}");
            }
            Expr::Field { expr, name } => {
                self.print_expr(*expr);
                w!(self, ".{}", name.display(self.db.upcast(), self.edition));
            }
            Expr::Await { expr } => {
                self.print_expr(*expr);
                w!(self, ".await");
            }
            Expr::Cast { expr, type_ref } => {
                self.print_expr(*expr);
                w!(self, " as ");
                self.print_type_ref(*type_ref, &self.body.types);
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
                    ClosureKind::Coroutine(Movability::Static) => {
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
                        self.print_type_ref(*ty, &self.body.types);
                    }
                }
                w!(self, "|");
                if let Some(ret_ty) = ret_type {
                    w!(self, " -> ");
                    self.print_type_ref(*ret_ty, &self.body.types);
                }
                self.whitespace();
                self.print_expr(*body);
            }
            Expr::Tuple { exprs } => {
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
                        Array::ElementList { elements } => {
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
                let label = label.map(|lbl| {
                    format!("{}: ", self.body[lbl].name.display(self.db.upcast(), self.edition))
                });
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
            &Expr::Assignment { target, value } => {
                self.print_pat(target);
                w!(self, " = ");
                self.print_expr(value);
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
                    if *ellipsis == Some(i as u32) {
                        w!(self, ".., ");
                    }
                    self.print_pat(*pat);
                }
                w!(self, ")");
            }
            Pat::Or(pats) => {
                w!(self, "(");
                for (i, pat) in pats.iter().enumerate() {
                    if i != 0 {
                        w!(self, " | ");
                    }
                    self.print_pat(*pat);
                }
                w!(self, ")");
            }
            Pat::Record { path, args, ellipsis } => {
                match path {
                    Some(path) => self.print_path(path),
                    None => w!(self, "�"),
                }

                w!(self, " {{");
                let edition = self.edition;
                let oneline = matches!(self.line_format, LineFormat::Oneline);
                self.indented(|p| {
                    for (idx, arg) in args.iter().enumerate() {
                        let field_name = arg.name.display(self.db.upcast(), edition).to_string();

                        let mut same_name = false;
                        if let Pat::Bind { id, subpat: None } = &self.body[arg.pat] {
                            if let Binding { name, mode: BindingAnnotation::Unannotated, .. } =
                                &self.body.bindings[*id]
                            {
                                if name.as_str() == field_name {
                                    same_name = true;
                                }
                            }
                        }

                        w!(p, "{}", field_name);

                        if !same_name {
                            w!(p, ": ");
                            p.print_pat(arg.pat);
                        }

                        // Do not print the extra comma if the line format is oneline
                        if oneline && idx == args.len() - 1 {
                            w!(p, " ");
                        } else {
                            wln!(p, ",");
                        }
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
                    if *ellipsis == Some(i as u32) {
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
            Pat::Expr(expr) => {
                self.print_expr(*expr);
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
                    self.print_type_ref(*ty, &self.body.types);
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
            Statement::Item(_) => (),
        }
    }

    fn print_literal_or_const(&mut self, literal_or_const: &LiteralOrConst) {
        match literal_or_const {
            LiteralOrConst::Literal(l) => self.print_literal(l),
            LiteralOrConst::Const(c) => self.print_pat(*c),
        }
    }

    fn print_literal(&mut self, literal: &Literal) {
        match literal {
            Literal::String(it) => w!(self, "{:?}", it),
            Literal::ByteString(it) => w!(self, "\"{}\"", it.escape_ascii()),
            Literal::CString(it) => w!(self, "\"{}\\0\"", it.escape_ascii()),
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

    fn print_type_ref(&mut self, ty: TypeRefId, map: &TypesMap) {
        let edition = self.edition;
        print_type_ref(self.db, ty, map, self, edition).unwrap();
    }

    fn print_path(&mut self, path: &Path) {
        let edition = self.edition;
        print_path(self.db, path, &self.body.types, self, edition).unwrap();
    }

    fn print_binding(&mut self, id: BindingId) {
        let Binding { name, mode, .. } = &self.body.bindings[id];
        let mode = match mode {
            BindingAnnotation::Unannotated => "",
            BindingAnnotation::Mutable => "mut ",
            BindingAnnotation::Ref => "ref ",
            BindingAnnotation::RefMut => "ref mut ",
        };
        w!(self, "{}{}", mode, name.display(self.db.upcast(), self.edition));
    }
}
