//! A pretty-printer for HIR.
#![allow(dead_code)]

use std::{
    fmt::{self, Write},
    mem,
};

use hir_expand::{Lookup, mod_path::PathKind};
use itertools::Itertools;
use span::Edition;
use syntax::ast::HasName;

use crate::{
    AdtId, DefWithBodyId, GenericDefId, TypeParamId, VariantId,
    expr_store::path::{GenericArg, GenericArgs},
    hir::{
        Array, BindingAnnotation, CaptureBy, ClosureKind, Literal, Movability, Statement,
        generics::{GenericParams, WherePredicate},
    },
    lang_item::LangItemTarget,
    signatures::{FnFlags, FunctionSignature, StructSignature},
    src::HasSource,
    type_ref::{ConstRef, LifetimeRef, Mutability, TraitBoundModifier, TypeBound, UseArgRef},
};
use crate::{LifetimeParamId, signatures::StructFlags};
use crate::{item_tree::FieldsShape, signatures::FieldData};

use super::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineFormat {
    Oneline,
    Newline,
    Indentation,
}

fn item_name<Id, Loc>(db: &dyn DefDatabase, id: Id, default: &str) -> String
where
    Id: Lookup<Database = dyn DefDatabase, Data = Loc>,
    Loc: HasSource,
    Loc::Value: ast::HasName,
{
    let loc = id.lookup(db);
    let source = loc.source(db);
    source.value.name().map_or_else(|| default.to_owned(), |name| name.to_string())
}

pub fn print_body_hir(
    db: &dyn DefDatabase,
    body: &Body,
    owner: DefWithBodyId,
    edition: Edition,
) -> String {
    let header = match owner {
        DefWithBodyId::FunctionId(it) => format!("fn {}", item_name(db, it, "<missing>")),
        DefWithBodyId::StaticId(it) => format!("static {} = ", item_name(db, it, "<missing>")),
        DefWithBodyId::ConstId(it) => format!("const {} = ", item_name(db, it, "_")),
        DefWithBodyId::VariantId(it) => format!(
            "enum {}::{}",
            item_name(db, it.lookup(db).parent, "<missing>"),
            item_name(db, it, "<missing>")
        ),
    };

    let mut p = Printer {
        db,
        store: body,
        buf: header,
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    if let DefWithBodyId::FunctionId(_) = owner {
        p.buf.push('(');
        if let Some(self_param) = body.self_param {
            p.print_binding(self_param);
            p.buf.push_str(", ");
        }
        body.params.iter().for_each(|param| {
            p.print_pat(*param);
            p.buf.push_str(", ");
        });
        // remove the last ", " in param list
        if !body.params.is_empty() {
            p.buf.truncate(p.buf.len() - 2);
        }
        p.buf.push(')');
        p.buf.push(' ');
    }
    p.print_expr(body.body_expr);
    if matches!(owner, DefWithBodyId::StaticId(_) | DefWithBodyId::ConstId(_)) {
        p.buf.push(';');
    }
    p.buf
}

pub fn print_variant_body_hir(db: &dyn DefDatabase, owner: VariantId, edition: Edition) -> String {
    let header = match owner {
        VariantId::StructId(it) => format!("struct {}", item_name(db, it, "<missing>")),
        VariantId::EnumVariantId(it) => format!(
            "enum {}::{}",
            item_name(db, it.lookup(db).parent, "<missing>"),
            item_name(db, it, "<missing>")
        ),
        VariantId::UnionId(it) => format!("union {}", item_name(db, it, "<missing>")),
    };

    let fields = db.variant_fields(owner);

    let mut p = Printer {
        db,
        store: &fields.store,
        buf: header,
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    match fields.shape {
        FieldsShape::Record => wln!(p, " {{"),
        FieldsShape::Tuple => wln!(p, "("),
        FieldsShape::Unit => (),
    }

    for (_, data) in fields.fields().iter() {
        let FieldData { name, type_ref, visibility, is_unsafe } = data;
        match visibility {
            crate::item_tree::RawVisibility::Module(interned, _visibility_explicitness) => {
                w!(p, "pub(in {})", interned.display(db, p.edition))
            }
            crate::item_tree::RawVisibility::Public => w!(p, "pub "),
            crate::item_tree::RawVisibility::PubCrate => w!(p, "pub(crate) "),
            crate::item_tree::RawVisibility::PubSelf(_) => w!(p, "pub(self) "),
        }
        if *is_unsafe {
            w!(p, "unsafe ");
        }
        w!(p, "{}: ", name.display(db, p.edition));
        p.print_type_ref(*type_ref);
    }

    match fields.shape {
        FieldsShape::Record => wln!(p, "}}"),
        FieldsShape::Tuple => wln!(p, ");"),
        FieldsShape::Unit => wln!(p, ";"),
    }
    p.buf
}

pub fn print_signature(db: &dyn DefDatabase, owner: GenericDefId, edition: Edition) -> String {
    match owner {
        GenericDefId::AdtId(id) => match id {
            AdtId::StructId(id) => {
                let signature = db.struct_signature(id);
                print_struct(db, &signature, edition)
            }
            AdtId::UnionId(id) => {
                format!("unimplemented {id:?}")
            }
            AdtId::EnumId(id) => {
                format!("unimplemented {id:?}")
            }
        },
        GenericDefId::ConstId(id) => format!("unimplemented {id:?}"),
        GenericDefId::FunctionId(id) => {
            let signature = db.function_signature(id);
            print_function(db, &signature, edition)
        }
        GenericDefId::ImplId(id) => format!("unimplemented {id:?}"),
        GenericDefId::StaticId(id) => format!("unimplemented {id:?}"),
        GenericDefId::TraitAliasId(id) => format!("unimplemented {id:?}"),
        GenericDefId::TraitId(id) => format!("unimplemented {id:?}"),
        GenericDefId::TypeAliasId(id) => format!("unimplemented {id:?}"),
    }
}

pub fn print_path(
    db: &dyn DefDatabase,
    store: &ExpressionStore,
    path: &Path,
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        store,
        buf: String::new(),
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    p.print_path(path);
    p.buf
}

pub fn print_struct(
    db: &dyn DefDatabase,
    StructSignature { name, generic_params, store, flags, shape, repr }: &StructSignature,
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        store,
        buf: String::new(),
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    if let Some(repr) = repr {
        if repr.c() {
            wln!(p, "#[repr(C)]");
        }
        if let Some(align) = repr.align {
            wln!(p, "#[repr(align({}))]", align.bytes());
        }
        if let Some(pack) = repr.pack {
            wln!(p, "#[repr(pack({}))]", pack.bytes());
        }
    }
    if flags.contains(StructFlags::FUNDAMENTAL) {
        wln!(p, "#[fundamental]");
    }
    w!(p, "struct ");
    w!(p, "{}", name.display(db, edition));
    print_generic_params(db, generic_params, &mut p);
    match shape {
        FieldsShape::Record => wln!(p, " {{...}}"),
        FieldsShape::Tuple => wln!(p, "(...)"),
        FieldsShape::Unit => (),
    }

    print_where_clauses(db, generic_params, &mut p);

    match shape {
        FieldsShape::Record => wln!(p),
        FieldsShape::Tuple => wln!(p, ";"),
        FieldsShape::Unit => wln!(p, ";"),
    }

    p.buf
}

pub fn print_function(
    db: &dyn DefDatabase,
    FunctionSignature {
        name,
        generic_params,
        store,
        params,
        ret_type,
        abi,
        flags,
        legacy_const_generics_indices,
    }: &FunctionSignature,
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        store,
        buf: String::new(),
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    if flags.contains(FnFlags::CONST) {
        w!(p, "const ");
    }
    if flags.contains(FnFlags::ASYNC) {
        w!(p, "async ");
    }
    if flags.contains(FnFlags::UNSAFE) {
        w!(p, "unsafe ");
    }
    if flags.contains(FnFlags::EXPLICIT_SAFE) {
        w!(p, "safe ");
    }
    if let Some(abi) = abi {
        w!(p, "extern \"{}\" ", abi.as_str());
    }
    w!(p, "fn ");
    w!(p, "{}", name.display(db, edition));
    print_generic_params(db, generic_params, &mut p);
    w!(p, "(");
    for (i, param) in params.iter().enumerate() {
        if i != 0 {
            w!(p, ", ");
        }
        if legacy_const_generics_indices.as_ref().is_some_and(|idx| idx.contains(&(i as u32))) {
            w!(p, "const: ");
        }
        p.print_type_ref(*param);
    }
    w!(p, ")");
    if let Some(ret_type) = ret_type {
        w!(p, " -> ");
        p.print_type_ref(*ret_type);
    }

    print_where_clauses(db, generic_params, &mut p);
    wln!(p, " {{...}}");

    p.buf
}

fn print_where_clauses(db: &dyn DefDatabase, generic_params: &GenericParams, p: &mut Printer<'_>) {
    if !generic_params.where_predicates.is_empty() {
        w!(p, "\nwhere\n");
        p.indented(|p| {
            for (i, pred) in generic_params.where_predicates.iter().enumerate() {
                if i != 0 {
                    w!(p, ",\n");
                }
                match pred {
                    WherePredicate::TypeBound { target, bound } => {
                        p.print_type_ref(*target);
                        w!(p, ": ");
                        p.print_type_bounds(std::slice::from_ref(bound));
                    }
                    WherePredicate::Lifetime { target, bound } => {
                        p.print_lifetime_ref(*target);
                        w!(p, ": ");
                        p.print_lifetime_ref(*bound);
                    }
                    WherePredicate::ForLifetime { lifetimes, target, bound } => {
                        w!(p, "for<");
                        for (i, lifetime) in lifetimes.iter().enumerate() {
                            if i != 0 {
                                w!(p, ", ");
                            }
                            w!(p, "{}", lifetime.display(db, p.edition));
                        }
                        w!(p, "> ");
                        p.print_type_ref(*target);
                        w!(p, ": ");
                        p.print_type_bounds(std::slice::from_ref(bound));
                    }
                }
            }
        });
        wln!(p);
    }
}

fn print_generic_params(db: &dyn DefDatabase, generic_params: &GenericParams, p: &mut Printer<'_>) {
    if !generic_params.is_empty() {
        w!(p, "<");
        let mut first = true;
        for (_i, param) in generic_params.iter_lt() {
            if !first {
                w!(p, ", ");
            }
            first = false;
            w!(p, "{}", param.name.display(db, p.edition));
        }
        for (i, param) in generic_params.iter_type_or_consts() {
            if !first {
                w!(p, ", ");
            }
            first = false;
            if let Some(const_param) = param.const_param() {
                w!(p, "const {}: ", const_param.name.display(db, p.edition));
                p.print_type_ref(const_param.ty);
                if let Some(default) = const_param.default {
                    w!(p, " = ");
                    p.print_expr(default.expr);
                }
            }
            if let Some(type_param) = param.type_param() {
                match &type_param.name {
                    Some(name) => w!(p, "{}", name.display(db, p.edition)),
                    None => w!(p, "Param[{}]", i.into_raw()),
                }
                if let Some(default) = type_param.default {
                    w!(p, " = ");
                    p.print_type_ref(default);
                }
            }
        }
        w!(p, ">");
    }
}

pub fn print_expr_hir(
    db: &dyn DefDatabase,
    store: &ExpressionStore,
    _owner: DefWithBodyId,
    expr: ExprId,
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        store,
        buf: String::new(),
        indent_level: 0,
        line_format: LineFormat::Newline,
        edition,
    };
    p.print_expr(expr);
    p.buf
}

pub fn print_pat_hir(
    db: &dyn DefDatabase,
    store: &ExpressionStore,
    _owner: DefWithBodyId,
    pat: PatId,
    oneline: bool,
    edition: Edition,
) -> String {
    let mut p = Printer {
        db,
        store,
        buf: String::new(),
        indent_level: 0,
        line_format: if oneline { LineFormat::Oneline } else { LineFormat::Newline },
        edition,
    };
    p.print_pat(pat);
    p.buf
}

struct Printer<'a> {
    db: &'a dyn DefDatabase,
    store: &'a ExpressionStore,
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
        let expr = &self.store[expr];

        match expr {
            Expr::Missing => w!(self, "�"),
            Expr::Underscore => w!(self, "_"),
            Expr::InlineAsm(_) => w!(self, "builtin#asm(_)"),
            Expr::OffsetOf(offset_of) => {
                w!(self, "builtin#offset_of(");
                self.print_type_ref(offset_of.container);
                let edition = self.edition;
                w!(
                    self,
                    ", {})",
                    offset_of
                        .fields
                        .iter()
                        .format_with(".", |field, f| f(&field.display(self.db, edition)))
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
                    w!(self, "{}: ", self.store[*lbl].name.display(self.db, self.edition));
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
                w!(self, ".{}", method_name.display(self.db, self.edition));
                if let Some(args) = generic_args {
                    w!(self, "::<");
                    self.print_generic_args(args);
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
                    w!(self, " {}", self.store[*lbl].name.display(self.db, self.edition));
                }
            }
            Expr::Break { expr, label } => {
                w!(self, "break");
                if let Some(lbl) = label {
                    w!(self, " {}", self.store[*lbl].name.display(self.db, self.edition));
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
                        w!(p, "{}: ", field.name.display(self.db, edition));
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
                w!(self, ".{}", name.display(self.db, self.edition));
            }
            Expr::Await { expr } => {
                self.print_expr(*expr);
                w!(self, ".await");
            }
            Expr::Cast { expr, type_ref } => {
                self.print_expr(*expr);
                w!(self, " as ");
                self.print_type_ref(*type_ref);
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
                        self.print_type_ref(*ty);
                    }
                }
                w!(self, "|");
                if let Some(ret_ty) = ret_type {
                    w!(self, " -> ");
                    self.print_type_ref(*ret_ty);
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
                    format!("{}: ", self.store[lbl].name.display(self.db, self.edition))
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
        let pat = &self.store[pat];

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
                        let field_name = arg.name.display(self.db, edition).to_string();

                        let mut same_name = false;
                        if let Pat::Bind { id, subpat: None } = &self.store[arg.pat] {
                            if let Binding { name, mode: BindingAnnotation::Unannotated, .. } =
                                &self.store.bindings[*id]
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
                    self.print_expr(*start);
                }
                w!(self, "..=");
                if let Some(end) = end {
                    self.print_expr(*end);
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
                    w!(self, "@ ");
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
                    self.print_type_ref(*ty);
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

    fn print_binding(&mut self, id: BindingId) {
        let Binding { name, mode, .. } = &self.store.bindings[id];
        let mode = match mode {
            BindingAnnotation::Unannotated => "",
            BindingAnnotation::Mutable => "mut ",
            BindingAnnotation::Ref => "ref ",
            BindingAnnotation::RefMut => "ref mut ",
        };
        w!(self, "{}{}", mode, name.display(self.db, self.edition));
    }

    fn print_path(&mut self, path: &Path) {
        if let Path::LangItem(it, s) = path {
            w!(self, "builtin#lang(");
            macro_rules! write_name {
                ($it:ident) => {{
                    w!(self, "{}", item_name(self.db, $it, "<missing>"));
                }};
            }
            match *it {
                LangItemTarget::ImplDef(it) => w!(self, "{it:?}"),
                LangItemTarget::EnumId(it) => write_name!(it),
                LangItemTarget::Function(it) => write_name!(it),
                LangItemTarget::Static(it) => write_name!(it),
                LangItemTarget::Struct(it) => write_name!(it),
                LangItemTarget::Union(it) => write_name!(it),
                LangItemTarget::TypeAlias(it) => write_name!(it),
                LangItemTarget::Trait(it) => write_name!(it),
                LangItemTarget::EnumVariant(it) => write_name!(it),
            }

            if let Some(s) = s {
                w!(self, "::{}", s.display(self.db, self.edition));
            }
            return w!(self, ")");
        }
        match path.type_anchor() {
            Some(anchor) => {
                w!(self, "<");
                self.print_type_ref(anchor);
                w!(self, ">::");
            }
            None => match path.kind() {
                PathKind::Plain => {}
                &PathKind::SELF => w!(self, "self"),
                PathKind::Super(n) => {
                    for i in 0..*n {
                        if i == 0 {
                            w!(self, "super");
                        } else {
                            w!(self, "::super");
                        }
                    }
                }
                PathKind::Crate => w!(self, "crate"),
                PathKind::Abs => {}
                PathKind::DollarCrate(krate) => w!(
                    self,
                    "{}",
                    krate
                        .extra_data(self.db)
                        .display_name
                        .as_ref()
                        .map(|it| it.crate_name().symbol().as_str())
                        .unwrap_or("$crate")
                ),
            },
        }

        for (i, segment) in path.segments().iter().enumerate() {
            if i != 0 || !matches!(path.kind(), PathKind::Plain) {
                w!(self, "::");
            }

            w!(self, "{}", segment.name.display(self.db, self.edition));
            if let Some(generics) = segment.args_and_bindings {
                w!(self, "::<");
                self.print_generic_args(generics);

                w!(self, ">");
            }
        }
    }

    pub(crate) fn print_generic_args(&mut self, generics: &GenericArgs) {
        let mut first = true;
        let args = if generics.has_self_type {
            let (self_ty, args) = generics.args.split_first().unwrap();
            w!(self, "Self=");
            self.print_generic_arg(self_ty);
            first = false;
            args
        } else {
            &generics.args
        };
        for arg in args {
            if !first {
                w!(self, ", ");
            }
            first = false;
            self.print_generic_arg(arg);
        }
        for binding in generics.bindings.iter() {
            if !first {
                w!(self, ", ");
            }
            first = false;
            w!(self, "{}", binding.name.display(self.db, self.edition));
            if !binding.bounds.is_empty() {
                w!(self, ": ");
                self.print_type_bounds(&binding.bounds);
            }
            if let Some(ty) = binding.type_ref {
                w!(self, " = ");
                self.print_type_ref(ty);
            }
        }
    }

    pub(crate) fn print_generic_arg(&mut self, arg: &GenericArg) {
        match arg {
            GenericArg::Type(ty) => self.print_type_ref(*ty),
            GenericArg::Const(ConstRef { expr }) => self.print_expr(*expr),
            GenericArg::Lifetime(lt) => self.print_lifetime_ref(*lt),
        }
    }

    pub(crate) fn print_type_param(&mut self, param: TypeParamId) {
        let generic_params = self.db.generic_params(param.parent());

        match generic_params[param.local_id()].name() {
            Some(name) => w!(self, "{}", name.display(self.db, self.edition)),
            None => w!(self, "Param[{}]", param.local_id().into_raw()),
        }
    }

    pub(crate) fn print_lifetime_param(&mut self, param: LifetimeParamId) {
        let generic_params = self.db.generic_params(param.parent);
        w!(self, "{}", generic_params[param.local_id].name.display(self.db, self.edition))
    }

    pub(crate) fn print_lifetime_ref(&mut self, lt_ref: LifetimeRefId) {
        match &self.store[lt_ref] {
            LifetimeRef::Static => w!(self, "'static"),
            LifetimeRef::Named(lt) => {
                w!(self, "{}", lt.display(self.db, self.edition))
            }
            LifetimeRef::Placeholder => w!(self, "'_"),
            LifetimeRef::Error => w!(self, "'{{error}}"),
            &LifetimeRef::Param(p) => self.print_lifetime_param(p),
        }
    }

    pub(crate) fn print_type_ref(&mut self, type_ref: TypeRefId) {
        // FIXME: deduplicate with `HirDisplay` impl
        match &self.store[type_ref] {
            TypeRef::Never => w!(self, "!"),
            &TypeRef::TypeParam(p) => self.print_type_param(p),
            TypeRef::Placeholder => w!(self, "_"),
            TypeRef::Tuple(fields) => {
                w!(self, "(");
                for (i, field) in fields.iter().enumerate() {
                    if i != 0 {
                        w!(self, ", ");
                    }
                    self.print_type_ref(*field);
                }
                w!(self, ")");
            }
            TypeRef::Path(path) => self.print_path(path),
            TypeRef::RawPtr(pointee, mtbl) => {
                let mtbl = match mtbl {
                    Mutability::Shared => "*const",
                    Mutability::Mut => "*mut",
                };
                w!(self, "{mtbl} ");
                self.print_type_ref(*pointee);
            }
            TypeRef::Reference(ref_) => {
                let mtbl = match ref_.mutability {
                    Mutability::Shared => "",
                    Mutability::Mut => "mut ",
                };
                w!(self, "&");
                if let Some(lt) = &ref_.lifetime {
                    self.print_lifetime_ref(*lt);
                    w!(self, " ");
                }
                w!(self, "{mtbl}");
                self.print_type_ref(ref_.ty);
            }
            TypeRef::Array(array) => {
                w!(self, "[");
                self.print_type_ref(array.ty);
                w!(self, "; ");
                self.print_generic_arg(&GenericArg::Const(array.len));
                w!(self, "]");
            }
            TypeRef::Slice(elem) => {
                w!(self, "[");
                self.print_type_ref(*elem);
                w!(self, "]");
            }
            TypeRef::Fn(fn_) => {
                let ((_, return_type), args) =
                    fn_.params.split_last().expect("TypeRef::Fn is missing return type");
                if fn_.is_unsafe {
                    w!(self, "unsafe ");
                }
                if let Some(abi) = &fn_.abi {
                    w!(self, "extern ");
                    w!(self, "{}", abi.as_str());
                    w!(self, " ");
                }
                w!(self, "fn(");
                for (i, (_, typeref)) in args.iter().enumerate() {
                    if i != 0 {
                        w!(self, ", ");
                    }
                    self.print_type_ref(*typeref);
                }
                if fn_.is_varargs {
                    if !args.is_empty() {
                        w!(self, ", ");
                    }
                    w!(self, "...");
                }
                w!(self, ") -> ");
                self.print_type_ref(*return_type);
            }
            TypeRef::Error => w!(self, "{{error}}"),
            TypeRef::ImplTrait(bounds) => {
                w!(self, "impl ");
                self.print_type_bounds(bounds);
            }
            TypeRef::DynTrait(bounds) => {
                w!(self, "dyn ");
                self.print_type_bounds(bounds);
            }
        }
    }

    pub(crate) fn print_type_bounds(&mut self, bounds: &[TypeBound]) {
        for (i, bound) in bounds.iter().enumerate() {
            if i != 0 {
                w!(self, " + ");
            }

            match bound {
                TypeBound::Path(path, modifier) => {
                    match modifier {
                        TraitBoundModifier::None => (),
                        TraitBoundModifier::Maybe => w!(self, "?"),
                    }
                    self.print_path(&self.store[*path]);
                }
                TypeBound::ForLifetime(lifetimes, path) => {
                    w!(
                        self,
                        "for<{}> ",
                        lifetimes
                            .iter()
                            .map(|it| it.display(self.db, self.edition))
                            .format(", ")
                            .to_string()
                    );
                    self.print_path(&self.store[*path]);
                }
                TypeBound::Lifetime(lt) => self.print_lifetime_ref(*lt),
                TypeBound::Use(args) => {
                    w!(self, "use<");
                    let mut first = true;
                    for arg in args {
                        if !mem::take(&mut first) {
                            w!(self, ", ");
                        }
                        match arg {
                            UseArgRef::Name(it) => {
                                w!(self, "{}", it.display(self.db, self.edition))
                            }
                            UseArgRef::Lifetime(it) => self.print_lifetime_ref(*it),
                        }
                    }
                    w!(self, ">")
                }
                TypeBound::Error => w!(self, "{{unknown}}"),
            }
        }
    }
}
