// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Lints built in to rustc.

use metadata::csearch;
use middle::def;
use middle::def::*;
use middle::pat_util;
use middle::trans::adt; // for `adt::is_ffi_safe`
use middle::ty;
use middle::typeck::astconv::{ast_ty_to_ty, AstConv};
use middle::typeck::infer;
use middle::typeck;
use util::ppaux::{ty_to_str};
use lint::Context;
use lint;

use std::cmp;
use std::collections::HashMap;
use std::i16;
use std::i32;
use std::i64;
use std::i8;
use std::u16;
use std::u32;
use std::u64;
use std::u8;
use syntax::abi;
use syntax::ast_map;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::{ast, ast_util, visit};

pub fn check_while_true_expr(cx: &Context, e: &ast::Expr) {
    match e.node {
        ast::ExprWhile(cond, _) => {
            match cond.node {
                ast::ExprLit(lit) => {
                    match lit.node {
                        ast::LitBool(true) => {
                            cx.span_lint(lint::WhileTrue,
                                         e.span,
                                         "denote infinite loops with loop \
                                          { ... }");
                        }
                        _ => {}
                    }
                }
                _ => ()
            }
        }
        _ => ()
    }
}

pub fn check_unused_casts(cx: &Context, e: &ast::Expr) {
    return match e.node {
        ast::ExprCast(expr, ty) => {
            let t_t = ast_ty_to_ty(cx, &infer::new_infer_ctxt(cx.tcx), ty);
            if  ty::get(ty::expr_ty(cx.tcx, expr)).sty == ty::get(t_t).sty {
                cx.span_lint(lint::UnnecessaryTypecast, ty.span,
                             "unnecessary type cast");
            }
        }
        _ => ()
    };
}

pub fn check_type_limits(cx: &Context, e: &ast::Expr) {
    return match e.node {
        ast::ExprUnary(ast::UnNeg, ex) => {
            match ex.node  {
                ast::ExprLit(lit) => {
                    match lit.node {
                        ast::LitUint(..) => {
                            cx.span_lint(lint::UnsignedNegate, e.span,
                                         "negation of unsigned int literal may be unintentional");
                        },
                        _ => ()
                    }
                },
                _ => {
                    let t = ty::expr_ty(cx.tcx, ex);
                    match ty::get(t).sty {
                        ty::ty_uint(_) => {
                            cx.span_lint(lint::UnsignedNegate, e.span,
                                         "negation of unsigned int variable may be unintentional");
                        },
                        _ => ()
                    }
                }
            }
        },
        ast::ExprBinary(binop, l, r) => {
            if is_comparison(binop) && !check_limits(cx.tcx, binop, l, r) {
                cx.span_lint(lint::TypeLimits, e.span,
                             "comparison is useless due to type limits");
            }
        },
        ast::ExprLit(lit) => {
            match ty::get(ty::expr_ty(cx.tcx, e)).sty {
                ty::ty_int(t) => {
                    let int_type = if t == ast::TyI {
                        cx.tcx.sess.targ_cfg.int_type
                    } else { t };
                    let (min, max) = int_ty_range(int_type);
                    let mut lit_val: i64 = match lit.node {
                        ast::LitInt(v, _) => v,
                        ast::LitUint(v, _) => v as i64,
                        ast::LitIntUnsuffixed(v) => v,
                        _ => fail!()
                    };
                    if cx.negated_expr_id == e.id {
                        lit_val *= -1;
                    }
                    if  lit_val < min || lit_val > max {
                        cx.span_lint(lint::TypeOverflow, e.span,
                                     "literal out of range for its type");
                    }
                },
                ty::ty_uint(t) => {
                    let uint_type = if t == ast::TyU {
                        cx.tcx.sess.targ_cfg.uint_type
                    } else { t };
                    let (min, max) = uint_ty_range(uint_type);
                    let lit_val: u64 = match lit.node {
                        ast::LitInt(v, _) => v as u64,
                        ast::LitUint(v, _) => v,
                        ast::LitIntUnsuffixed(v) => v as u64,
                        _ => fail!()
                    };
                    if  lit_val < min || lit_val > max {
                        cx.span_lint(lint::TypeOverflow, e.span,
                                     "literal out of range for its type");
                    }
                },

                _ => ()
            };
        },
        _ => ()
    };

    fn is_valid<T:cmp::PartialOrd>(binop: ast::BinOp, v: T,
                            min: T, max: T) -> bool {
        match binop {
            ast::BiLt => v >  min && v <= max,
            ast::BiLe => v >= min && v <  max,
            ast::BiGt => v >= min && v <  max,
            ast::BiGe => v >  min && v <= max,
            ast::BiEq | ast::BiNe => v >= min && v <= max,
            _ => fail!()
        }
    }

    fn rev_binop(binop: ast::BinOp) -> ast::BinOp {
        match binop {
            ast::BiLt => ast::BiGt,
            ast::BiLe => ast::BiGe,
            ast::BiGt => ast::BiLt,
            ast::BiGe => ast::BiLe,
            _ => binop
        }
    }

    // for int & uint, be conservative with the warnings, so that the
    // warnings are consistent between 32- and 64-bit platforms
    fn int_ty_range(int_ty: ast::IntTy) -> (i64, i64) {
        match int_ty {
            ast::TyI =>    (i64::MIN,        i64::MAX),
            ast::TyI8 =>   (i8::MIN  as i64, i8::MAX  as i64),
            ast::TyI16 =>  (i16::MIN as i64, i16::MAX as i64),
            ast::TyI32 =>  (i32::MIN as i64, i32::MAX as i64),
            ast::TyI64 =>  (i64::MIN,        i64::MAX)
        }
    }

    fn uint_ty_range(uint_ty: ast::UintTy) -> (u64, u64) {
        match uint_ty {
            ast::TyU =>   (u64::MIN,         u64::MAX),
            ast::TyU8 =>  (u8::MIN   as u64, u8::MAX   as u64),
            ast::TyU16 => (u16::MIN  as u64, u16::MAX  as u64),
            ast::TyU32 => (u32::MIN  as u64, u32::MAX  as u64),
            ast::TyU64 => (u64::MIN,         u64::MAX)
        }
    }

    fn check_limits(tcx: &ty::ctxt, binop: ast::BinOp,
                    l: &ast::Expr, r: &ast::Expr) -> bool {
        let (lit, expr, swap) = match (&l.node, &r.node) {
            (&ast::ExprLit(_), _) => (l, r, true),
            (_, &ast::ExprLit(_)) => (r, l, false),
            _ => return true
        };
        // Normalize the binop so that the literal is always on the RHS in
        // the comparison
        let norm_binop = if swap { rev_binop(binop) } else { binop };
        match ty::get(ty::expr_ty(tcx, expr)).sty {
            ty::ty_int(int_ty) => {
                let (min, max) = int_ty_range(int_ty);
                let lit_val: i64 = match lit.node {
                    ast::ExprLit(li) => match li.node {
                        ast::LitInt(v, _) => v,
                        ast::LitUint(v, _) => v as i64,
                        ast::LitIntUnsuffixed(v) => v,
                        _ => return true
                    },
                    _ => fail!()
                };
                is_valid(norm_binop, lit_val, min, max)
            }
            ty::ty_uint(uint_ty) => {
                let (min, max): (u64, u64) = uint_ty_range(uint_ty);
                let lit_val: u64 = match lit.node {
                    ast::ExprLit(li) => match li.node {
                        ast::LitInt(v, _) => v as u64,
                        ast::LitUint(v, _) => v,
                        ast::LitIntUnsuffixed(v) => v as u64,
                        _ => return true
                    },
                    _ => fail!()
                };
                is_valid(norm_binop, lit_val, min, max)
            }
            _ => true
        }
    }

    fn is_comparison(binop: ast::BinOp) -> bool {
        match binop {
            ast::BiEq | ast::BiLt | ast::BiLe |
            ast::BiNe | ast::BiGe | ast::BiGt => true,
            _ => false
        }
    }
}

pub fn check_item_ctypes(cx: &Context, it: &ast::Item) {
    fn check_ty(cx: &Context, ty: &ast::Ty) {
        match ty.node {
            ast::TyPath(_, _, id) => {
                match cx.tcx.def_map.borrow().get_copy(&id) {
                    def::DefPrimTy(ast::TyInt(ast::TyI)) => {
                        cx.span_lint(lint::CTypes, ty.span,
                                "found rust type `int` in foreign module, while \
                                libc::c_int or libc::c_long should be used");
                    }
                    def::DefPrimTy(ast::TyUint(ast::TyU)) => {
                        cx.span_lint(lint::CTypes, ty.span,
                                "found rust type `uint` in foreign module, while \
                                libc::c_uint or libc::c_ulong should be used");
                    }
                    def::DefTy(def_id) => {
                        if !adt::is_ffi_safe(cx.tcx, def_id) {
                            cx.span_lint(lint::CTypes, ty.span,
                                         "found enum type without foreign-function-safe \
                                          representation annotation in foreign module");
                            // hmm... this message could be more helpful
                        }
                    }
                    _ => ()
                }
            }
            ast::TyPtr(ref mt) => { check_ty(cx, mt.ty) }
            _ => {}
        }
    }

    fn check_foreign_fn(cx: &Context, decl: &ast::FnDecl) {
        for input in decl.inputs.iter() {
            check_ty(cx, input.ty);
        }
        check_ty(cx, decl.output)
    }

    match it.node {
      ast::ItemForeignMod(ref nmod) if nmod.abi != abi::RustIntrinsic => {
        for ni in nmod.items.iter() {
            match ni.node {
                ast::ForeignItemFn(decl, _) => check_foreign_fn(cx, decl),
                ast::ForeignItemStatic(t, _) => check_ty(cx, t)
            }
        }
      }
      _ => {/* nothing to do */ }
    }
}

pub fn check_heap_type(cx: &Context, span: Span, ty: ty::t) {
    let xs = [lint::ManagedHeapMemory, lint::OwnedHeapMemory, lint::HeapMemory];
    for &lint in xs.iter() {
        if cx.get_level(lint) == lint::Allow { continue }

        let mut n_box = 0;
        let mut n_uniq = 0;
        ty::fold_ty(cx.tcx, ty, |t| {
            match ty::get(t).sty {
                ty::ty_box(_) => {
                    n_box += 1;
                }
                ty::ty_uniq(_) |
                ty::ty_trait(box ty::TyTrait {
                    store: ty::UniqTraitStore, ..
                }) |
                ty::ty_closure(box ty::ClosureTy {
                    store: ty::UniqTraitStore,
                    ..
                }) => {
                    n_uniq += 1;
                }

                _ => ()
            };
            t
        });

        if n_uniq > 0 && lint != lint::ManagedHeapMemory {
            let s = ty_to_str(cx.tcx, ty);
            let m = format!("type uses owned (Box type) pointers: {}", s);
            cx.span_lint(lint, span, m.as_slice());
        }

        if n_box > 0 && lint != lint::OwnedHeapMemory {
            let s = ty_to_str(cx.tcx, ty);
            let m = format!("type uses managed (@ type) pointers: {}", s);
            cx.span_lint(lint, span, m.as_slice());
        }
    }
}

pub fn check_heap_item(cx: &Context, it: &ast::Item) {
    match it.node {
        ast::ItemFn(..) |
        ast::ItemTy(..) |
        ast::ItemEnum(..) |
        ast::ItemStruct(..) => check_heap_type(cx, it.span,
                                               ty::node_id_to_type(cx.tcx,
                                                                   it.id)),
        _ => ()
    }

    // If it's a struct, we also have to check the fields' types
    match it.node {
        ast::ItemStruct(struct_def, _) => {
            for struct_field in struct_def.fields.iter() {
                check_heap_type(cx, struct_field.span,
                                ty::node_id_to_type(cx.tcx,
                                                    struct_field.node.id));
            }
        }
        _ => ()
    }
}

struct RawPtrDerivingVisitor<'a> {
    cx: &'a Context<'a>
}

impl<'a> Visitor<()> for RawPtrDerivingVisitor<'a> {
    fn visit_ty(&mut self, ty: &ast::Ty, _: ()) {
        static MSG: &'static str = "use of `#[deriving]` with a raw pointer";
        match ty.node {
            ast::TyPtr(..) => self.cx.span_lint(lint::RawPointerDeriving, ty.span, MSG),
            _ => {}
        }
        visit::walk_ty(self, ty, ());
    }
    // explicit override to a no-op to reduce code bloat
    fn visit_expr(&mut self, _: &ast::Expr, _: ()) {}
    fn visit_block(&mut self, _: &ast::Block, _: ()) {}
}

pub fn check_raw_ptr_deriving(cx: &mut Context, item: &ast::Item) {
    if !attr::contains_name(item.attrs.as_slice(), "automatically_derived") {
        return
    }
    let did = match item.node {
        ast::ItemImpl(..) => {
            match ty::get(ty::node_id_to_type(cx.tcx, item.id)).sty {
                ty::ty_enum(did, _) => did,
                ty::ty_struct(did, _) => did,
                _ => return,
            }
        }
        _ => return,
    };
    if !ast_util::is_local(did) { return }
    let item = match cx.tcx.map.find(did.node) {
        Some(ast_map::NodeItem(item)) => item,
        _ => return,
    };
    if !cx.checked_raw_pointers.insert(item.id) { return }
    match item.node {
        ast::ItemStruct(..) | ast::ItemEnum(..) => {
            let mut visitor = RawPtrDerivingVisitor { cx: cx };
            visit::walk_item(&mut visitor, item, ());
        }
        _ => {}
    }
}

pub fn check_unused_attribute(cx: &Context, attr: &ast::Attribute) {
    static ATTRIBUTE_WHITELIST: &'static [&'static str] = &'static [
        // FIXME: #14408 whitelist docs since rustdoc looks at them
        "doc",

        // FIXME: #14406 these are processed in trans, which happens after the
        // lint pass
        "address_insignificant",
        "cold",
        "inline",
        "link",
        "link_name",
        "link_section",
        "no_builtins",
        "no_mangle",
        "no_split_stack",
        "packed",
        "static_assert",
        "thread_local",

        // not used anywhere (!?) but apparently we want to keep them around
        "comment",
        "desc",
        "license",

        // FIXME: #14407 these are only looked at on-demand so we can't
        // guarantee they'll have already been checked
        "deprecated",
        "experimental",
        "frozen",
        "locked",
        "must_use",
        "stable",
        "unstable",
    ];

    static CRATE_ATTRS: &'static [&'static str] = &'static [
        "crate_type",
        "feature",
        "no_start",
        "no_main",
        "no_std",
        "crate_id",
        "desc",
        "comment",
        "license",
        "copyright",
        "no_builtins",
    ];

    for &name in ATTRIBUTE_WHITELIST.iter() {
        if attr.check_name(name) {
            break;
        }
    }

    if !attr::is_used(attr) {
        cx.span_lint(lint::UnusedAttribute, attr.span, "unused attribute");
        if CRATE_ATTRS.contains(&attr.name().get()) {
            let msg = match attr.node.style {
               ast::AttrOuter => "crate-level attribute should be an inner \
                                  attribute: add an exclamation mark: #![foo]",
                ast::AttrInner => "crate-level attribute should be in the \
                                   root module",
            };
            cx.span_lint(lint::UnusedAttribute, attr.span, msg);
        }
    }
}

pub fn check_heap_expr(cx: &Context, e: &ast::Expr) {
    let ty = ty::expr_ty(cx.tcx, e);
    check_heap_type(cx, e.span, ty);
}

pub fn check_path_statement(cx: &Context, s: &ast::Stmt) {
    match s.node {
        ast::StmtSemi(expr, _) => {
            match expr.node {
                ast::ExprPath(_) => {
                    cx.span_lint(lint::PathStatement,
                                 s.span,
                                 "path statement with no effect");
                }
                _ => {}
            }
        }
        _ => ()
    }
}

pub fn check_unused_result(cx: &Context, s: &ast::Stmt) {
    let expr = match s.node {
        ast::StmtSemi(expr, _) => expr,
        _ => return
    };
    let t = ty::expr_ty(cx.tcx, expr);
    match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot | ty::ty_bool => return,
        _ => {}
    }
    match expr.node {
        ast::ExprRet(..) => return,
        _ => {}
    }

    let t = ty::expr_ty(cx.tcx, expr);
    let mut warned = false;
    match ty::get(t).sty {
        ty::ty_struct(did, _) |
        ty::ty_enum(did, _) => {
            if ast_util::is_local(did) {
                match cx.tcx.map.get(did.node) {
                    ast_map::NodeItem(it) => {
                        if attr::contains_name(it.attrs.as_slice(),
                                               "must_use") {
                            cx.span_lint(lint::UnusedMustUse, s.span,
                                         "unused result which must be used");
                            warned = true;
                        }
                    }
                    _ => {}
                }
            } else {
                csearch::get_item_attrs(&cx.tcx.sess.cstore, did, |attrs| {
                    if attr::contains_name(attrs.as_slice(), "must_use") {
                        cx.span_lint(lint::UnusedMustUse, s.span,
                                     "unused result which must be used");
                        warned = true;
                    }
                });
            }
        }
        _ => {}
    }
    if !warned {
        cx.span_lint(lint::UnusedResult, s.span, "unused result");
    }
}

pub fn check_deprecated_owned_vector(cx: &Context, e: &ast::Expr) {
    let t = ty::expr_ty(cx.tcx, e);
    match ty::get(t).sty {
        ty::ty_uniq(t) => match ty::get(t).sty {
            ty::ty_vec(_, None) => {
                cx.span_lint(lint::DeprecatedOwnedVector, e.span,
                             "use of deprecated `~[]` vector; replaced by `std::vec::Vec`")
            }
            _ => {}
        },
        _ => {}
    }
}

pub fn check_item_non_camel_case_types(cx: &Context, it: &ast::Item) {
    fn is_camel_case(ident: ast::Ident) -> bool {
        let ident = token::get_ident(ident);
        assert!(!ident.get().is_empty());
        let ident = ident.get().trim_chars('_');

        // start with a non-lowercase letter rather than non-uppercase
        // ones (some scripts don't have a concept of upper/lowercase)
        !ident.char_at(0).is_lowercase() && !ident.contains_char('_')
    }

    fn to_camel_case(s: &str) -> String {
        s.split('_').flat_map(|word| word.chars().enumerate().map(|(i, c)|
            if i == 0 { c.to_uppercase() }
            else { c }
        )).collect()
    }

    fn check_case(cx: &Context, sort: &str, ident: ast::Ident, span: Span) {
        let s = token::get_ident(ident);

        if !is_camel_case(ident) {
            cx.span_lint(lint::
                NonCamelCaseTypes, span,
                format!("{} `{}` should have a camel case name such as `{}`",
                    sort, s, to_camel_case(s.get())).as_slice());
        }
    }

    match it.node {
        ast::ItemTy(..) | ast::ItemStruct(..) => {
            check_case(cx, "type", it.ident, it.span)
        }
        ast::ItemTrait(..) => {
            check_case(cx, "trait", it.ident, it.span)
        }
        ast::ItemEnum(ref enum_definition, _) => {
            check_case(cx, "type", it.ident, it.span);
            for variant in enum_definition.variants.iter() {
                check_case(cx, "variant", variant.node.name, variant.span);
            }
        }
        _ => ()
    }
}

pub fn check_snake_case(cx: &Context, sort: &str, ident: ast::Ident, span: Span) {
    fn is_snake_case(ident: ast::Ident) -> bool {
        let ident = token::get_ident(ident);
        assert!(!ident.get().is_empty());
        let ident = ident.get().trim_chars('_');

        let mut allow_underscore = true;
        ident.chars().all(|c| {
            allow_underscore = match c {
                c if c.is_lowercase() || c.is_digit() => true,
                '_' if allow_underscore => false,
                _ => return false,
            };
            true
        })
    }

    fn to_snake_case(str: &str) -> String {
        let mut words = vec![];
        for s in str.split('_') {
            let mut buf = String::new();
            if s.is_empty() { continue; }
            for ch in s.chars() {
                if !buf.is_empty() && ch.is_uppercase() {
                    words.push(buf);
                    buf = String::new();
                }
                buf.push_char(ch.to_lowercase());
            }
            words.push(buf);
        }
        words.connect("_")
    }

    let s = token::get_ident(ident);

    if !is_snake_case(ident) {
        cx.span_lint(lint::NonSnakeCaseFunctions, span,
            format!("{} `{}` should have a snake case name such as `{}`",
                sort, s, to_snake_case(s.get())).as_slice());
    }
}

pub fn check_item_non_uppercase_statics(cx: &Context, it: &ast::Item) {
    match it.node {
        // only check static constants
        ast::ItemStatic(_, ast::MutImmutable, _) => {
            let s = token::get_ident(it.ident);
            // check for lowercase letters rather than non-uppercase
            // ones (some scripts don't have a concept of
            // upper/lowercase)
            if s.get().chars().any(|c| c.is_lowercase()) {
                cx.span_lint(lint::NonUppercaseStatics, it.span,
                    format!("static constant `{}` should have an uppercase name \
                        such as `{}`", s.get(),
                        s.get().chars().map(|c| c.to_uppercase())
                            .collect::<String>().as_slice()).as_slice());
            }
        }
        _ => {}
    }
}

pub fn check_pat_non_uppercase_statics(cx: &Context, p: &ast::Pat) {
    // Lint for constants that look like binding identifiers (#7526)
    match (&p.node, cx.tcx.def_map.borrow().find(&p.id)) {
        (&ast::PatIdent(_, ref path, _), Some(&def::DefStatic(_, false))) => {
            // last identifier alone is right choice for this lint.
            let ident = path.segments.last().unwrap().identifier;
            let s = token::get_ident(ident);
            if s.get().chars().any(|c| c.is_lowercase()) {
                cx.span_lint(lint::NonUppercasePatternStatics, path.span,
                    format!("static constant in pattern `{}` should have an uppercase \
                        name such as `{}`", s.get(),
                        s.get().chars().map(|c| c.to_uppercase())
                            .collect::<String>().as_slice()).as_slice());
            }
        }
        _ => {}
    }
}

pub fn check_pat_uppercase_variable(cx: &Context, p: &ast::Pat) {
    match &p.node {
        &ast::PatIdent(_, ref path, _) => {
            match cx.tcx.def_map.borrow().find(&p.id) {
                Some(&def::DefLocal(_, _)) | Some(&def::DefBinding(_, _)) |
                        Some(&def::DefArg(_, _)) => {
                    // last identifier alone is right choice for this lint.
                    let ident = path.segments.last().unwrap().identifier;
                    let s = token::get_ident(ident);
                    if s.get().len() > 0 && s.get().char_at(0).is_uppercase() {
                        cx.span_lint(lint::
                            UppercaseVariables,
                            path.span,
                            "variable names should start with a lowercase character");
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

pub fn check_struct_uppercase_variable(cx: &Context, s: &ast::StructDef) {
    for sf in s.fields.iter() {
        match sf.node {
            ast::StructField_ { kind: ast::NamedField(ident, _), .. } => {
                let s = token::get_ident(ident);
                if s.get().char_at(0).is_uppercase() {
                    cx.span_lint(lint::
                        UppercaseVariables,
                        sf.span,
                        "structure field names should start with a lowercase character");
                }
            }
            _ => {}
        }
    }
}

pub fn check_unnecessary_parens_core(cx: &Context, value: &ast::Expr, msg: &str) {
    match value.node {
        ast::ExprParen(_) => {
            cx.span_lint(lint::UnnecessaryParens, value.span,
                         format!("unnecessary parentheses around {}",
                                 msg).as_slice())
        }
        _ => {}
    }
}

pub fn check_unnecessary_parens_expr(cx: &Context, e: &ast::Expr) {
    let (value, msg) = match e.node {
        ast::ExprIf(cond, _, _) => (cond, "`if` condition"),
        ast::ExprWhile(cond, _) => (cond, "`while` condition"),
        ast::ExprMatch(head, _) => (head, "`match` head expression"),
        ast::ExprRet(Some(value)) => (value, "`return` value"),
        ast::ExprAssign(_, value) => (value, "assigned value"),
        ast::ExprAssignOp(_, _, value) => (value, "assigned value"),
        _ => return
    };
    check_unnecessary_parens_core(cx, value, msg);
}

pub fn check_unnecessary_parens_stmt(cx: &Context, s: &ast::Stmt) {
    let (value, msg) = match s.node {
        ast::StmtDecl(decl, _) => match decl.node {
            ast::DeclLocal(local) => match local.init {
                Some(value) => (value, "assigned value"),
                None => return
            },
            _ => return
        },
        _ => return
    };
    check_unnecessary_parens_core(cx, value, msg);
}

pub fn check_unused_unsafe(cx: &Context, e: &ast::Expr) {
    match e.node {
        // Don't warn about generated blocks, that'll just pollute the output.
        ast::ExprBlock(ref blk) => {
            if blk.rules == ast::UnsafeBlock(ast::UserProvided) &&
                !cx.tcx.used_unsafe.borrow().contains(&blk.id) {
                cx.span_lint(lint::UnusedUnsafe, blk.span,
                             "unnecessary `unsafe` block");
            }
        }
        _ => ()
    }
}

pub fn check_unsafe_block(cx: &Context, e: &ast::Expr) {
    match e.node {
        // Don't warn about generated blocks, that'll just pollute the output.
        ast::ExprBlock(ref blk) if blk.rules == ast::UnsafeBlock(ast::UserProvided) => {
            cx.span_lint(lint::UnsafeBlock, blk.span, "usage of an `unsafe` block");
        }
        _ => ()
    }
}

pub fn check_unused_mut_pat(cx: &Context, pats: &[@ast::Pat]) {
    // collect all mutable pattern and group their NodeIDs by their Identifier to
    // avoid false warnings in match arms with multiple patterns
    let mut mutables = HashMap::new();
    for &p in pats.iter() {
        pat_util::pat_bindings(&cx.tcx.def_map, p, |mode, id, _, path| {
            match mode {
                ast::BindByValue(ast::MutMutable) => {
                    if path.segments.len() != 1 {
                        cx.tcx.sess.span_bug(p.span,
                                             "mutable binding that doesn't consist \
                                              of exactly one segment");
                    }
                    let ident = path.segments.get(0).identifier;
                    if !token::get_ident(ident).get().starts_with("_") {
                        mutables.insert_or_update_with(ident.name as uint, vec!(id), |_, old| {
                            old.push(id);
                        });
                    }
                }
                _ => {
                }
            }
        });
    }

    let used_mutables = cx.tcx.used_mut_nodes.borrow();
    for (_, v) in mutables.iter() {
        if !v.iter().any(|e| used_mutables.contains(e)) {
            cx.span_lint(lint::UnusedMut, cx.tcx.map.span(*v.get(0)),
                         "variable does not need to be mutable");
        }
    }
}

enum Allocation {
    VectorAllocation,
    BoxAllocation
}

pub fn check_unnecessary_allocation(cx: &Context, e: &ast::Expr) {
    // Warn if string and vector literals with sigils, or boxing expressions,
    // are immediately borrowed.
    let allocation = match e.node {
        ast::ExprVstore(e2, ast::ExprVstoreUniq) => {
            match e2.node {
                ast::ExprLit(lit) if ast_util::lit_is_str(lit) => {
                    VectorAllocation
                }
                ast::ExprVec(..) => VectorAllocation,
                _ => return
            }
        }
        ast::ExprUnary(ast::UnUniq, _) |
        ast::ExprUnary(ast::UnBox, _) => BoxAllocation,

        _ => return
    };

    let report = |msg| {
        cx.span_lint(lint::UnnecessaryAllocation, e.span, msg);
    };

    match cx.tcx.adjustments.borrow().find(&e.id) {
        Some(adjustment) => {
            match *adjustment {
                ty::AutoDerefRef(ty::AutoDerefRef { autoref, .. }) => {
                    match (allocation, autoref) {
                        (VectorAllocation, Some(ty::AutoBorrowVec(..))) => {
                            report("unnecessary allocation, the sigil can be \
                                    removed");
                        }
                        (BoxAllocation,
                         Some(ty::AutoPtr(_, ast::MutImmutable))) => {
                            report("unnecessary allocation, use & instead");
                        }
                        (BoxAllocation,
                         Some(ty::AutoPtr(_, ast::MutMutable))) => {
                            report("unnecessary allocation, use &mut \
                                    instead");
                        }
                        _ => ()
                    }
                }
                _ => {}
            }
        }

        _ => ()
    }
}

pub fn check_missing_doc_attrs(cx: &Context,
                           id: Option<ast::NodeId>,
                           attrs: &[ast::Attribute],
                           sp: Span,
                           desc: &'static str) {
    // If we're building a test harness, then warning about
    // documentation is probably not really relevant right now.
    if cx.tcx.sess.opts.test { return }

    // `#[doc(hidden)]` disables missing_doc check.
    if cx.is_doc_hidden { return }

    // Only check publicly-visible items, using the result from the privacy pass. It's an option so
    // the crate root can also use this function (it doesn't have a NodeId).
    match id {
        Some(ref id) if !cx.exported_items.contains(id) => return,
        _ => ()
    }

    let has_doc = attrs.iter().any(|a| {
        match a.node.value.node {
            ast::MetaNameValue(ref name, _) if name.equiv(&("doc")) => true,
            _ => false
        }
    });
    if !has_doc {
        cx.span_lint(lint::MissingDoc,
                     sp,
                     format!("missing documentation for {}",
                             desc).as_slice());
    }
}

pub fn check_missing_doc_item(cx: &Context, it: &ast::Item) {
    let desc = match it.node {
        ast::ItemFn(..) => "a function",
        ast::ItemMod(..) => "a module",
        ast::ItemEnum(..) => "an enum",
        ast::ItemStruct(..) => "a struct",
        ast::ItemTrait(..) => "a trait",
        _ => return
    };
    check_missing_doc_attrs(cx,
                            Some(it.id),
                            it.attrs.as_slice(),
                            it.span,
                            desc);
}

pub fn check_missing_doc_method(cx: &Context, m: &ast::Method) {
    // If the method is an impl for a trait, don't doc.
    if lint::method_context(cx, m) == lint::TraitImpl { return; }

    // Otherwise, doc according to privacy. This will also check
    // doc for default methods defined on traits.
    check_missing_doc_attrs(cx,
                            Some(m.id),
                            m.attrs.as_slice(),
                            m.span,
                            "a method");
}

pub fn check_missing_doc_ty_method(cx: &Context, tm: &ast::TypeMethod) {
    check_missing_doc_attrs(cx,
                            Some(tm.id),
                            tm.attrs.as_slice(),
                            tm.span,
                            "a type method");
}

pub fn check_missing_doc_struct_field(cx: &Context, sf: &ast::StructField) {
    match sf.node.kind {
        ast::NamedField(_, vis) if vis == ast::Public =>
            check_missing_doc_attrs(cx,
                                    Some(cx.cur_struct_def_id),
                                    sf.node.attrs.as_slice(),
                                    sf.span,
                                    "a struct field"),
        _ => {}
    }
}

pub fn check_missing_doc_variant(cx: &Context, v: &ast::Variant) {
    check_missing_doc_attrs(cx,
                            Some(v.node.id),
                            v.node.attrs.as_slice(),
                            v.span,
                            "a variant");
}

/// Checks for use of items with #[deprecated], #[experimental] and
/// #[unstable] (or none of them) attributes.
pub fn check_stability(cx: &Context, e: &ast::Expr) {
    let id = match e.node {
        ast::ExprPath(..) | ast::ExprStruct(..) => {
            match cx.tcx.def_map.borrow().find(&e.id) {
                Some(&def) => def.def_id(),
                None => return
            }
        }
        ast::ExprMethodCall(..) => {
            let method_call = typeck::MethodCall::expr(e.id);
            match cx.tcx.method_map.borrow().find(&method_call) {
                Some(method) => {
                    match method.origin {
                        typeck::MethodStatic(def_id) => {
                            // If this implements a trait method, get def_id
                            // of the method inside trait definition.
                            // Otherwise, use the current def_id (which refers
                            // to the method inside impl).
                            ty::trait_method_of_method(
                                cx.tcx, def_id).unwrap_or(def_id)
                        }
                        typeck::MethodParam(typeck::MethodParam {
                            trait_id: trait_id,
                            method_num: index,
                            ..
                        })
                        | typeck::MethodObject(typeck::MethodObject {
                            trait_id: trait_id,
                            method_num: index,
                            ..
                        }) => ty::trait_method(cx.tcx, trait_id, index).def_id
                    }
                }
                None => return
            }
        }
        _ => return
    };

    let stability = if ast_util::is_local(id) {
        // this crate
        let s = cx.tcx.map.with_attrs(id.node, |attrs| {
            attrs.map(|a| attr::find_stability(a.as_slice()))
        });
        match s {
            Some(s) => s,

            // no possibility of having attributes
            // (e.g. it's a local variable), so just
            // ignore it.
            None => return
        }
    } else {
        // cross-crate

        let mut s = None;
        // run through all the attributes and take the first
        // stability one.
        csearch::get_item_attrs(&cx.tcx.sess.cstore, id, |attrs| {
            if s.is_none() {
                s = attr::find_stability(attrs.as_slice())
            }
        });
        s
    };

    let (lint, label) = match stability {
        // no stability attributes == Unstable
        None => (lint::Unstable, "unmarked"),
        Some(attr::Stability { level: attr::Unstable, .. }) =>
                (lint::Unstable, "unstable"),
        Some(attr::Stability { level: attr::Experimental, .. }) =>
                (lint::Experimental, "experimental"),
        Some(attr::Stability { level: attr::Deprecated, .. }) =>
                (lint::Deprecated, "deprecated"),
        _ => return
    };

    let msg = match stability {
        Some(attr::Stability { text: Some(ref s), .. }) => {
            format!("use of {} item: {}", label, *s)
        }
        _ => format!("use of {} item", label)
    };

    cx.span_lint(lint, e.span, msg.as_slice());
}

pub fn check_enum_variant_sizes(cx: &mut Context, it: &ast::Item) {
    match it.node {
        ast::ItemEnum(..) => {
            match cx.cur.find(&(lint::VariantSizeDifference as uint)) {
                Some(&(lvl, src)) if lvl != lint::Allow => {
                    cx.node_levels.insert((it.id, lint::VariantSizeDifference), (lvl, src));
                },
                _ => { }
            }
        },
        _ => { }
    }
}
