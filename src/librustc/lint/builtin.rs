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
//!
//! This is a sibling of `lint::context` in order to ensure that
//! lints implemented here use the same public API as lint plugins.
//!
//! To add a new lint to rustc, declare it here using `declare_lint!()`.
//! Then add code to emit the new lint in the appropriate circumstances.
//! You can do that in an existing `LintPass` if it makes sense, or in
//! a new `LintPass`, or using `Session::add_lint` elsewhere in the
//! compiler. Only do the latter if the check can't be written cleanly
//! as a `LintPass`.
//!
//! If you define a new `LintPass`, you will also need to add it to the
//! `add_builtin_lints!()` invocation in `context.rs`. That macro
//! requires a `Default` impl for your `LintPass` type.

use metadata::csearch;
use middle::def::*;
use middle::trans::adt; // for `adt::is_ffi_safe`
use middle::typeck::astconv::ast_ty_to_ty;
use middle::typeck::infer;
use middle::privacy::ExportedItems;
use middle::{typeck, ty, def, pat_util};
use util::ppaux::{ty_to_str};
use util::nodemap::NodeSet;
use lint::{Context, LintPass, LintArray};

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
use std::default::Default;
use syntax::abi;
use syntax::ast_map;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::{ast, ast_util, visit};

declare_lint!(while_true, Warn,
    "suggest using `loop { }` instead of `while true { }`")

#[deriving(Default)]
pub struct WhileTrue;

impl LintPass for WhileTrue {
    fn get_lints(&self) -> LintArray {
        lint_array!(while_true)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            ast::ExprWhile(cond, _) => {
                match cond.node {
                    ast::ExprLit(lit) => {
                        match lit.node {
                            ast::LitBool(true) => {
                                cx.span_lint(while_true, e.span,
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
}

declare_lint!(unnecessary_typecast, Allow,
    "detects unnecessary type casts, that can be removed")

#[deriving(Default)]
pub struct UnusedCasts;

impl LintPass for UnusedCasts {
    fn get_lints(&self) -> LintArray {
        lint_array!(unnecessary_typecast)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            ast::ExprCast(expr, ty) => {
                let t_t = ast_ty_to_ty(cx, &infer::new_infer_ctxt(cx.tcx), ty);
                if  ty::get(ty::expr_ty(cx.tcx, expr)).sty == ty::get(t_t).sty {
                    cx.span_lint(unnecessary_typecast, ty.span, "unnecessary type cast");
                }
            }
            _ => ()
        }
    }
}

declare_lint!(unsigned_negate, Warn,
    "using an unary minus operator on unsigned type")

declare_lint!(type_limits, Warn,
    "comparisons made useless by limits of the types involved")

declare_lint!(type_overflow, Warn,
    "literal out of range for its type")

pub struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: ast::NodeId,
}

impl Default for TypeLimits {
    fn default() -> TypeLimits {
        TypeLimits {
            negated_expr_id: -1,
        }
    }
}

impl LintPass for TypeLimits {
    fn get_lints(&self) -> LintArray {
        lint_array!(unsigned_negate, type_limits, type_overflow)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            ast::ExprUnary(ast::UnNeg, expr) => {
                match expr.node  {
                    ast::ExprLit(lit) => {
                        match lit.node {
                            ast::LitUint(..) => {
                                cx.span_lint(unsigned_negate, e.span,
                                             "negation of unsigned int literal may \
                                             be unintentional");
                            },
                            _ => ()
                        }
                    },
                    _ => {
                        let t = ty::expr_ty(cx.tcx, expr);
                        match ty::get(t).sty {
                            ty::ty_uint(_) => {
                                cx.span_lint(unsigned_negate, e.span,
                                             "negation of unsigned int variable may \
                                             be unintentional");
                            },
                            _ => ()
                        }
                    }
                };
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != e.id {
                    self.negated_expr_id = expr.id;
                }
            },
            ast::ExprParen(expr) if self.negated_expr_id == e.id => {
                self.negated_expr_id = expr.id;
            },
            ast::ExprBinary(binop, l, r) => {
                if is_comparison(binop) && !check_limits(cx.tcx, binop, l, r) {
                    cx.span_lint(type_limits, e.span,
                                 "comparison is useless due to type limits");
                }
            },
            ast::ExprLit(lit) => {
                match ty::get(ty::expr_ty(cx.tcx, e)).sty {
                    ty::ty_int(t) => {
                        let int_type = if t == ast::TyI {
                            cx.sess().targ_cfg.int_type
                        } else { t };
                        let (min, max) = int_ty_range(int_type);
                        let mut lit_val: i64 = match lit.node {
                            ast::LitInt(v, _) => v,
                            ast::LitUint(v, _) => v as i64,
                            ast::LitIntUnsuffixed(v) => v,
                            _ => fail!()
                        };
                        if self.negated_expr_id == e.id {
                            lit_val *= -1;
                        }
                        if  lit_val < min || lit_val > max {
                            cx.span_lint(type_overflow, e.span,
                                         "literal out of range for its type");
                        }
                    },
                    ty::ty_uint(t) => {
                        let uint_type = if t == ast::TyU {
                            cx.sess().targ_cfg.uint_type
                        } else { t };
                        let (min, max) = uint_ty_range(uint_type);
                        let lit_val: u64 = match lit.node {
                            ast::LitInt(v, _) => v as u64,
                            ast::LitUint(v, _) => v,
                            ast::LitIntUnsuffixed(v) => v as u64,
                            _ => fail!()
                        };
                        if  lit_val < min || lit_val > max {
                            cx.span_lint(type_overflow, e.span,
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
}

declare_lint!(ctypes, Warn,
    "proper use of libc types in foreign modules")

#[deriving(Default)]
pub struct CTypes;

impl LintPass for CTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(ctypes)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        fn check_ty(cx: &Context, ty: &ast::Ty) {
            match ty.node {
                ast::TyPath(_, _, id) => {
                    match cx.tcx.def_map.borrow().get_copy(&id) {
                        def::DefPrimTy(ast::TyInt(ast::TyI)) => {
                            cx.span_lint(ctypes, ty.span,
                                "found rust type `int` in foreign module, while \
                                 libc::c_int or libc::c_long should be used");
                        }
                        def::DefPrimTy(ast::TyUint(ast::TyU)) => {
                            cx.span_lint(ctypes, ty.span,
                                "found rust type `uint` in foreign module, while \
                                 libc::c_uint or libc::c_ulong should be used");
                        }
                        def::DefTy(def_id) => {
                            if !adt::is_ffi_safe(cx.tcx, def_id) {
                                cx.span_lint(ctypes, ty.span,
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
}

declare_lint!(managed_heap_memory, Allow,
    "use of managed (@ type) heap memory")

declare_lint!(owned_heap_memory, Allow,
    "use of owned (Box type) heap memory")

declare_lint!(heap_memory, Allow,
    "use of any (Box type or @ type) heap memory")

#[deriving(Default)]
pub struct HeapMemory;

impl HeapMemory {
    fn check_heap_type(&self, cx: &Context, span: Span, ty: ty::t) {
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

        if n_uniq > 0 {
            let s = ty_to_str(cx.tcx, ty);
            let m = format!("type uses owned (Box type) pointers: {}", s);
            cx.span_lint(owned_heap_memory, span, m.as_slice());
            cx.span_lint(heap_memory, span, m.as_slice());
        }

        if n_box > 0 {
            let s = ty_to_str(cx.tcx, ty);
            let m = format!("type uses managed (@ type) pointers: {}", s);
            cx.span_lint(managed_heap_memory, span, m.as_slice());
            cx.span_lint(heap_memory, span, m.as_slice());
        }
    }
}

impl LintPass for HeapMemory {
    fn get_lints(&self) -> LintArray {
        lint_array!(managed_heap_memory, owned_heap_memory, heap_memory)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            ast::ItemFn(..) |
            ast::ItemTy(..) |
            ast::ItemEnum(..) |
            ast::ItemStruct(..)
                => self.check_heap_type(cx, it.span,
                    ty::node_id_to_type(cx.tcx, it.id)),
            _ => ()
        }

        // If it's a struct, we also have to check the fields' types
        match it.node {
            ast::ItemStruct(struct_def, _) => {
                for struct_field in struct_def.fields.iter() {
                    self.check_heap_type(cx, struct_field.span,
                    ty::node_id_to_type(cx.tcx, struct_field.node.id));
                }
            }
            _ => ()
        }
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        let ty = ty::expr_ty(cx.tcx, e);
        self.check_heap_type(cx, e.span, ty);
    }
}

declare_lint!(raw_pointer_deriving, Warn,
    "uses of #[deriving] with raw pointers are rarely correct")

struct RawPtrDerivingVisitor<'a> {
    cx: &'a Context<'a>
}

impl<'a> visit::Visitor<()> for RawPtrDerivingVisitor<'a> {
    fn visit_ty(&mut self, ty: &ast::Ty, _: ()) {
        static MSG: &'static str = "use of `#[deriving]` with a raw pointer";
        match ty.node {
            ast::TyPtr(..) => self.cx.span_lint(raw_pointer_deriving, ty.span, MSG),
            _ => {}
        }
        visit::walk_ty(self, ty, ());
    }
    // explicit override to a no-op to reduce code bloat
    fn visit_expr(&mut self, _: &ast::Expr, _: ()) {}
    fn visit_block(&mut self, _: &ast::Block, _: ()) {}
}

pub struct RawPointerDeriving {
    checked_raw_pointers: NodeSet,
}

impl Default for RawPointerDeriving {
    fn default() -> RawPointerDeriving {
        RawPointerDeriving {
            checked_raw_pointers: NodeSet::new(),
        }
    }
}

impl LintPass for RawPointerDeriving {
    fn get_lints(&self) -> LintArray {
        lint_array!(raw_pointer_deriving)
    }

    fn check_item(&mut self, cx: &Context, item: &ast::Item) {
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
        if !self.checked_raw_pointers.insert(item.id) { return }
        match item.node {
            ast::ItemStruct(..) | ast::ItemEnum(..) => {
                let mut visitor = RawPtrDerivingVisitor { cx: cx };
                visit::walk_item(&mut visitor, item, ());
            }
            _ => {}
        }
    }
}

declare_lint!(unused_attribute, Warn,
    "detects attributes that were not used by the compiler")

#[deriving(Default)]
pub struct UnusedAttribute;

impl LintPass for UnusedAttribute {
    fn get_lints(&self) -> LintArray {
        lint_array!(unused_attribute)
    }

    fn check_attribute(&mut self, cx: &Context, attr: &ast::Attribute) {
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
            cx.span_lint(unused_attribute, attr.span, "unused attribute");
            if CRATE_ATTRS.contains(&attr.name().get()) {
                let msg = match attr.node.style {
                   ast::AttrOuter => "crate-level attribute should be an inner \
                                      attribute: add an exclamation mark: #![foo]",
                    ast::AttrInner => "crate-level attribute should be in the \
                                       root module",
                };
                cx.span_lint(unused_attribute, attr.span, msg);
            }
        }
    }
}

declare_lint!(path_statement, Warn,
    "path statements with no effect")

#[deriving(Default)]
pub struct PathStatement;

impl LintPass for PathStatement {
    fn get_lints(&self) -> LintArray {
        lint_array!(path_statement)
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
        match s.node {
            ast::StmtSemi(expr, _) => {
                match expr.node {
                    ast::ExprPath(_) => cx.span_lint(path_statement, s.span,
                                                     "path statement with no effect"),
                    _ => ()
                }
            }
            _ => ()
        }
    }
}

declare_lint!(unused_must_use, Warn,
    "unused result of a type flagged as #[must_use]")

declare_lint!(unused_result, Allow,
    "unused result of an expression in a statement")

#[deriving(Default)]
pub struct UnusedResult;

impl LintPass for UnusedResult {
    fn get_lints(&self) -> LintArray {
        lint_array!(unused_must_use, unused_result)
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
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
                                cx.span_lint(unused_must_use, s.span,
                                             "unused result which must be used");
                                warned = true;
                            }
                        }
                        _ => {}
                    }
                } else {
                    csearch::get_item_attrs(&cx.sess().cstore, did, |attrs| {
                        if attr::contains_name(attrs.as_slice(), "must_use") {
                            cx.span_lint(unused_must_use, s.span,
                                         "unused result which must be used");
                            warned = true;
                        }
                    });
                }
            }
            _ => {}
        }
        if !warned {
            cx.span_lint(unused_result, s.span, "unused result");
        }
    }
}

declare_lint!(deprecated_owned_vector, Allow,
    "use of a `~[T]` vector")

#[deriving(Default)]
pub struct DeprecatedOwnedVector;

impl LintPass for DeprecatedOwnedVector {
    fn get_lints(&self) -> LintArray {
        lint_array!(deprecated_owned_vector)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        let t = ty::expr_ty(cx.tcx, e);
        match ty::get(t).sty {
            ty::ty_uniq(t) => match ty::get(t).sty {
                ty::ty_vec(_, None) => {
                    cx.span_lint(deprecated_owned_vector, e.span,
                        "use of deprecated `~[]` vector; replaced by `std::vec::Vec`")
                }
                _ => {}
            },
            _ => {}
        }
    }
}

declare_lint!(non_camel_case_types, Warn,
    "types, variants and traits should have camel case names")

#[deriving(Default)]
pub struct NonCamelCaseTypes;

impl LintPass for NonCamelCaseTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(non_camel_case_types)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
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
                cx.span_lint(non_camel_case_types, span,
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
}

#[deriving(PartialEq)]
enum MethodContext {
    TraitDefaultImpl,
    TraitImpl,
    PlainImpl
}

fn method_context(cx: &Context, m: &ast::Method) -> MethodContext {
    let did = ast::DefId {
        krate: ast::LOCAL_CRATE,
        node: m.id
    };

    match cx.tcx.methods.borrow().find_copy(&did) {
        None => cx.sess().span_bug(m.span, "missing method descriptor?!"),
        Some(md) => {
            match md.container {
                ty::TraitContainer(..) => TraitDefaultImpl,
                ty::ImplContainer(cid) => {
                    match ty::impl_trait_ref(cx.tcx, cid) {
                        Some(..) => TraitImpl,
                        None => PlainImpl
                    }
                }
            }
        }
    }
}

declare_lint!(non_snake_case_functions, Warn,
    "methods and functions should have snake case names")

#[deriving(Default)]
pub struct NonSnakeCaseFunctions;

impl NonSnakeCaseFunctions {
    fn check_snake_case(&self, cx: &Context, sort: &str, ident: ast::Ident, span: Span) {
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
            cx.span_lint(non_snake_case_functions, span,
                format!("{} `{}` should have a snake case name such as `{}`",
                    sort, s, to_snake_case(s.get())).as_slice());
        }
    }
}

impl LintPass for NonSnakeCaseFunctions {
    fn get_lints(&self) -> LintArray {
        lint_array!(non_snake_case_functions)
    }

    fn check_fn(&mut self, cx: &Context,
                fk: &visit::FnKind, _: &ast::FnDecl,
                _: &ast::Block, span: Span, _: ast::NodeId) {
        match *fk {
            visit::FkMethod(ident, _, m) => match method_context(cx, m) {
                PlainImpl
                    => self.check_snake_case(cx, "method", ident, span),
                TraitDefaultImpl
                    => self.check_snake_case(cx, "trait method", ident, span),
                _ => (),
            },
            visit::FkItemFn(ident, _, _, _)
                => self.check_snake_case(cx, "function", ident, span),
            _ => (),
        }
    }

    fn check_ty_method(&mut self, cx: &Context, t: &ast::TypeMethod) {
        self.check_snake_case(cx, "trait method", t.ident, t.span);
    }
}

declare_lint!(non_uppercase_statics, Allow,
    "static constants should have uppercase identifiers")

#[deriving(Default)]
pub struct NonUppercaseStatics;

impl LintPass for NonUppercaseStatics {
    fn get_lints(&self) -> LintArray {
        lint_array!(non_uppercase_statics)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            // only check static constants
            ast::ItemStatic(_, ast::MutImmutable, _) => {
                let s = token::get_ident(it.ident);
                // check for lowercase letters rather than non-uppercase
                // ones (some scripts don't have a concept of
                // upper/lowercase)
                if s.get().chars().any(|c| c.is_lowercase()) {
                    cx.span_lint(non_uppercase_statics, it.span,
                        format!("static constant `{}` should have an uppercase name \
                            such as `{}`", s.get(),
                            s.get().chars().map(|c| c.to_uppercase())
                                .collect::<String>().as_slice()).as_slice());
                }
            }
            _ => {}
        }
    }
}

declare_lint!(non_uppercase_pattern_statics, Warn,
    "static constants in match patterns should be all caps")

#[deriving(Default)]
pub struct NonUppercasePatternStatics;

impl LintPass for NonUppercasePatternStatics {
    fn get_lints(&self) -> LintArray {
        lint_array!(non_uppercase_pattern_statics)
    }

    fn check_pat(&mut self, cx: &Context, p: &ast::Pat) {
        // Lint for constants that look like binding identifiers (#7526)
        match (&p.node, cx.tcx.def_map.borrow().find(&p.id)) {
            (&ast::PatIdent(_, ref path, _), Some(&def::DefStatic(_, false))) => {
                // last identifier alone is right choice for this lint.
                let ident = path.segments.last().unwrap().identifier;
                let s = token::get_ident(ident);
                if s.get().chars().any(|c| c.is_lowercase()) {
                    cx.span_lint(non_uppercase_pattern_statics, path.span,
                        format!("static constant in pattern `{}` should have an uppercase \
                            name such as `{}`", s.get(),
                            s.get().chars().map(|c| c.to_uppercase())
                                .collect::<String>().as_slice()).as_slice());
                }
            }
            _ => {}
        }
    }
}

declare_lint!(uppercase_variables, Warn,
    "variable and structure field names should start with a lowercase character")

#[deriving(Default)]
pub struct UppercaseVariables;

impl LintPass for UppercaseVariables {
    fn get_lints(&self) -> LintArray {
        lint_array!(uppercase_variables)
    }

    fn check_pat(&mut self, cx: &Context, p: &ast::Pat) {
        match &p.node {
            &ast::PatIdent(_, ref path, _) => {
                match cx.tcx.def_map.borrow().find(&p.id) {
                    Some(&def::DefLocal(_, _)) | Some(&def::DefBinding(_, _)) |
                            Some(&def::DefArg(_, _)) => {
                        // last identifier alone is right choice for this lint.
                        let ident = path.segments.last().unwrap().identifier;
                        let s = token::get_ident(ident);
                        if s.get().len() > 0 && s.get().char_at(0).is_uppercase() {
                            cx.span_lint(uppercase_variables, path.span,
                                "variable names should start with a lowercase character");
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn check_struct_def(&mut self, cx: &Context, s: &ast::StructDef,
            _: ast::Ident, _: &ast::Generics, _: ast::NodeId) {
        for sf in s.fields.iter() {
            match sf.node {
                ast::StructField_ { kind: ast::NamedField(ident, _), .. } => {
                    let s = token::get_ident(ident);
                    if s.get().char_at(0).is_uppercase() {
                        cx.span_lint(uppercase_variables, sf.span,
                            "structure field names should start with a lowercase character");
                    }
                }
                _ => {}
            }
        }
    }
}

declare_lint!(unnecessary_parens, Warn,
    "`if`, `match`, `while` and `return` do not need parentheses")

#[deriving(Default)]
pub struct UnnecessaryParens;

impl UnnecessaryParens {
    fn check_unnecessary_parens_core(&self, cx: &Context, value: &ast::Expr, msg: &str) {
        match value.node {
            ast::ExprParen(_) => {
                cx.span_lint(unnecessary_parens, value.span,
                    format!("unnecessary parentheses around {}", msg).as_slice())
            }
            _ => {}
        }
    }
}

impl LintPass for UnnecessaryParens {
    fn get_lints(&self) -> LintArray {
        lint_array!(unnecessary_parens)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        let (value, msg) = match e.node {
            ast::ExprIf(cond, _, _) => (cond, "`if` condition"),
            ast::ExprWhile(cond, _) => (cond, "`while` condition"),
            ast::ExprMatch(head, _) => (head, "`match` head expression"),
            ast::ExprRet(Some(value)) => (value, "`return` value"),
            ast::ExprAssign(_, value) => (value, "assigned value"),
            ast::ExprAssignOp(_, _, value) => (value, "assigned value"),
            _ => return
        };
        self.check_unnecessary_parens_core(cx, value, msg);
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
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
        self.check_unnecessary_parens_core(cx, value, msg);
    }
}

declare_lint!(unused_unsafe, Warn,
    "unnecessary use of an `unsafe` block")

#[deriving(Default)]
pub struct UnusedUnsafe;

impl LintPass for UnusedUnsafe {
    fn get_lints(&self) -> LintArray {
        lint_array!(unused_unsafe)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            // Don't warn about generated blocks, that'll just pollute the output.
            ast::ExprBlock(ref blk) => {
                if blk.rules == ast::UnsafeBlock(ast::UserProvided) &&
                    !cx.tcx.used_unsafe.borrow().contains(&blk.id) {
                    cx.span_lint(unused_unsafe, blk.span, "unnecessary `unsafe` block");
                }
            }
            _ => ()
        }
    }
}

declare_lint!(unsafe_block, Allow,
    "usage of an `unsafe` block")

#[deriving(Default)]
pub struct UnsafeBlock;

impl LintPass for UnsafeBlock {
    fn get_lints(&self) -> LintArray {
        lint_array!(unsafe_block)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            // Don't warn about generated blocks, that'll just pollute the output.
            ast::ExprBlock(ref blk) if blk.rules == ast::UnsafeBlock(ast::UserProvided) => {
                cx.span_lint(unsafe_block, blk.span, "usage of an `unsafe` block");
            }
            _ => ()
        }
    }
}

declare_lint!(unused_mut, Warn,
    "detect mut variables which don't need to be mutable")

#[deriving(Default)]
pub struct UnusedMut;

impl UnusedMut {
    fn check_unused_mut_pat(&self, cx: &Context, pats: &[@ast::Pat]) {
        // collect all mutable pattern and group their NodeIDs by their Identifier to
        // avoid false warnings in match arms with multiple patterns
        let mut mutables = HashMap::new();
        for &p in pats.iter() {
            pat_util::pat_bindings(&cx.tcx.def_map, p, |mode, id, _, path| {
                match mode {
                    ast::BindByValue(ast::MutMutable) => {
                        if path.segments.len() != 1 {
                            cx.sess().span_bug(p.span,
                                                 "mutable binding that doesn't consist \
                                                  of exactly one segment");
                        }
                        let ident = path.segments.get(0).identifier;
                        if !token::get_ident(ident).get().starts_with("_") {
                            mutables.insert_or_update_with(ident.name as uint,
                                vec!(id), |_, old| { old.push(id); });
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
                cx.span_lint(unused_mut, cx.tcx.map.span(*v.get(0)),
                    "variable does not need to be mutable");
            }
        }
    }
}

impl LintPass for UnusedMut {
    fn get_lints(&self) -> LintArray {
        lint_array!(unused_mut)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            ast::ExprMatch(_, ref arms) => {
                for a in arms.iter() {
                    self.check_unused_mut_pat(cx, a.pats.as_slice())
                }
            }
            _ => {}
        }
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
        match s.node {
            ast::StmtDecl(d, _) => {
                match d.node {
                    ast::DeclLocal(l) => {
                        self.check_unused_mut_pat(cx, &[l.pat]);
                    },
                    _ => {}
                }
            },
            _ => {}
        }
    }

    fn check_fn(&mut self, cx: &Context,
                _: &visit::FnKind, decl: &ast::FnDecl,
                _: &ast::Block, _: Span, _: ast::NodeId) {
        for a in decl.inputs.iter() {
            self.check_unused_mut_pat(cx, &[a.pat]);
        }
    }
}

enum Allocation {
    VectorAllocation,
    BoxAllocation
}

declare_lint!(unnecessary_allocation, Warn,
    "detects unnecessary allocations that can be eliminated")

#[deriving(Default)]
pub struct UnnecessaryAllocation;

impl LintPass for UnnecessaryAllocation {
    fn get_lints(&self) -> LintArray {
        lint_array!(unnecessary_allocation)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
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

        match cx.tcx.adjustments.borrow().find(&e.id) {
            Some(adjustment) => {
                match *adjustment {
                    ty::AutoDerefRef(ty::AutoDerefRef { autoref, .. }) => {
                        match (allocation, autoref) {
                            (VectorAllocation, Some(ty::AutoBorrowVec(..))) => {
                                cx.span_lint(unnecessary_allocation, e.span,
                                    "unnecessary allocation, the sigil can be removed");
                            }
                            (BoxAllocation,
                             Some(ty::AutoPtr(_, ast::MutImmutable))) => {
                                cx.span_lint(unnecessary_allocation, e.span,
                                    "unnecessary allocation, use & instead");
                            }
                            (BoxAllocation,
                             Some(ty::AutoPtr(_, ast::MutMutable))) => {
                                cx.span_lint(unnecessary_allocation, e.span,
                                    "unnecessary allocation, use &mut instead");
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
}

declare_lint!(missing_doc, Allow,
    "detects missing documentation for public members")

pub struct MissingDoc {
    /// Set of nodes exported from this module.
    exported_items: Option<ExportedItems>,

    /// Stack of IDs of struct definitions.
    struct_def_stack: Vec<ast::NodeId>,

    /// Stack of whether #[doc(hidden)] is set
    /// at each level which has lint attributes.
    doc_hidden_stack: Vec<bool>,
}

impl Default for MissingDoc {
    fn default() -> MissingDoc {
        MissingDoc {
            exported_items: None,
            struct_def_stack: vec!(),
            doc_hidden_stack: vec!(false),
        }
    }
}

impl MissingDoc {
    fn doc_hidden(&self) -> bool {
        *self.doc_hidden_stack.last().expect("empty doc_hidden_stack")
    }

    fn check_missing_doc_attrs(&self,
                               cx: &Context,
                               id: Option<ast::NodeId>,
                               attrs: &[ast::Attribute],
                               sp: Span,
                               desc: &'static str) {
        // If we're building a test harness, then warning about
        // documentation is probably not really relevant right now.
        if cx.sess().opts.test { return }

        // `#[doc(hidden)]` disables missing_doc check.
        if self.doc_hidden() { return }

        // Only check publicly-visible items, using the result from the privacy pass.
        // It's an option so the crate root can also use this function (it doesn't
        // have a NodeId).
        let exported = self.exported_items.as_ref().expect("exported_items not set");
        match id {
            Some(ref id) if !exported.contains(id) => return,
            _ => ()
        }

        let has_doc = attrs.iter().any(|a| {
            match a.node.value.node {
                ast::MetaNameValue(ref name, _) if name.equiv(&("doc")) => true,
                _ => false
            }
        });
        if !has_doc {
            cx.span_lint(missing_doc, sp,
                format!("missing documentation for {}", desc).as_slice());
        }
    }
}

impl LintPass for MissingDoc {
    fn get_lints(&self) -> LintArray {
        lint_array!(missing_doc)
    }

    fn enter_lint_attrs(&mut self, _: &Context, attrs: &[ast::Attribute]) {
        let doc_hidden = self.doc_hidden() || attrs.iter().any(|attr| {
            attr.check_name("doc") && match attr.meta_item_list() {
                None => false,
                Some(l) => attr::contains_name(l.as_slice(), "hidden"),
            }
        });
        self.doc_hidden_stack.push(doc_hidden);
    }

    fn exit_lint_attrs(&mut self, _: &Context, _: &[ast::Attribute]) {
        self.doc_hidden_stack.pop().expect("empty doc_hidden_stack");
    }

    fn check_struct_def(&mut self, _: &Context,
        _: &ast::StructDef, _: ast::Ident, _: &ast::Generics, id: ast::NodeId) {
        self.struct_def_stack.push(id);
    }

    fn check_struct_def_post(&mut self, _: &Context,
        _: &ast::StructDef, _: ast::Ident, _: &ast::Generics, id: ast::NodeId) {
        let popped = self.struct_def_stack.pop().expect("empty struct_def_stack");
        assert!(popped == id);
    }

    fn check_crate(&mut self, cx: &Context, exported: &ExportedItems, krate: &ast::Crate) {
        // FIXME: clone to avoid lifetime trickiness
        self.exported_items = Some(exported.clone());

        self.check_missing_doc_attrs(cx, None, krate.attrs.as_slice(),
            krate.span, "crate");
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let desc = match it.node {
            ast::ItemFn(..) => "a function",
            ast::ItemMod(..) => "a module",
            ast::ItemEnum(..) => "an enum",
            ast::ItemStruct(..) => "a struct",
            ast::ItemTrait(..) => "a trait",
            _ => return
        };
        self.check_missing_doc_attrs(cx, Some(it.id), it.attrs.as_slice(),
            it.span, desc);
    }

    fn check_fn(&mut self, cx: &Context,
            fk: &visit::FnKind, _: &ast::FnDecl,
            _: &ast::Block, _: Span, _: ast::NodeId) {
        match *fk {
            visit::FkMethod(_, _, m) => {
                // If the method is an impl for a trait, don't doc.
                if method_context(cx, m) == TraitImpl { return; }

                // Otherwise, doc according to privacy. This will also check
                // doc for default methods defined on traits.
                self.check_missing_doc_attrs(cx, Some(m.id), m.attrs.as_slice(),
                    m.span, "a method");
            }
            _ => {}
        }
    }

    fn check_ty_method(&mut self, cx: &Context, tm: &ast::TypeMethod) {
        self.check_missing_doc_attrs(cx, Some(tm.id), tm.attrs.as_slice(),
            tm.span, "a type method");
    }

    fn check_struct_field(&mut self, cx: &Context, sf: &ast::StructField) {
        match sf.node.kind {
            ast::NamedField(_, vis) if vis == ast::Public => {
                let cur_struct_def = *self.struct_def_stack.last()
                    .expect("empty struct_def_stack");
                self.check_missing_doc_attrs(cx, Some(cur_struct_def),
                    sf.node.attrs.as_slice(), sf.span, "a struct field")
            }
            _ => {}
        }
    }

    fn check_variant(&mut self, cx: &Context, v: &ast::Variant, _: &ast::Generics) {
        self.check_missing_doc_attrs(cx, Some(v.node.id), v.node.attrs.as_slice(),
            v.span, "a variant");
    }
}

declare_lint!(deprecated, Warn,
    "detects use of #[deprecated] items")

declare_lint!(experimental, Warn,
    "detects use of #[experimental] items")

declare_lint!(unstable, Allow,
    "detects use of #[unstable] items (incl. items with no stability attribute)")

/// Checks for use of items with `#[deprecated]`, `#[experimental]` and
/// `#[unstable]` attributes, or no stability attribute.
#[deriving(Default)]
pub struct Stability;

impl LintPass for Stability {
    fn get_lints(&self) -> LintArray {
        lint_array!(deprecated, experimental, unstable)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
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
            csearch::get_item_attrs(&cx.sess().cstore, id, |attrs| {
                if s.is_none() {
                    s = attr::find_stability(attrs.as_slice())
                }
            });
            s
        };

        let (lint, label) = match stability {
            // no stability attributes == Unstable
            None => (unstable, "unmarked"),
            Some(attr::Stability { level: attr::Unstable, .. }) =>
                    (unstable, "unstable"),
            Some(attr::Stability { level: attr::Experimental, .. }) =>
                    (experimental, "experimental"),
            Some(attr::Stability { level: attr::Deprecated, .. }) =>
                    (deprecated, "deprecated"),
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
}

declare_lint!(pub unused_imports, Warn,
    "imports that are never used")

declare_lint!(pub unnecessary_qualification, Allow,
    "detects unnecessarily qualified names")

declare_lint!(pub unrecognized_lint, Warn,
    "unrecognized lint attribute")

declare_lint!(pub unused_variable, Warn,
    "detect variables which are not used in any way")

declare_lint!(pub dead_assignment, Warn,
    "detect assignments that will never be read")

declare_lint!(pub dead_code, Warn,
    "detect piece of code that will never be used")

declare_lint!(pub visible_private_types, Warn,
    "detect use of private types in exported type signatures")

declare_lint!(pub unreachable_code, Warn,
    "detects unreachable code")

declare_lint!(pub warnings, Warn,
    "mass-change the level for lints which produce warnings")

declare_lint!(pub unknown_features, Deny,
    "unknown features found in crate-level #[feature] directives")

declare_lint!(pub unknown_crate_type, Deny,
    "unknown crate type found in #[crate_type] directive")

declare_lint!(pub variant_size_difference, Allow,
    "detects enums with widely varying variant sizes")

/// Does nothing as a lint pass, but registers some `Lint`s
/// which are used by other parts of the compiler.
#[deriving(Default)]
pub struct HardwiredLints;

impl LintPass for HardwiredLints {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            unused_imports, unnecessary_qualification, unrecognized_lint,
            unused_variable, dead_assignment, dead_code, visible_private_types,
            unreachable_code, warnings, unknown_features, unknown_crate_type,
            variant_size_difference)
    }
}
