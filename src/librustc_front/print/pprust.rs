// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::AnnNode::*;

use syntax::abi;
use syntax::ast;
use syntax::owned_slice::OwnedSlice;
use syntax::codemap::{self, CodeMap, BytePos, Spanned};
use syntax::diagnostic;
use syntax::parse::token::{self, BinOpToken};
use syntax::parse::lexer::comments;
use syntax::parse;
use syntax::print::pp::{self, break_offset, word, space, hardbreak};
use syntax::print::pp::{Breaks, eof};
use syntax::print::pp::Breaks::{Consistent, Inconsistent};
use syntax::print::pprust::{self as ast_pp, PrintState};
use syntax::ptr::P;

use hir;
use hir::{RegionTyParamBound, TraitTyParamBound, TraitBoundModifier};

use std::io::{self, Write, Read};

pub enum AnnNode<'a> {
    NodeName(&'a ast::Name),
    NodeBlock(&'a hir::Block),
    NodeItem(&'a hir::Item),
    NodeSubItem(ast::NodeId),
    NodeExpr(&'a hir::Expr),
    NodePat(&'a hir::Pat),
}

pub trait PpAnn {
    fn pre(&self, _state: &mut State, _node: AnnNode) -> io::Result<()> {
        Ok(())
    }
    fn post(&self, _state: &mut State, _node: AnnNode) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Copy, Clone)]
pub struct NoAnn;

impl PpAnn for NoAnn {}


pub struct State<'a> {
    pub s: pp::Printer<'a>,
    cm: Option<&'a CodeMap>,
    comments: Option<Vec<comments::Comment>>,
    literals: Option<Vec<comments::Literal>>,
    cur_cmnt_and_lit: ast_pp::CurrentCommentAndLiteral,
    boxes: Vec<pp::Breaks>,
    ann: &'a (PpAnn + 'a),
}

impl<'a> PrintState<'a> for State<'a> {
    fn writer(&mut self) -> &mut pp::Printer<'a> {
        &mut self.s
    }

    fn boxes(&mut self) -> &mut Vec<pp::Breaks> {
        &mut self.boxes
    }

    fn comments(&mut self) -> &mut Option<Vec<comments::Comment>> {
        &mut self.comments
    }

    fn cur_cmnt_and_lit(&mut self) -> &mut ast_pp::CurrentCommentAndLiteral {
        &mut self.cur_cmnt_and_lit
    }

    fn literals(&self) -> &Option<Vec<comments::Literal>> {
        &self.literals
    }
}

pub fn rust_printer<'a>(writer: Box<Write + 'a>) -> State<'a> {
    static NO_ANN: NoAnn = NoAnn;
    rust_printer_annotated(writer, &NO_ANN)
}

pub fn rust_printer_annotated<'a>(writer: Box<Write + 'a>, ann: &'a PpAnn) -> State<'a> {
    State {
        s: pp::mk_printer(writer, default_columns),
        cm: None,
        comments: None,
        literals: None,
        cur_cmnt_and_lit: ast_pp::CurrentCommentAndLiteral {
            cur_cmnt: 0,
            cur_lit: 0,
        },
        boxes: Vec::new(),
        ann: ann,
    }
}

#[allow(non_upper_case_globals)]
pub const indent_unit: usize = 4;

#[allow(non_upper_case_globals)]
pub const default_columns: usize = 78;


/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments and literals to
/// copy forward.
pub fn print_crate<'a>(cm: &'a CodeMap,
                       span_diagnostic: &diagnostic::SpanHandler,
                       krate: &hir::Crate,
                       filename: String,
                       input: &mut Read,
                       out: Box<Write + 'a>,
                       ann: &'a PpAnn,
                       is_expanded: bool)
                       -> io::Result<()> {
    let mut s = State::new_from_input(cm, span_diagnostic, filename, input, out, ann, is_expanded);

    // When printing the AST, we sometimes need to inject `#[no_std]` here.
    // Since you can't compile the HIR, it's not necessary.

    try!(s.print_mod(&krate.module, &krate.attrs));
    try!(s.print_remaining_comments());
    eof(&mut s.s)
}

impl<'a> State<'a> {
    pub fn new_from_input(cm: &'a CodeMap,
                          span_diagnostic: &diagnostic::SpanHandler,
                          filename: String,
                          input: &mut Read,
                          out: Box<Write + 'a>,
                          ann: &'a PpAnn,
                          is_expanded: bool)
                          -> State<'a> {
        let (cmnts, lits) = comments::gather_comments_and_literals(span_diagnostic,
                                                                   filename,
                                                                   input);

        State::new(cm,
                   out,
                   ann,
                   Some(cmnts),
                   // If the code is post expansion, don't use the table of
                   // literals, since it doesn't correspond with the literals
                   // in the AST anymore.
                   if is_expanded {
                       None
                   } else {
                       Some(lits)
                   })
    }

    pub fn new(cm: &'a CodeMap,
               out: Box<Write + 'a>,
               ann: &'a PpAnn,
               comments: Option<Vec<comments::Comment>>,
               literals: Option<Vec<comments::Literal>>)
               -> State<'a> {
        State {
            s: pp::mk_printer(out, default_columns),
            cm: Some(cm),
            comments: comments.clone(),
            literals: literals.clone(),
            cur_cmnt_and_lit: ast_pp::CurrentCommentAndLiteral {
                cur_cmnt: 0,
                cur_lit: 0,
            },
            boxes: Vec::new(),
            ann: ann,
        }
    }
}

pub fn to_string<F>(f: F) -> String
    where F: FnOnce(&mut State) -> io::Result<()>
{
    let mut wr = Vec::new();
    {
        let mut printer = rust_printer(Box::new(&mut wr));
        f(&mut printer).unwrap();
        eof(&mut printer.s).unwrap();
    }
    String::from_utf8(wr).unwrap()
}

pub fn binop_to_string(op: BinOpToken) -> &'static str {
    match op {
        token::Plus => "+",
        token::Minus => "-",
        token::Star => "*",
        token::Slash => "/",
        token::Percent => "%",
        token::Caret => "^",
        token::And => "&",
        token::Or => "|",
        token::Shl => "<<",
        token::Shr => ">>",
    }
}

pub fn ty_to_string(ty: &hir::Ty) -> String {
    to_string(|s| s.print_type(ty))
}

pub fn bounds_to_string(bounds: &[hir::TyParamBound]) -> String {
    to_string(|s| s.print_bounds("", bounds))
}

pub fn pat_to_string(pat: &hir::Pat) -> String {
    to_string(|s| s.print_pat(pat))
}

pub fn arm_to_string(arm: &hir::Arm) -> String {
    to_string(|s| s.print_arm(arm))
}

pub fn expr_to_string(e: &hir::Expr) -> String {
    to_string(|s| s.print_expr(e))
}

pub fn lifetime_to_string(e: &hir::Lifetime) -> String {
    to_string(|s| s.print_lifetime(e))
}

pub fn stmt_to_string(stmt: &hir::Stmt) -> String {
    to_string(|s| s.print_stmt(stmt))
}

pub fn item_to_string(i: &hir::Item) -> String {
    to_string(|s| s.print_item(i))
}

pub fn impl_item_to_string(i: &hir::ImplItem) -> String {
    to_string(|s| s.print_impl_item(i))
}

pub fn trait_item_to_string(i: &hir::TraitItem) -> String {
    to_string(|s| s.print_trait_item(i))
}

pub fn generics_to_string(generics: &hir::Generics) -> String {
    to_string(|s| s.print_generics(generics))
}

pub fn where_clause_to_string(i: &hir::WhereClause) -> String {
    to_string(|s| s.print_where_clause(i))
}

pub fn fn_block_to_string(p: &hir::FnDecl) -> String {
    to_string(|s| s.print_fn_block_args(p))
}

pub fn path_to_string(p: &hir::Path) -> String {
    to_string(|s| s.print_path(p, false, 0))
}

pub fn name_to_string(name: ast::Name) -> String {
    to_string(|s| s.print_name(name))
}

pub fn fun_to_string(decl: &hir::FnDecl,
                     unsafety: hir::Unsafety,
                     constness: hir::Constness,
                     name: ast::Name,
                     opt_explicit_self: Option<&hir::ExplicitSelf_>,
                     generics: &hir::Generics)
                     -> String {
    to_string(|s| {
        try!(s.head(""));
        try!(s.print_fn(decl,
                        unsafety,
                        constness,
                        abi::Rust,
                        Some(name),
                        generics,
                        opt_explicit_self,
                        hir::Inherited));
        try!(s.end()); // Close the head box
        s.end() // Close the outer box
    })
}

pub fn block_to_string(blk: &hir::Block) -> String {
    to_string(|s| {
        // containing cbox, will be closed by print-block at }
        try!(s.cbox(indent_unit));
        // head-ibox, will be closed by print-block after {
        try!(s.ibox(0));
        s.print_block(blk)
    })
}

pub fn explicit_self_to_string(explicit_self: &hir::ExplicitSelf_) -> String {
    to_string(|s| s.print_explicit_self(explicit_self, hir::MutImmutable).map(|_| {}))
}

pub fn variant_to_string(var: &hir::Variant) -> String {
    to_string(|s| s.print_variant(var))
}

pub fn arg_to_string(arg: &hir::Arg) -> String {
    to_string(|s| s.print_arg(arg))
}

pub fn visibility_qualified(vis: hir::Visibility, s: &str) -> String {
    match vis {
        hir::Public => format!("pub {}", s),
        hir::Inherited => s.to_string(),
    }
}

fn needs_parentheses(expr: &hir::Expr) -> bool {
    match expr.node {
        hir::ExprAssign(..) |
        hir::ExprBinary(..) |
        hir::ExprClosure(..) |
        hir::ExprAssignOp(..) |
        hir::ExprCast(..) => true,
        _ => false,
    }
}

impl<'a> State<'a> {
    pub fn cbox(&mut self, u: usize) -> io::Result<()> {
        self.boxes.push(pp::Breaks::Consistent);
        pp::cbox(&mut self.s, u)
    }

    pub fn nbsp(&mut self) -> io::Result<()> {
        word(&mut self.s, " ")
    }

    pub fn word_nbsp(&mut self, w: &str) -> io::Result<()> {
        try!(word(&mut self.s, w));
        self.nbsp()
    }

    pub fn head(&mut self, w: &str) -> io::Result<()> {
        // outer-box is consistent
        try!(self.cbox(indent_unit));
        // head-box is inconsistent
        try!(self.ibox(w.len() + 1));
        // keyword that starts the head
        if !w.is_empty() {
            try!(self.word_nbsp(w));
        }
        Ok(())
    }

    pub fn bopen(&mut self) -> io::Result<()> {
        try!(word(&mut self.s, "{"));
        self.end() // close the head-box
    }

    pub fn bclose_(&mut self, span: codemap::Span, indented: usize) -> io::Result<()> {
        self.bclose_maybe_open(span, indented, true)
    }
    pub fn bclose_maybe_open(&mut self,
                             span: codemap::Span,
                             indented: usize,
                             close_box: bool)
                             -> io::Result<()> {
        try!(self.maybe_print_comment(span.hi));
        try!(self.break_offset_if_not_bol(1, -(indented as isize)));
        try!(word(&mut self.s, "}"));
        if close_box {
            try!(self.end()); // close the outer-box
        }
        Ok(())
    }
    pub fn bclose(&mut self, span: codemap::Span) -> io::Result<()> {
        self.bclose_(span, indent_unit)
    }

    pub fn in_cbox(&self) -> bool {
        match self.boxes.last() {
            Some(&last_box) => last_box == pp::Breaks::Consistent,
            None => false,
        }
    }
    pub fn space_if_not_bol(&mut self) -> io::Result<()> {
        if !self.is_bol() {
            try!(space(&mut self.s));
        }
        Ok(())
    }
    pub fn break_offset_if_not_bol(&mut self, n: usize, off: isize) -> io::Result<()> {
        if !self.is_bol() {
            break_offset(&mut self.s, n, off)
        } else {
            if off != 0 && self.s.last_token().is_hardbreak_tok() {
                // We do something pretty sketchy here: tuck the nonzero
                // offset-adjustment we were going to deposit along with the
                // break into the previous hardbreak.
                self.s.replace_last_token(pp::hardbreak_tok_offset(off));
            }
            Ok(())
        }
    }

    // Synthesizes a comment that was not textually present in the original source
    // file.
    pub fn synth_comment(&mut self, text: String) -> io::Result<()> {
        try!(word(&mut self.s, "/*"));
        try!(space(&mut self.s));
        try!(word(&mut self.s, &text[..]));
        try!(space(&mut self.s));
        word(&mut self.s, "*/")
    }


    pub fn commasep_cmnt<T, F, G>(&mut self,
                                  b: Breaks,
                                  elts: &[T],
                                  mut op: F,
                                  mut get_span: G)
                                  -> io::Result<()>
        where F: FnMut(&mut State, &T) -> io::Result<()>,
              G: FnMut(&T) -> codemap::Span
    {
        try!(self.rbox(0, b));
        let len = elts.len();
        let mut i = 0;
        for elt in elts {
            try!(self.maybe_print_comment(get_span(elt).hi));
            try!(op(self, elt));
            i += 1;
            if i < len {
                try!(word(&mut self.s, ","));
                try!(self.maybe_print_trailing_comment(get_span(elt), Some(get_span(&elts[i]).hi)));
                try!(self.space_if_not_bol());
            }
        }
        self.end()
    }

    pub fn commasep_exprs(&mut self, b: Breaks, exprs: &[P<hir::Expr>]) -> io::Result<()> {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(&**e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &hir::Mod, attrs: &[ast::Attribute]) -> io::Result<()> {
        try!(self.print_inner_attributes(attrs));
        for item in &_mod.items {
            try!(self.print_item(&**item));
        }
        Ok(())
    }

    pub fn print_foreign_mod(&mut self,
                             nmod: &hir::ForeignMod,
                             attrs: &[ast::Attribute])
                             -> io::Result<()> {
        try!(self.print_inner_attributes(attrs));
        for item in &nmod.items {
            try!(self.print_foreign_item(&**item));
        }
        Ok(())
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &Option<hir::Lifetime>) -> io::Result<()> {
        if let Some(l) = *lifetime {
            try!(self.print_lifetime(&l));
            try!(self.nbsp());
        }
        Ok(())
    }

    pub fn print_type(&mut self, ty: &hir::Ty) -> io::Result<()> {
        try!(self.maybe_print_comment(ty.span.lo));
        try!(self.ibox(0));
        match ty.node {
            hir::TyVec(ref ty) => {
                try!(word(&mut self.s, "["));
                try!(self.print_type(&**ty));
                try!(word(&mut self.s, "]"));
            }
            hir::TyPtr(ref mt) => {
                try!(word(&mut self.s, "*"));
                match mt.mutbl {
                    hir::MutMutable => try!(self.word_nbsp("mut")),
                    hir::MutImmutable => try!(self.word_nbsp("const")),
                }
                try!(self.print_type(&*mt.ty));
            }
            hir::TyRptr(ref lifetime, ref mt) => {
                try!(word(&mut self.s, "&"));
                try!(self.print_opt_lifetime(lifetime));
                try!(self.print_mt(mt));
            }
            hir::TyTup(ref elts) => {
                try!(self.popen());
                try!(self.commasep(Inconsistent, &elts[..], |s, ty| s.print_type(&**ty)));
                if elts.len() == 1 {
                    try!(word(&mut self.s, ","));
                }
                try!(self.pclose());
            }
            hir::TyParen(ref typ) => {
                try!(self.popen());
                try!(self.print_type(&**typ));
                try!(self.pclose());
            }
            hir::TyBareFn(ref f) => {
                let generics = hir::Generics {
                    lifetimes: f.lifetimes.clone(),
                    ty_params: OwnedSlice::empty(),
                    where_clause: hir::WhereClause {
                        id: ast::DUMMY_NODE_ID,
                        predicates: Vec::new(),
                    },
                };
                try!(self.print_ty_fn(f.abi, f.unsafety, &*f.decl, None, &generics, None));
            }
            hir::TyPath(None, ref path) => {
                try!(self.print_path(path, false, 0));
            }
            hir::TyPath(Some(ref qself), ref path) => {
                try!(self.print_qpath(path, qself, false))
            }
            hir::TyObjectSum(ref ty, ref bounds) => {
                try!(self.print_type(&**ty));
                try!(self.print_bounds("+", &bounds[..]));
            }
            hir::TyPolyTraitRef(ref bounds) => {
                try!(self.print_bounds("", &bounds[..]));
            }
            hir::TyFixedLengthVec(ref ty, ref v) => {
                try!(word(&mut self.s, "["));
                try!(self.print_type(&**ty));
                try!(word(&mut self.s, "; "));
                try!(self.print_expr(&**v));
                try!(word(&mut self.s, "]"));
            }
            hir::TyTypeof(ref e) => {
                try!(word(&mut self.s, "typeof("));
                try!(self.print_expr(&**e));
                try!(word(&mut self.s, ")"));
            }
            hir::TyInfer => {
                try!(word(&mut self.s, "_"));
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self, item: &hir::ForeignItem) -> io::Result<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(item.span.lo));
        try!(self.print_outer_attributes(&item.attrs));
        match item.node {
            hir::ForeignItemFn(ref decl, ref generics) => {
                try!(self.head(""));
                try!(self.print_fn(decl,
                                   hir::Unsafety::Normal,
                                   hir::Constness::NotConst,
                                   abi::Rust,
                                   Some(item.name),
                                   generics,
                                   None,
                                   item.vis));
                try!(self.end()); // end head-ibox
                try!(word(&mut self.s, ";"));
                self.end() // end the outer fn box
            }
            hir::ForeignItemStatic(ref t, m) => {
                try!(self.head(&visibility_qualified(item.vis, "static")));
                if m {
                    try!(self.word_space("mut"));
                }
                try!(self.print_name(item.name));
                try!(self.word_space(":"));
                try!(self.print_type(&**t));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the head-ibox
                self.end() // end the outer cbox
            }
        }
    }

    fn print_associated_const(&mut self,
                              name: ast::Name,
                              ty: &hir::Ty,
                              default: Option<&hir::Expr>,
                              vis: hir::Visibility)
                              -> io::Result<()> {
        try!(word(&mut self.s, &visibility_qualified(vis, "")));
        try!(self.word_space("const"));
        try!(self.print_name(name));
        try!(self.word_space(":"));
        try!(self.print_type(ty));
        if let Some(expr) = default {
            try!(space(&mut self.s));
            try!(self.word_space("="));
            try!(self.print_expr(expr));
        }
        word(&mut self.s, ";")
    }

    fn print_associated_type(&mut self,
                             name: ast::Name,
                             bounds: Option<&hir::TyParamBounds>,
                             ty: Option<&hir::Ty>)
                             -> io::Result<()> {
        try!(self.word_space("type"));
        try!(self.print_name(name));
        if let Some(bounds) = bounds {
            try!(self.print_bounds(":", bounds));
        }
        if let Some(ty) = ty {
            try!(space(&mut self.s));
            try!(self.word_space("="));
            try!(self.print_type(ty));
        }
        word(&mut self.s, ";")
    }

    /// Pretty-print an item
    pub fn print_item(&mut self, item: &hir::Item) -> io::Result<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(item.span.lo));
        try!(self.print_outer_attributes(&item.attrs));
        try!(self.ann.pre(self, NodeItem(item)));
        match item.node {
            hir::ItemExternCrate(ref optional_path) => {
                try!(self.head(&visibility_qualified(item.vis, "extern crate")));
                if let Some(p) = *optional_path {
                    let val = p.as_str();
                    if val.contains("-") {
                        try!(self.print_string(&val, ast::CookedStr));
                    } else {
                        try!(self.print_name(p));
                    }
                    try!(space(&mut self.s));
                    try!(word(&mut self.s, "as"));
                    try!(space(&mut self.s));
                }
                try!(self.print_name(item.name));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end inner head-block
                try!(self.end()); // end outer head-block
            }
            hir::ItemUse(ref vp) => {
                try!(self.head(&visibility_qualified(item.vis, "use")));
                try!(self.print_view_path(&**vp));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end inner head-block
                try!(self.end()); // end outer head-block
            }
            hir::ItemStatic(ref ty, m, ref expr) => {
                try!(self.head(&visibility_qualified(item.vis, "static")));
                if m == hir::MutMutable {
                    try!(self.word_space("mut"));
                }
                try!(self.print_name(item.name));
                try!(self.word_space(":"));
                try!(self.print_type(&**ty));
                try!(space(&mut self.s));
                try!(self.end()); // end the head-ibox

                try!(self.word_space("="));
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the outer cbox
            }
            hir::ItemConst(ref ty, ref expr) => {
                try!(self.head(&visibility_qualified(item.vis, "const")));
                try!(self.print_name(item.name));
                try!(self.word_space(":"));
                try!(self.print_type(&**ty));
                try!(space(&mut self.s));
                try!(self.end()); // end the head-ibox

                try!(self.word_space("="));
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the outer cbox
            }
            hir::ItemFn(ref decl, unsafety, constness, abi, ref typarams, ref body) => {
                try!(self.head(""));
                try!(self.print_fn(decl,
                                   unsafety,
                                   constness,
                                   abi,
                                   Some(item.name),
                                   typarams,
                                   None,
                                   item.vis));
                try!(word(&mut self.s, " "));
                try!(self.print_block_with_attrs(&**body, &item.attrs));
            }
            hir::ItemMod(ref _mod) => {
                try!(self.head(&visibility_qualified(item.vis, "mod")));
                try!(self.print_name(item.name));
                try!(self.nbsp());
                try!(self.bopen());
                try!(self.print_mod(_mod, &item.attrs));
                try!(self.bclose(item.span));
            }
            hir::ItemForeignMod(ref nmod) => {
                try!(self.head("extern"));
                try!(self.word_nbsp(&nmod.abi.to_string()));
                try!(self.bopen());
                try!(self.print_foreign_mod(nmod, &item.attrs));
                try!(self.bclose(item.span));
            }
            hir::ItemTy(ref ty, ref params) => {
                try!(self.ibox(indent_unit));
                try!(self.ibox(0));
                try!(self.word_nbsp(&visibility_qualified(item.vis, "type")));
                try!(self.print_name(item.name));
                try!(self.print_generics(params));
                try!(self.end()); // end the inner ibox

                try!(self.print_where_clause(&params.where_clause));
                try!(space(&mut self.s));
                try!(self.word_space("="));
                try!(self.print_type(&**ty));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the outer ibox
            }
            hir::ItemEnum(ref enum_definition, ref params) => {
                try!(self.print_enum_def(enum_definition, params, item.name, item.span, item.vis));
            }
            hir::ItemStruct(ref struct_def, ref generics) => {
                try!(self.head(&visibility_qualified(item.vis, "struct")));
                try!(self.print_struct(&**struct_def, generics, item.name, item.span, true));
            }

            hir::ItemDefaultImpl(unsafety, ref trait_ref) => {
                try!(self.head(""));
                try!(self.print_visibility(item.vis));
                try!(self.print_unsafety(unsafety));
                try!(self.word_nbsp("impl"));
                try!(self.print_trait_ref(trait_ref));
                try!(space(&mut self.s));
                try!(self.word_space("for"));
                try!(self.word_space(".."));
                try!(self.bopen());
                try!(self.bclose(item.span));
            }
            hir::ItemImpl(unsafety,
                          polarity,
                          ref generics,
                          ref opt_trait,
                          ref ty,
                          ref impl_items) => {
                try!(self.head(""));
                try!(self.print_visibility(item.vis));
                try!(self.print_unsafety(unsafety));
                try!(self.word_nbsp("impl"));

                if generics.is_parameterized() {
                    try!(self.print_generics(generics));
                    try!(space(&mut self.s));
                }

                match polarity {
                    hir::ImplPolarity::Negative => {
                        try!(word(&mut self.s, "!"));
                    }
                    _ => {}
                }

                match opt_trait {
                    &Some(ref t) => {
                        try!(self.print_trait_ref(t));
                        try!(space(&mut self.s));
                        try!(self.word_space("for"));
                    }
                    &None => {}
                }

                try!(self.print_type(&**ty));
                try!(self.print_where_clause(&generics.where_clause));

                try!(space(&mut self.s));
                try!(self.bopen());
                try!(self.print_inner_attributes(&item.attrs));
                for impl_item in impl_items {
                    try!(self.print_impl_item(impl_item));
                }
                try!(self.bclose(item.span));
            }
            hir::ItemTrait(unsafety, ref generics, ref bounds, ref trait_items) => {
                try!(self.head(""));
                try!(self.print_visibility(item.vis));
                try!(self.print_unsafety(unsafety));
                try!(self.word_nbsp("trait"));
                try!(self.print_name(item.name));
                try!(self.print_generics(generics));
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let TraitTyParamBound(ref ptr, hir::TraitBoundModifier::Maybe) = *b {
                        try!(space(&mut self.s));
                        try!(self.word_space("for ?"));
                        try!(self.print_trait_ref(&ptr.trait_ref));
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                try!(self.print_bounds(":", &real_bounds[..]));
                try!(self.print_where_clause(&generics.where_clause));
                try!(word(&mut self.s, " "));
                try!(self.bopen());
                for trait_item in trait_items {
                    try!(self.print_trait_item(trait_item));
                }
                try!(self.bclose(item.span));
            }
        }
        self.ann.post(self, NodeItem(item))
    }

    fn print_trait_ref(&mut self, t: &hir::TraitRef) -> io::Result<()> {
        self.print_path(&t.path, false, 0)
    }

    fn print_formal_lifetime_list(&mut self, lifetimes: &[hir::LifetimeDef]) -> io::Result<()> {
        if !lifetimes.is_empty() {
            try!(word(&mut self.s, "for<"));
            let mut comma = false;
            for lifetime_def in lifetimes {
                if comma {
                    try!(self.word_space(","))
                }
                try!(self.print_lifetime_def(lifetime_def));
                comma = true;
            }
            try!(word(&mut self.s, ">"));
        }
        Ok(())
    }

    fn print_poly_trait_ref(&mut self, t: &hir::PolyTraitRef) -> io::Result<()> {
        try!(self.print_formal_lifetime_list(&t.bound_lifetimes));
        self.print_trait_ref(&t.trait_ref)
    }

    pub fn print_enum_def(&mut self,
                          enum_definition: &hir::EnumDef,
                          generics: &hir::Generics,
                          name: ast::Name,
                          span: codemap::Span,
                          visibility: hir::Visibility)
                          -> io::Result<()> {
        try!(self.head(&visibility_qualified(visibility, "enum")));
        try!(self.print_name(name));
        try!(self.print_generics(generics));
        try!(self.print_where_clause(&generics.where_clause));
        try!(space(&mut self.s));
        self.print_variants(&enum_definition.variants, span)
    }

    pub fn print_variants(&mut self,
                          variants: &[P<hir::Variant>],
                          span: codemap::Span)
                          -> io::Result<()> {
        try!(self.bopen());
        for v in variants {
            try!(self.space_if_not_bol());
            try!(self.maybe_print_comment(v.span.lo));
            try!(self.print_outer_attributes(&v.node.attrs));
            try!(self.ibox(indent_unit));
            try!(self.print_variant(&**v));
            try!(word(&mut self.s, ","));
            try!(self.end());
            try!(self.maybe_print_trailing_comment(v.span, None));
        }
        self.bclose(span)
    }

    pub fn print_visibility(&mut self, vis: hir::Visibility) -> io::Result<()> {
        match vis {
            hir::Public => self.word_nbsp("pub"),
            hir::Inherited => Ok(()),
        }
    }

    pub fn print_struct(&mut self,
                        struct_def: &hir::VariantData,
                        generics: &hir::Generics,
                        name: ast::Name,
                        span: codemap::Span,
                        print_finalizer: bool)
                        -> io::Result<()> {
        try!(self.print_name(name));
        try!(self.print_generics(generics));
        if !struct_def.is_struct() {
            if struct_def.is_tuple() {
                try!(self.popen());
                try!(self.commasep_iter(Inconsistent,
                                   struct_def.fields(),
                                   |s, field| {
                                       match field.node.kind {
                                           hir::NamedField(..) => panic!("unexpected named field"),
                                           hir::UnnamedField(vis) => {
                                               try!(s.print_visibility(vis));
                                               try!(s.maybe_print_comment(field.span.lo));
                                               s.print_type(&*field.node.ty)
                                           }
                                       }
                                   }));
                try!(self.pclose());
            }
            try!(self.print_where_clause(&generics.where_clause));
            if print_finalizer {
                try!(word(&mut self.s, ";"));
            }
            try!(self.end());
            self.end() // close the outer-box
        } else {
            try!(self.print_where_clause(&generics.where_clause));
            try!(self.nbsp());
            try!(self.bopen());
            try!(self.hardbreak_if_not_bol());

            for field in struct_def.fields() {
                match field.node.kind {
                    hir::UnnamedField(..) => panic!("unexpected unnamed field"),
                    hir::NamedField(name, visibility) => {
                        try!(self.hardbreak_if_not_bol());
                        try!(self.maybe_print_comment(field.span.lo));
                        try!(self.print_outer_attributes(&field.node.attrs));
                        try!(self.print_visibility(visibility));
                        try!(self.print_name(name));
                        try!(self.word_nbsp(":"));
                        try!(self.print_type(&*field.node.ty));
                        try!(word(&mut self.s, ","));
                    }
                }
            }

            self.bclose(span)
        }
    }

    pub fn print_variant(&mut self, v: &hir::Variant) -> io::Result<()> {
        try!(self.head(""));
        let generics = ::util::empty_generics();
        try!(self.print_struct(&v.node.data, &generics, v.node.name, v.span, false));
        match v.node.disr_expr {
            Some(ref d) => {
                try!(space(&mut self.s));
                try!(self.word_space("="));
                self.print_expr(&**d)
            }
            _ => Ok(()),
        }
    }

    pub fn print_method_sig(&mut self,
                            name: ast::Name,
                            m: &hir::MethodSig,
                            vis: hir::Visibility)
                            -> io::Result<()> {
        self.print_fn(&m.decl,
                      m.unsafety,
                      m.constness,
                      m.abi,
                      Some(name),
                      &m.generics,
                      Some(&m.explicit_self.node),
                      vis)
    }

    pub fn print_trait_item(&mut self, ti: &hir::TraitItem) -> io::Result<()> {
        try!(self.ann.pre(self, NodeSubItem(ti.id)));
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(ti.span.lo));
        try!(self.print_outer_attributes(&ti.attrs));
        match ti.node {
            hir::ConstTraitItem(ref ty, ref default) => {
                try!(self.print_associated_const(ti.name,
                                                 &ty,
                                                 default.as_ref().map(|expr| &**expr),
                                                 hir::Inherited));
            }
            hir::MethodTraitItem(ref sig, ref body) => {
                if body.is_some() {
                    try!(self.head(""));
                }
                try!(self.print_method_sig(ti.name, sig, hir::Inherited));
                if let Some(ref body) = *body {
                    try!(self.nbsp());
                    try!(self.print_block_with_attrs(body, &ti.attrs));
                } else {
                    try!(word(&mut self.s, ";"));
                }
            }
            hir::TypeTraitItem(ref bounds, ref default) => {
                try!(self.print_associated_type(ti.name,
                                                Some(bounds),
                                                default.as_ref().map(|ty| &**ty)));
            }
        }
        self.ann.post(self, NodeSubItem(ti.id))
    }

    pub fn print_impl_item(&mut self, ii: &hir::ImplItem) -> io::Result<()> {
        try!(self.ann.pre(self, NodeSubItem(ii.id)));
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(ii.span.lo));
        try!(self.print_outer_attributes(&ii.attrs));
        match ii.node {
            hir::ConstImplItem(ref ty, ref expr) => {
                try!(self.print_associated_const(ii.name, &ty, Some(&expr), ii.vis));
            }
            hir::MethodImplItem(ref sig, ref body) => {
                try!(self.head(""));
                try!(self.print_method_sig(ii.name, sig, ii.vis));
                try!(self.nbsp());
                try!(self.print_block_with_attrs(body, &ii.attrs));
            }
            hir::TypeImplItem(ref ty) => {
                try!(self.print_associated_type(ii.name, None, Some(ty)));
            }
        }
        self.ann.post(self, NodeSubItem(ii.id))
    }

    pub fn print_stmt(&mut self, st: &hir::Stmt) -> io::Result<()> {
        try!(self.maybe_print_comment(st.span.lo));
        match st.node {
            hir::StmtDecl(ref decl, _) => {
                try!(self.print_decl(&**decl));
            }
            hir::StmtExpr(ref expr, _) => {
                try!(self.space_if_not_bol());
                try!(self.print_expr(&**expr));
            }
            hir::StmtSemi(ref expr, _) => {
                try!(self.space_if_not_bol());
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, ";"));
            }
        }
        if stmt_ends_with_semi(&st.node) {
            try!(word(&mut self.s, ";"));
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &hir::Block) -> io::Result<()> {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &hir::Block) -> io::Result<()> {
        self.print_block_unclosed_indent(blk, indent_unit)
    }

    pub fn print_block_unclosed_indent(&mut self,
                                       blk: &hir::Block,
                                       indented: usize)
                                       -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    pub fn print_block_with_attrs(&mut self,
                                  blk: &hir::Block,
                                  attrs: &[ast::Attribute])
                                  -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, indent_unit, attrs, true)
    }

    pub fn print_block_maybe_unclosed(&mut self,
                                      blk: &hir::Block,
                                      indented: usize,
                                      attrs: &[ast::Attribute],
                                      close_box: bool)
                                      -> io::Result<()> {
        match blk.rules {
            hir::UnsafeBlock(..) => try!(self.word_space("unsafe")),
            hir::PushUnsafeBlock(..) => try!(self.word_space("push_unsafe")),
            hir::PopUnsafeBlock(..) => try!(self.word_space("pop_unsafe")),
            hir::PushUnstableBlock => try!(self.word_space("push_unstable")),
            hir::PopUnstableBlock => try!(self.word_space("pop_unstable")),
            hir::DefaultBlock => (),
        }
        try!(self.maybe_print_comment(blk.span.lo));
        try!(self.ann.pre(self, NodeBlock(blk)));
        try!(self.bopen());

        try!(self.print_inner_attributes(attrs));

        for st in &blk.stmts {
            try!(self.print_stmt(&**st));
        }
        match blk.expr {
            Some(ref expr) => {
                try!(self.space_if_not_bol());
                try!(self.print_expr(&**expr));
                try!(self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi)));
            }
            _ => (),
        }
        try!(self.bclose_maybe_open(blk.span, indented, close_box));
        self.ann.post(self, NodeBlock(blk))
    }

    fn print_else(&mut self, els: Option<&hir::Expr>) -> io::Result<()> {
        match els {
            Some(_else) => {
                match _else.node {
                    // "another else-if"
                    hir::ExprIf(ref i, ref then, ref e) => {
                        try!(self.cbox(indent_unit - 1));
                        try!(self.ibox(0));
                        try!(word(&mut self.s, " else if "));
                        try!(self.print_expr(&**i));
                        try!(space(&mut self.s));
                        try!(self.print_block(&**then));
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "final else"
                    hir::ExprBlock(ref b) => {
                        try!(self.cbox(indent_unit - 1));
                        try!(self.ibox(0));
                        try!(word(&mut self.s, " else "));
                        self.print_block(&**b)
                    }
                    // BLEAH, constraints would be great here
                    _ => {
                        panic!("print_if saw if with weird alternative");
                    }
                }
            }
            _ => Ok(()),
        }
    }

    pub fn print_if(&mut self,
                    test: &hir::Expr,
                    blk: &hir::Block,
                    elseopt: Option<&hir::Expr>)
                    -> io::Result<()> {
        try!(self.head("if"));
        try!(self.print_expr(test));
        try!(space(&mut self.s));
        try!(self.print_block(blk));
        self.print_else(elseopt)
    }

    pub fn print_if_let(&mut self,
                        pat: &hir::Pat,
                        expr: &hir::Expr,
                        blk: &hir::Block,
                        elseopt: Option<&hir::Expr>)
                        -> io::Result<()> {
        try!(self.head("if let"));
        try!(self.print_pat(pat));
        try!(space(&mut self.s));
        try!(self.word_space("="));
        try!(self.print_expr(expr));
        try!(space(&mut self.s));
        try!(self.print_block(blk));
        self.print_else(elseopt)
    }


    fn print_call_post(&mut self, args: &[P<hir::Expr>]) -> io::Result<()> {
        try!(self.popen());
        try!(self.commasep_exprs(Inconsistent, args));
        self.pclose()
    }

    pub fn print_expr_maybe_paren(&mut self, expr: &hir::Expr) -> io::Result<()> {
        let needs_par = needs_parentheses(expr);
        if needs_par {
            try!(self.popen());
        }
        try!(self.print_expr(expr));
        if needs_par {
            try!(self.pclose());
        }
        Ok(())
    }

    fn print_expr_vec(&mut self, exprs: &[P<hir::Expr>]) -> io::Result<()> {
        try!(self.ibox(indent_unit));
        try!(word(&mut self.s, "["));
        try!(self.commasep_exprs(Inconsistent, &exprs[..]));
        try!(word(&mut self.s, "]"));
        self.end()
    }

    fn print_expr_repeat(&mut self, element: &hir::Expr, count: &hir::Expr) -> io::Result<()> {
        try!(self.ibox(indent_unit));
        try!(word(&mut self.s, "["));
        try!(self.print_expr(element));
        try!(self.word_space(";"));
        try!(self.print_expr(count));
        try!(word(&mut self.s, "]"));
        self.end()
    }

    fn print_expr_struct(&mut self,
                         path: &hir::Path,
                         fields: &[hir::Field],
                         wth: &Option<P<hir::Expr>>)
                         -> io::Result<()> {
        try!(self.print_path(path, true, 0));
        try!(word(&mut self.s, "{"));
        try!(self.commasep_cmnt(Consistent,
                                &fields[..],
                                |s, field| {
                                    try!(s.ibox(indent_unit));
                                    try!(s.print_name(field.name.node));
                                    try!(s.word_space(":"));
                                    try!(s.print_expr(&*field.expr));
                                    s.end()
                                },
                                |f| f.span));
        match *wth {
            Some(ref expr) => {
                try!(self.ibox(indent_unit));
                if !fields.is_empty() {
                    try!(word(&mut self.s, ","));
                    try!(space(&mut self.s));
                }
                try!(word(&mut self.s, ".."));
                try!(self.print_expr(&**expr));
                try!(self.end());
            }
            _ => if !fields.is_empty() {
                try!(word(&mut self.s, ","))
            },
        }
        try!(word(&mut self.s, "}"));
        Ok(())
    }

    fn print_expr_tup(&mut self, exprs: &[P<hir::Expr>]) -> io::Result<()> {
        try!(self.popen());
        try!(self.commasep_exprs(Inconsistent, &exprs[..]));
        if exprs.len() == 1 {
            try!(word(&mut self.s, ","));
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &hir::Expr, args: &[P<hir::Expr>]) -> io::Result<()> {
        try!(self.print_expr_maybe_paren(func));
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self,
                              name: Spanned<ast::Name>,
                              tys: &[P<hir::Ty>],
                              args: &[P<hir::Expr>])
                              -> io::Result<()> {
        let base_args = &args[1..];
        try!(self.print_expr(&*args[0]));
        try!(word(&mut self.s, "."));
        try!(self.print_name(name.node));
        if !tys.is_empty() {
            try!(word(&mut self.s, "::<"));
            try!(self.commasep(Inconsistent, tys, |s, ty| s.print_type(&**ty)));
            try!(word(&mut self.s, ">"));
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self,
                         op: hir::BinOp,
                         lhs: &hir::Expr,
                         rhs: &hir::Expr)
                         -> io::Result<()> {
        try!(self.print_expr(lhs));
        try!(space(&mut self.s));
        try!(self.word_space(::util::binop_to_string(op.node)));
        self.print_expr(rhs)
    }

    fn print_expr_unary(&mut self, op: hir::UnOp, expr: &hir::Expr) -> io::Result<()> {
        try!(word(&mut self.s, ::util::unop_to_string(op)));
        self.print_expr_maybe_paren(expr)
    }

    fn print_expr_addr_of(&mut self,
                          mutability: hir::Mutability,
                          expr: &hir::Expr)
                          -> io::Result<()> {
        try!(word(&mut self.s, "&"));
        try!(self.print_mutability(mutability));
        self.print_expr_maybe_paren(expr)
    }

    pub fn print_expr(&mut self, expr: &hir::Expr) -> io::Result<()> {
        try!(self.maybe_print_comment(expr.span.lo));
        try!(self.ibox(indent_unit));
        try!(self.ann.pre(self, NodeExpr(expr)));
        match expr.node {
            hir::ExprBox(ref expr) => {
                try!(self.word_space("box"));
                try!(self.print_expr(expr));
            }
            hir::ExprVec(ref exprs) => {
                try!(self.print_expr_vec(&exprs[..]));
            }
            hir::ExprRepeat(ref element, ref count) => {
                try!(self.print_expr_repeat(&**element, &**count));
            }
            hir::ExprStruct(ref path, ref fields, ref wth) => {
                try!(self.print_expr_struct(path, &fields[..], wth));
            }
            hir::ExprTup(ref exprs) => {
                try!(self.print_expr_tup(&exprs[..]));
            }
            hir::ExprCall(ref func, ref args) => {
                try!(self.print_expr_call(&**func, &args[..]));
            }
            hir::ExprMethodCall(name, ref tys, ref args) => {
                try!(self.print_expr_method_call(name, &tys[..], &args[..]));
            }
            hir::ExprBinary(op, ref lhs, ref rhs) => {
                try!(self.print_expr_binary(op, &**lhs, &**rhs));
            }
            hir::ExprUnary(op, ref expr) => {
                try!(self.print_expr_unary(op, &**expr));
            }
            hir::ExprAddrOf(m, ref expr) => {
                try!(self.print_expr_addr_of(m, &**expr));
            }
            hir::ExprLit(ref lit) => {
                try!(self.print_literal(&**lit));
            }
            hir::ExprCast(ref expr, ref ty) => {
                try!(self.print_expr(&**expr));
                try!(space(&mut self.s));
                try!(self.word_space("as"));
                try!(self.print_type(&**ty));
            }
            hir::ExprIf(ref test, ref blk, ref elseopt) => {
                try!(self.print_if(&**test, &**blk, elseopt.as_ref().map(|e| &**e)));
            }
            hir::ExprWhile(ref test, ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    try!(self.print_name(ident.name));
                    try!(self.word_space(":"));
                }
                try!(self.head("while"));
                try!(self.print_expr(&**test));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            hir::ExprLoop(ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    try!(self.print_name(ident.name));
                    try!(self.word_space(":"));
                }
                try!(self.head("loop"));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            hir::ExprMatch(ref expr, ref arms, _) => {
                try!(self.cbox(indent_unit));
                try!(self.ibox(4));
                try!(self.word_nbsp("match"));
                try!(self.print_expr(&**expr));
                try!(space(&mut self.s));
                try!(self.bopen());
                for arm in arms {
                    try!(self.print_arm(arm));
                }
                try!(self.bclose_(expr.span, indent_unit));
            }
            hir::ExprClosure(capture_clause, ref decl, ref body) => {
                try!(self.print_capture_clause(capture_clause));

                try!(self.print_fn_block_args(&**decl));
                try!(space(&mut self.s));

                let default_return = match decl.output {
                    hir::DefaultReturn(..) => true,
                    _ => false,
                };

                if !default_return || !body.stmts.is_empty() || body.expr.is_none() {
                    try!(self.print_block_unclosed(&**body));
                } else {
                    // we extract the block, so as not to create another set of boxes
                    match body.expr.as_ref().unwrap().node {
                        hir::ExprBlock(ref blk) => {
                            try!(self.print_block_unclosed(&**blk));
                        }
                        _ => {
                            // this is a bare expression
                            try!(self.print_expr(body.expr.as_ref().map(|e| &**e).unwrap()));
                            try!(self.end()); // need to close a box
                        }
                    }
                }
                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                try!(self.ibox(0));
            }
            hir::ExprBlock(ref blk) => {
                // containing cbox, will be closed by print-block at }
                try!(self.cbox(indent_unit));
                // head-box, will be closed by print-block after {
                try!(self.ibox(0));
                try!(self.print_block(&**blk));
            }
            hir::ExprAssign(ref lhs, ref rhs) => {
                try!(self.print_expr(&**lhs));
                try!(space(&mut self.s));
                try!(self.word_space("="));
                try!(self.print_expr(&**rhs));
            }
            hir::ExprAssignOp(op, ref lhs, ref rhs) => {
                try!(self.print_expr(&**lhs));
                try!(space(&mut self.s));
                try!(word(&mut self.s, ::util::binop_to_string(op.node)));
                try!(self.word_space("="));
                try!(self.print_expr(&**rhs));
            }
            hir::ExprField(ref expr, name) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "."));
                try!(self.print_name(name.node));
            }
            hir::ExprTupField(ref expr, id) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "."));
                try!(self.print_usize(id.node));
            }
            hir::ExprIndex(ref expr, ref index) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "["));
                try!(self.print_expr(&**index));
                try!(word(&mut self.s, "]"));
            }
            hir::ExprRange(ref start, ref end) => {
                if let &Some(ref e) = start {
                    try!(self.print_expr(&**e));
                }
                try!(word(&mut self.s, ".."));
                if let &Some(ref e) = end {
                    try!(self.print_expr(&**e));
                }
            }
            hir::ExprPath(None, ref path) => {
                try!(self.print_path(path, true, 0))
            }
            hir::ExprPath(Some(ref qself), ref path) => {
                try!(self.print_qpath(path, qself, true))
            }
            hir::ExprBreak(opt_ident) => {
                try!(word(&mut self.s, "break"));
                try!(space(&mut self.s));
                if let Some(ident) = opt_ident {
                    try!(self.print_name(ident.node.name));
                    try!(space(&mut self.s));
                }
            }
            hir::ExprAgain(opt_ident) => {
                try!(word(&mut self.s, "continue"));
                try!(space(&mut self.s));
                if let Some(ident) = opt_ident {
                    try!(self.print_name(ident.node.name));
                    try!(space(&mut self.s))
                }
            }
            hir::ExprRet(ref result) => {
                try!(word(&mut self.s, "return"));
                match *result {
                    Some(ref expr) => {
                        try!(word(&mut self.s, " "));
                        try!(self.print_expr(&**expr));
                    }
                    _ => (),
                }
            }
            hir::ExprInlineAsm(ref a) => {
                try!(word(&mut self.s, "asm!"));
                try!(self.popen());
                try!(self.print_string(&a.asm, a.asm_str_style));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent,
                                   &a.outputs,
                                   |s, &(ref co, ref o, is_rw)| {
                                       match co.slice_shift_char() {
                                           Some(('=', operand)) if is_rw => {
                                               try!(s.print_string(&format!("+{}", operand),
                                                                   ast::CookedStr))
                                           }
                                           _ => try!(s.print_string(&co, ast::CookedStr)),
                                       }
                                       try!(s.popen());
                                       try!(s.print_expr(&**o));
                                       try!(s.pclose());
                                       Ok(())
                                   }));
                try!(space(&mut self.s));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent,
                                   &a.inputs,
                                   |s, &(ref co, ref o)| {
                                       try!(s.print_string(&co, ast::CookedStr));
                                       try!(s.popen());
                                       try!(s.print_expr(&**o));
                                       try!(s.pclose());
                                       Ok(())
                                   }));
                try!(space(&mut self.s));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent,
                                   &a.clobbers,
                                   |s, co| {
                                       try!(s.print_string(&co, ast::CookedStr));
                                       Ok(())
                                   }));

                let mut options = vec!();
                if a.volatile {
                    options.push("volatile");
                }
                if a.alignstack {
                    options.push("alignstack");
                }
                if a.dialect == ast::AsmDialect::Intel {
                    options.push("intel");
                }

                if !options.is_empty() {
                    try!(space(&mut self.s));
                    try!(self.word_space(":"));
                    try!(self.commasep(Inconsistent,
                                       &*options,
                                       |s, &co| {
                                           try!(s.print_string(co, ast::CookedStr));
                                           Ok(())
                                       }));
                }

                try!(self.pclose());
            }
        }
        try!(self.ann.post(self, NodeExpr(expr)));
        self.end()
    }

    pub fn print_local_decl(&mut self, loc: &hir::Local) -> io::Result<()> {
        try!(self.print_pat(&*loc.pat));
        if let Some(ref ty) = loc.ty {
            try!(self.word_space(":"));
            try!(self.print_type(&**ty));
        }
        Ok(())
    }

    pub fn print_decl(&mut self, decl: &hir::Decl) -> io::Result<()> {
        try!(self.maybe_print_comment(decl.span.lo));
        match decl.node {
            hir::DeclLocal(ref loc) => {
                try!(self.space_if_not_bol());
                try!(self.ibox(indent_unit));
                try!(self.word_nbsp("let"));

                try!(self.ibox(indent_unit));
                try!(self.print_local_decl(&**loc));
                try!(self.end());
                if let Some(ref init) = loc.init {
                    try!(self.nbsp());
                    try!(self.word_space("="));
                    try!(self.print_expr(&**init));
                }
                self.end()
            }
            hir::DeclItem(ref item) => self.print_item(&**item),
        }
    }

    pub fn print_usize(&mut self, i: usize) -> io::Result<()> {
        word(&mut self.s, &i.to_string())
    }

    pub fn print_name(&mut self, name: ast::Name) -> io::Result<()> {
        try!(word(&mut self.s, &name.as_str()));
        self.ann.post(self, NodeName(&name))
    }

    pub fn print_for_decl(&mut self, loc: &hir::Local, coll: &hir::Expr) -> io::Result<()> {
        try!(self.print_local_decl(loc));
        try!(space(&mut self.s));
        try!(self.word_space("in"));
        self.print_expr(coll)
    }

    fn print_path(&mut self,
                  path: &hir::Path,
                  colons_before_params: bool,
                  depth: usize)
                  -> io::Result<()> {
        try!(self.maybe_print_comment(path.span.lo));

        let mut first = !path.global;
        for segment in &path.segments[..path.segments.len()-depth] {
            if first {
                first = false
            } else {
                try!(word(&mut self.s, "::"))
            }

            try!(self.print_name(segment.identifier.name));

            try!(self.print_path_parameters(&segment.parameters, colons_before_params));
        }

        Ok(())
    }

    fn print_qpath(&mut self,
                   path: &hir::Path,
                   qself: &hir::QSelf,
                   colons_before_params: bool)
                   -> io::Result<()> {
        try!(word(&mut self.s, "<"));
        try!(self.print_type(&qself.ty));
        if qself.position > 0 {
            try!(space(&mut self.s));
            try!(self.word_space("as"));
            let depth = path.segments.len() - qself.position;
            try!(self.print_path(&path, false, depth));
        }
        try!(word(&mut self.s, ">"));
        try!(word(&mut self.s, "::"));
        let item_segment = path.segments.last().unwrap();
        try!(self.print_name(item_segment.identifier.name));
        self.print_path_parameters(&item_segment.parameters, colons_before_params)
    }

    fn print_path_parameters(&mut self,
                             parameters: &hir::PathParameters,
                             colons_before_params: bool)
                             -> io::Result<()> {
        if parameters.is_empty() {
            return Ok(());
        }

        if colons_before_params {
            try!(word(&mut self.s, "::"))
        }

        match *parameters {
            hir::AngleBracketedParameters(ref data) => {
                try!(word(&mut self.s, "<"));

                let mut comma = false;
                for lifetime in &data.lifetimes {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.print_lifetime(lifetime));
                    comma = true;
                }

                if !data.types.is_empty() {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.commasep(Inconsistent, &data.types, |s, ty| s.print_type(&**ty)));
                    comma = true;
                }

                for binding in data.bindings.iter() {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.print_name(binding.name));
                    try!(space(&mut self.s));
                    try!(self.word_space("="));
                    try!(self.print_type(&*binding.ty));
                    comma = true;
                }

                try!(word(&mut self.s, ">"))
            }

            hir::ParenthesizedParameters(ref data) => {
                try!(word(&mut self.s, "("));
                try!(self.commasep(Inconsistent,
                                   &data.inputs,
                                   |s, ty| s.print_type(&**ty)));
                try!(word(&mut self.s, ")"));

                match data.output {
                    None => {}
                    Some(ref ty) => {
                        try!(self.space_if_not_bol());
                        try!(self.word_space("->"));
                        try!(self.print_type(&**ty));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn print_pat(&mut self, pat: &hir::Pat) -> io::Result<()> {
        try!(self.maybe_print_comment(pat.span.lo));
        try!(self.ann.pre(self, NodePat(pat)));
        /* Pat isn't normalized, but the beauty of it
         is that it doesn't matter */
        match pat.node {
            hir::PatWild(hir::PatWildSingle) => try!(word(&mut self.s, "_")),
            hir::PatWild(hir::PatWildMulti) => try!(word(&mut self.s, "..")),
            hir::PatIdent(binding_mode, ref path1, ref sub) => {
                match binding_mode {
                    hir::BindByRef(mutbl) => {
                        try!(self.word_nbsp("ref"));
                        try!(self.print_mutability(mutbl));
                    }
                    hir::BindByValue(hir::MutImmutable) => {}
                    hir::BindByValue(hir::MutMutable) => {
                        try!(self.word_nbsp("mut"));
                    }
                }
                try!(self.print_name(path1.node.name));
                match *sub {
                    Some(ref p) => {
                        try!(word(&mut self.s, "@"));
                        try!(self.print_pat(&**p));
                    }
                    None => (),
                }
            }
            hir::PatEnum(ref path, ref args_) => {
                try!(self.print_path(path, true, 0));
                match *args_ {
                    None => try!(word(&mut self.s, "(..)")),
                    Some(ref args) => {
                        if !args.is_empty() {
                            try!(self.popen());
                            try!(self.commasep(Inconsistent, &args[..], |s, p| s.print_pat(&**p)));
                            try!(self.pclose());
                        }
                    }
                }
            }
            hir::PatQPath(ref qself, ref path) => {
                try!(self.print_qpath(path, qself, false));
            }
            hir::PatStruct(ref path, ref fields, etc) => {
                try!(self.print_path(path, true, 0));
                try!(self.nbsp());
                try!(self.word_space("{"));
                try!(self.commasep_cmnt(Consistent,
                                        &fields[..],
                                        |s, f| {
                                            try!(s.cbox(indent_unit));
                                            if !f.node.is_shorthand {
                                                try!(s.print_name(f.node.name));
                                                try!(s.word_nbsp(":"));
                                            }
                                            try!(s.print_pat(&*f.node.pat));
                                            s.end()
                                        },
                                        |f| f.node.pat.span));
                if etc {
                    if !fields.is_empty() {
                        try!(self.word_space(","));
                    }
                    try!(word(&mut self.s, ".."));
                }
                try!(space(&mut self.s));
                try!(word(&mut self.s, "}"));
            }
            hir::PatTup(ref elts) => {
                try!(self.popen());
                try!(self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(&**p)));
                if elts.len() == 1 {
                    try!(word(&mut self.s, ","));
                }
                try!(self.pclose());
            }
            hir::PatBox(ref inner) => {
                try!(word(&mut self.s, "box "));
                try!(self.print_pat(&**inner));
            }
            hir::PatRegion(ref inner, mutbl) => {
                try!(word(&mut self.s, "&"));
                if mutbl == hir::MutMutable {
                    try!(word(&mut self.s, "mut "));
                }
                try!(self.print_pat(&**inner));
            }
            hir::PatLit(ref e) => try!(self.print_expr(&**e)),
            hir::PatRange(ref begin, ref end) => {
                try!(self.print_expr(&**begin));
                try!(space(&mut self.s));
                try!(word(&mut self.s, "..."));
                try!(self.print_expr(&**end));
            }
            hir::PatVec(ref before, ref slice, ref after) => {
                try!(word(&mut self.s, "["));
                try!(self.commasep(Inconsistent, &before[..], |s, p| s.print_pat(&**p)));
                if let Some(ref p) = *slice {
                    if !before.is_empty() {
                        try!(self.word_space(","));
                    }
                    try!(self.print_pat(&**p));
                    match **p {
                        hir::Pat { node: hir::PatWild(hir::PatWildMulti), .. } => {
                            // this case is handled by print_pat
                        }
                        _ => try!(word(&mut self.s, "..")),
                    }
                    if !after.is_empty() {
                        try!(self.word_space(","));
                    }
                }
                try!(self.commasep(Inconsistent, &after[..], |s, p| s.print_pat(&**p)));
                try!(word(&mut self.s, "]"));
            }
        }
        self.ann.post(self, NodePat(pat))
    }

    fn print_arm(&mut self, arm: &hir::Arm) -> io::Result<()> {
        // I have no idea why this check is necessary, but here it
        // is :(
        if arm.attrs.is_empty() {
            try!(space(&mut self.s));
        }
        try!(self.cbox(indent_unit));
        try!(self.ibox(0));
        try!(self.print_outer_attributes(&arm.attrs));
        let mut first = true;
        for p in &arm.pats {
            if first {
                first = false;
            } else {
                try!(space(&mut self.s));
                try!(self.word_space("|"));
            }
            try!(self.print_pat(&**p));
        }
        try!(space(&mut self.s));
        if let Some(ref e) = arm.guard {
            try!(self.word_space("if"));
            try!(self.print_expr(&**e));
            try!(space(&mut self.s));
        }
        try!(self.word_space("=>"));

        match arm.body.node {
            hir::ExprBlock(ref blk) => {
                // the block will close the pattern's ibox
                try!(self.print_block_unclosed_indent(&**blk, indent_unit));

                // If it is a user-provided unsafe block, print a comma after it
                if let hir::UnsafeBlock(hir::UserProvided) = blk.rules {
                    try!(word(&mut self.s, ","));
                }
            }
            _ => {
                try!(self.end()); // close the ibox for the pattern
                try!(self.print_expr(&*arm.body));
                try!(word(&mut self.s, ","));
            }
        }
        self.end() // close enclosing cbox
    }

    // Returns whether it printed anything
    fn print_explicit_self(&mut self,
                           explicit_self: &hir::ExplicitSelf_,
                           mutbl: hir::Mutability)
                           -> io::Result<bool> {
        try!(self.print_mutability(mutbl));
        match *explicit_self {
            hir::SelfStatic => {
                return Ok(false);
            }
            hir::SelfValue(_) => {
                try!(word(&mut self.s, "self"));
            }
            hir::SelfRegion(ref lt, m, _) => {
                try!(word(&mut self.s, "&"));
                try!(self.print_opt_lifetime(lt));
                try!(self.print_mutability(m));
                try!(word(&mut self.s, "self"));
            }
            hir::SelfExplicit(ref typ, _) => {
                try!(word(&mut self.s, "self"));
                try!(self.word_space(":"));
                try!(self.print_type(&**typ));
            }
        }
        return Ok(true);
    }

    pub fn print_fn(&mut self,
                    decl: &hir::FnDecl,
                    unsafety: hir::Unsafety,
                    constness: hir::Constness,
                    abi: abi::Abi,
                    name: Option<ast::Name>,
                    generics: &hir::Generics,
                    opt_explicit_self: Option<&hir::ExplicitSelf_>,
                    vis: hir::Visibility)
                    -> io::Result<()> {
        try!(self.print_fn_header_info(unsafety, constness, abi, vis));

        if let Some(name) = name {
            try!(self.nbsp());
            try!(self.print_name(name));
        }
        try!(self.print_generics(generics));
        try!(self.print_fn_args_and_ret(decl, opt_explicit_self));
        self.print_where_clause(&generics.where_clause)
    }

    pub fn print_fn_args(&mut self,
                         decl: &hir::FnDecl,
                         opt_explicit_self: Option<&hir::ExplicitSelf_>)
                         -> io::Result<()> {
        // It is unfortunate to duplicate the commasep logic, but we want the
        // self type and the args all in the same box.
        try!(self.rbox(0, Inconsistent));
        let mut first = true;
        if let Some(explicit_self) = opt_explicit_self {
            let m = match explicit_self {
                &hir::SelfStatic => hir::MutImmutable,
                _ => match decl.inputs[0].pat.node {
                    hir::PatIdent(hir::BindByValue(m), _, _) => m,
                    _ => hir::MutImmutable,
                },
            };
            first = !try!(self.print_explicit_self(explicit_self, m));
        }

        // HACK(eddyb) ignore the separately printed self argument.
        let args = if first {
            &decl.inputs[..]
        } else {
            &decl.inputs[1..]
        };

        for arg in args {
            if first {
                first = false;
            } else {
                try!(self.word_space(","));
            }
            try!(self.print_arg(arg));
        }

        self.end()
    }

    pub fn print_fn_args_and_ret(&mut self,
                                 decl: &hir::FnDecl,
                                 opt_explicit_self: Option<&hir::ExplicitSelf_>)
                                 -> io::Result<()> {
        try!(self.popen());
        try!(self.print_fn_args(decl, opt_explicit_self));
        if decl.variadic {
            try!(word(&mut self.s, ", ..."));
        }
        try!(self.pclose());

        self.print_fn_output(decl)
    }

    pub fn print_fn_block_args(&mut self, decl: &hir::FnDecl) -> io::Result<()> {
        try!(word(&mut self.s, "|"));
        try!(self.print_fn_args(decl, None));
        try!(word(&mut self.s, "|"));

        if let hir::DefaultReturn(..) = decl.output {
            return Ok(());
        }

        try!(self.space_if_not_bol());
        try!(self.word_space("->"));
        match decl.output {
            hir::Return(ref ty) => {
                try!(self.print_type(&**ty));
                self.maybe_print_comment(ty.span.lo)
            }
            hir::DefaultReturn(..) => unreachable!(),
            hir::NoReturn(span) => {
                try!(self.word_nbsp("!"));
                self.maybe_print_comment(span.lo)
            }
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: hir::CaptureClause) -> io::Result<()> {
        match capture_clause {
            hir::CaptureByValue => self.word_space("move"),
            hir::CaptureByRef => Ok(()),
        }
    }

    pub fn print_bounds(&mut self, prefix: &str, bounds: &[hir::TyParamBound]) -> io::Result<()> {
        if !bounds.is_empty() {
            try!(word(&mut self.s, prefix));
            let mut first = true;
            for bound in bounds {
                try!(self.nbsp());
                if first {
                    first = false;
                } else {
                    try!(self.word_space("+"));
                }

                try!(match *bound {
                    TraitTyParamBound(ref tref, TraitBoundModifier::None) => {
                        self.print_poly_trait_ref(tref)
                    }
                    TraitTyParamBound(ref tref, TraitBoundModifier::Maybe) => {
                        try!(word(&mut self.s, "?"));
                        self.print_poly_trait_ref(tref)
                    }
                    RegionTyParamBound(ref lt) => {
                        self.print_lifetime(lt)
                    }
                })
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn print_lifetime(&mut self, lifetime: &hir::Lifetime) -> io::Result<()> {
        self.print_name(lifetime.name)
    }

    pub fn print_lifetime_def(&mut self, lifetime: &hir::LifetimeDef) -> io::Result<()> {
        try!(self.print_lifetime(&lifetime.lifetime));
        let mut sep = ":";
        for v in &lifetime.bounds {
            try!(word(&mut self.s, sep));
            try!(self.print_lifetime(v));
            sep = "+";
        }
        Ok(())
    }

    pub fn print_generics(&mut self, generics: &hir::Generics) -> io::Result<()> {
        let total = generics.lifetimes.len() + generics.ty_params.len();
        if total == 0 {
            return Ok(());
        }

        try!(word(&mut self.s, "<"));

        let mut ints = Vec::new();
        for i in 0..total {
            ints.push(i);
        }

        try!(self.commasep(Inconsistent,
                           &ints[..],
                           |s, &idx| {
                               if idx < generics.lifetimes.len() {
                                   let lifetime = &generics.lifetimes[idx];
                                   s.print_lifetime_def(lifetime)
                               } else {
                                   let idx = idx - generics.lifetimes.len();
                                   let param = &generics.ty_params[idx];
                                   s.print_ty_param(param)
                               }
                           }));

        try!(word(&mut self.s, ">"));
        Ok(())
    }

    pub fn print_ty_param(&mut self, param: &hir::TyParam) -> io::Result<()> {
        try!(self.print_name(param.name));
        try!(self.print_bounds(":", &param.bounds));
        match param.default {
            Some(ref default) => {
                try!(space(&mut self.s));
                try!(self.word_space("="));
                self.print_type(&**default)
            }
            _ => Ok(()),
        }
    }

    pub fn print_where_clause(&mut self, where_clause: &hir::WhereClause) -> io::Result<()> {
        if where_clause.predicates.is_empty() {
            return Ok(())
        }

        try!(space(&mut self.s));
        try!(self.word_space("where"));

        for (i, predicate) in where_clause.predicates.iter().enumerate() {
            if i != 0 {
                try!(self.word_space(","));
            }

            match predicate {
                &hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate{ref bound_lifetimes,
                                                                              ref bounded_ty,
                                                                              ref bounds,
                                                                              ..}) => {
                    try!(self.print_formal_lifetime_list(bound_lifetimes));
                    try!(self.print_type(&**bounded_ty));
                    try!(self.print_bounds(":", bounds));
                }
                &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                ..}) => {
                    try!(self.print_lifetime(lifetime));
                    try!(word(&mut self.s, ":"));

                    for (i, bound) in bounds.iter().enumerate() {
                        try!(self.print_lifetime(bound));

                        if i != 0 {
                            try!(word(&mut self.s, ":"));
                        }
                    }
                }
                &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate{ref path, ref ty, ..}) => {
                    try!(self.print_path(path, false, 0));
                    try!(space(&mut self.s));
                    try!(self.word_space("="));
                    try!(self.print_type(&**ty));
                }
            }
        }

        Ok(())
    }

    pub fn print_view_path(&mut self, vp: &hir::ViewPath) -> io::Result<()> {
        match vp.node {
            hir::ViewPathSimple(name, ref path) => {
                try!(self.print_path(path, false, 0));

                if path.segments.last().unwrap().identifier.name != name {
                    try!(space(&mut self.s));
                    try!(self.word_space("as"));
                    try!(self.print_name(name));
                }

                Ok(())
            }

            hir::ViewPathGlob(ref path) => {
                try!(self.print_path(path, false, 0));
                word(&mut self.s, "::*")
            }

            hir::ViewPathList(ref path, ref segments) => {
                if path.segments.is_empty() {
                    try!(word(&mut self.s, "{"));
                } else {
                    try!(self.print_path(path, false, 0));
                    try!(word(&mut self.s, "::{"));
                }
                try!(self.commasep(Inconsistent,
                                   &segments[..],
                                   |s, w| {
                                       match w.node {
                                           hir::PathListIdent { name, .. } => {
                                               s.print_name(name)
                                           }
                                           hir::PathListMod { .. } => {
                                               word(&mut s.s, "self")
                                           }
                                       }
                                   }));
                word(&mut self.s, "}")
            }
        }
    }

    pub fn print_mutability(&mut self, mutbl: hir::Mutability) -> io::Result<()> {
        match mutbl {
            hir::MutMutable => self.word_nbsp("mut"),
            hir::MutImmutable => Ok(()),
        }
    }

    pub fn print_mt(&mut self, mt: &hir::MutTy) -> io::Result<()> {
        try!(self.print_mutability(mt.mutbl));
        self.print_type(&*mt.ty)
    }

    pub fn print_arg(&mut self, input: &hir::Arg) -> io::Result<()> {
        try!(self.ibox(indent_unit));
        match input.ty.node {
            hir::TyInfer => try!(self.print_pat(&*input.pat)),
            _ => {
                match input.pat.node {
                    hir::PatIdent(_, ref path1, _) if
                        path1.node.name ==
                            parse::token::special_idents::invalid.name => {
                        // Do nothing.
                    }
                    _ => {
                        try!(self.print_pat(&*input.pat));
                        try!(word(&mut self.s, ":"));
                        try!(space(&mut self.s));
                    }
                }
                try!(self.print_type(&*input.ty));
            }
        }
        self.end()
    }

    pub fn print_fn_output(&mut self, decl: &hir::FnDecl) -> io::Result<()> {
        if let hir::DefaultReturn(..) = decl.output {
            return Ok(());
        }

        try!(self.space_if_not_bol());
        try!(self.ibox(indent_unit));
        try!(self.word_space("->"));
        match decl.output {
            hir::NoReturn(_) => try!(self.word_nbsp("!")),
            hir::DefaultReturn(..) => unreachable!(),
            hir::Return(ref ty) => try!(self.print_type(&**ty)),
        }
        try!(self.end());

        match decl.output {
            hir::Return(ref output) => self.maybe_print_comment(output.span.lo),
            _ => Ok(()),
        }
    }

    pub fn print_ty_fn(&mut self,
                       abi: abi::Abi,
                       unsafety: hir::Unsafety,
                       decl: &hir::FnDecl,
                       name: Option<ast::Name>,
                       generics: &hir::Generics,
                       opt_explicit_self: Option<&hir::ExplicitSelf_>)
                       -> io::Result<()> {
        try!(self.ibox(indent_unit));
        if !generics.lifetimes.is_empty() || !generics.ty_params.is_empty() {
            try!(word(&mut self.s, "for"));
            try!(self.print_generics(generics));
        }
        let generics = hir::Generics {
            lifetimes: Vec::new(),
            ty_params: OwnedSlice::empty(),
            where_clause: hir::WhereClause {
                id: ast::DUMMY_NODE_ID,
                predicates: Vec::new(),
            },
        };
        try!(self.print_fn(decl,
                           unsafety,
                           hir::Constness::NotConst,
                           abi,
                           name,
                           &generics,
                           opt_explicit_self,
                           hir::Inherited));
        self.end()
    }

    pub fn maybe_print_trailing_comment(&mut self,
                                        span: codemap::Span,
                                        next_pos: Option<BytePos>)
                                        -> io::Result<()> {
        let cm = match self.cm {
            Some(cm) => cm,
            _ => return Ok(()),
        };
        match self.next_comment() {
            Some(ref cmnt) => {
                if (*cmnt).style != comments::Trailing {
                    return Ok(())
                }
                let span_line = cm.lookup_char_pos(span.hi);
                let comment_line = cm.lookup_char_pos((*cmnt).pos);
                let mut next = (*cmnt).pos + BytePos(1);
                match next_pos {
                    None => (),
                    Some(p) => next = p,
                }
                if span.hi < (*cmnt).pos && (*cmnt).pos < next &&
                   span_line.line == comment_line.line {
                    try!(self.print_comment(cmnt));
                    self.cur_cmnt_and_lit.cur_cmnt += 1;
                }
            }
            _ => (),
        }
        Ok(())
    }

    pub fn print_remaining_comments(&mut self) -> io::Result<()> {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.next_comment().is_none() {
            try!(hardbreak(&mut self.s));
        }
        loop {
            match self.next_comment() {
                Some(ref cmnt) => {
                    try!(self.print_comment(cmnt));
                    self.cur_cmnt_and_lit.cur_cmnt += 1;
                }
                _ => break,
            }
        }
        Ok(())
    }

    pub fn print_opt_abi_and_extern_if_nondefault(&mut self,
                                                  opt_abi: Option<abi::Abi>)
                                                  -> io::Result<()> {
        match opt_abi {
            Some(abi::Rust) => Ok(()),
            Some(abi) => {
                try!(self.word_nbsp("extern"));
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(()),
        }
    }

    pub fn print_extern_opt_abi(&mut self, opt_abi: Option<abi::Abi>) -> io::Result<()> {
        match opt_abi {
            Some(abi) => {
                try!(self.word_nbsp("extern"));
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(()),
        }
    }

    pub fn print_fn_header_info(&mut self,
                                unsafety: hir::Unsafety,
                                constness: hir::Constness,
                                abi: abi::Abi,
                                vis: hir::Visibility)
                                -> io::Result<()> {
        try!(word(&mut self.s, &visibility_qualified(vis, "")));
        try!(self.print_unsafety(unsafety));

        match constness {
            hir::Constness::NotConst => {}
            hir::Constness::Const => try!(self.word_nbsp("const")),
        }

        if abi != abi::Rust {
            try!(self.word_nbsp("extern"));
            try!(self.word_nbsp(&abi.to_string()));
        }

        word(&mut self.s, "fn")
    }

    pub fn print_unsafety(&mut self, s: hir::Unsafety) -> io::Result<()> {
        match s {
            hir::Unsafety::Normal => Ok(()),
            hir::Unsafety::Unsafe => self.word_nbsp("unsafe"),
        }
    }
}

// Dup'ed from parse::classify, but adapted for the HIR.
/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
fn expr_requires_semi_to_be_stmt(e: &hir::Expr) -> bool {
    match e.node {
        hir::ExprIf(..) |
        hir::ExprMatch(..) |
        hir::ExprBlock(_) |
        hir::ExprWhile(..) |
        hir::ExprLoop(..) => false,
        _ => true,
    }
}

/// this statement requires a semicolon after it.
/// note that in one case (stmt_semi), we've already
/// seen the semicolon, and thus don't need another.
fn stmt_ends_with_semi(stmt: &hir::Stmt_) -> bool {
    match *stmt {
        hir::StmtDecl(ref d, _) => {
            match d.node {
                hir::DeclLocal(_) => true,
                hir::DeclItem(_) => false,
            }
        }
        hir::StmtExpr(ref e, _) => {
            expr_requires_semi_to_be_stmt(&**e)
        }
        hir::StmtSemi(..) => {
            false
        }
    }
}
