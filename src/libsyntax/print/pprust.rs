// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::AbiSet;
use ast::{P, RegionTyParamBound, TraitTyParamBound, Required, Provided};
use ast;
use ast_util;
use opt_vec::OptVec;
use opt_vec;
use attr::{AttrMetaMethods, AttributeMethods};
use codemap::{CodeMap, BytePos};
use codemap;
use diagnostic;
use parse::classify::expr_is_simple_block;
use parse::token::IdentInterner;
use parse::{comments, token};
use parse;
use print::pp::{break_offset, word, space, zerobreak, hardbreak};
use print::pp::{Breaks, Consistent, Inconsistent, eof};
use print::pp;
use print::pprust;

use std::cast;
use std::cell::RefCell;
use std::char;
use std::str;
use std::io;
use std::io::MemWriter;
use std::vec_ng::Vec;

// The &mut State is stored here to prevent recursive type.
pub enum AnnNode<'a, 'b> {
    NodeBlock(&'a mut State<'a>, &'b ast::Block),
    NodeItem(&'a mut State<'a>, &'b ast::Item),
    NodeExpr(&'a mut State<'a>, &'b ast::Expr),
    NodePat(&'a mut State<'a>, &'b ast::Pat),
}

pub trait PpAnn {
    fn pre(&self, _node: AnnNode) -> io::IoResult<()> { Ok(()) }
    fn post(&self, _node: AnnNode) -> io::IoResult<()> { Ok(()) }
}

pub struct NoAnn;

impl PpAnn for NoAnn {}

pub struct CurrentCommentAndLiteral {
    cur_cmnt: uint,
    cur_lit: uint,
}

pub struct State<'a> {
    s: pp::Printer,
    cm: Option<@CodeMap>,
    intr: @token::IdentInterner,
    comments: Option<Vec<comments::Comment> >,
    literals: Option<Vec<comments::Literal> >,
    cur_cmnt_and_lit: CurrentCommentAndLiteral,
    boxes: RefCell<Vec<pp::Breaks> >,
    ann: &'a PpAnn
}

pub fn ibox(s: &mut State, u: uint) -> io::IoResult<()> {
    {
        let mut boxes = s.boxes.borrow_mut();
        boxes.get().push(pp::Inconsistent);
    }
    pp::ibox(&mut s.s, u)
}

pub fn end(s: &mut State) -> io::IoResult<()> {
    {
        let mut boxes = s.boxes.borrow_mut();
        boxes.get().pop().unwrap();
    }
    pp::end(&mut s.s)
}

pub fn rust_printer(writer: ~io::Writer) -> State<'static> {
    rust_printer_annotated(writer, &NoAnn)
}

pub fn rust_printer_annotated<'a>(writer: ~io::Writer, ann: &'a PpAnn) -> State<'a> {
    State {
        s: pp::mk_printer(writer, default_columns),
        cm: None,
        intr: token::get_ident_interner(),
        comments: None,
        literals: None,
        cur_cmnt_and_lit: CurrentCommentAndLiteral {
            cur_cmnt: 0,
            cur_lit: 0
        },
        boxes: RefCell::new(Vec::new()),
        ann: ann
    }
}

pub static indent_unit: uint = 4u;

pub static default_columns: uint = 78u;

// Requires you to pass an input filename and reader so that
// it can scan the input text for comments and literals to
// copy forward.
pub fn print_crate(cm: @CodeMap,
                   span_diagnostic: @diagnostic::SpanHandler,
                   krate: &ast::Crate,
                   filename: ~str,
                   input: &mut io::Reader,
                   out: ~io::Writer,
                   ann: &PpAnn,
                   is_expanded: bool) -> io::IoResult<()> {
    let (cmnts, lits) = comments::gather_comments_and_literals(
        span_diagnostic,
        filename,
        input
    );
    let mut s = State {
        s: pp::mk_printer(out, default_columns),
        cm: Some(cm),
        intr: token::get_ident_interner(),
        comments: Some(cmnts),
        // If the code is post expansion, don't use the table of
        // literals, since it doesn't correspond with the literals
        // in the AST anymore.
        literals: if is_expanded {
            None
        } else {
            Some(lits)
        },
        cur_cmnt_and_lit: CurrentCommentAndLiteral {
            cur_cmnt: 0,
            cur_lit: 0
        },
        boxes: RefCell::new(Vec::new()),
        ann: ann
    };
    print_crate_(&mut s, krate)
}

pub fn print_crate_(s: &mut State, krate: &ast::Crate) -> io::IoResult<()> {
    try!(print_mod(s, &krate.module, krate.attrs.as_slice()));
    try!(print_remaining_comments(s));
    try!(eof(&mut s.s));
    Ok(())
}

pub fn ty_to_str(ty: &ast::Ty) -> ~str {
    to_str(ty, print_type)
}

pub fn pat_to_str(pat: &ast::Pat) -> ~str {
    to_str(pat, print_pat)
}

pub fn expr_to_str(e: &ast::Expr) -> ~str {
    to_str(e, print_expr)
}

pub fn lifetime_to_str(e: &ast::Lifetime) -> ~str {
    to_str(e, print_lifetime)
}

pub fn tt_to_str(tt: &ast::TokenTree) -> ~str {
    to_str(tt, print_tt)
}

pub fn tts_to_str(tts: &[ast::TokenTree]) -> ~str {
    to_str(&tts, print_tts)
}

pub fn stmt_to_str(s: &ast::Stmt) -> ~str {
    to_str(s, print_stmt)
}

pub fn item_to_str(i: &ast::Item) -> ~str {
    to_str(i, print_item)
}

pub fn generics_to_str(generics: &ast::Generics) -> ~str {
    to_str(generics, print_generics)
}

pub fn path_to_str(p: &ast::Path) -> ~str {
    to_str(p, |a,b| print_path(a, b, false))
}

pub fn fun_to_str(decl: &ast::FnDecl, purity: ast::Purity, name: ast::Ident,
                  opt_explicit_self: Option<ast::ExplicitSelf_>,
                  generics: &ast::Generics) -> ~str {
    let wr = ~MemWriter::new();
    let mut s = rust_printer(wr as ~io::Writer);
    print_fn(&mut s, decl, Some(purity), AbiSet::Rust(),
             name, generics, opt_explicit_self, ast::Inherited).unwrap();
    end(&mut s).unwrap(); // Close the head box
    end(&mut s).unwrap(); // Close the outer box
    eof(&mut s.s).unwrap();
    unsafe {
        get_mem_writer(&mut s.s.out)
    }
}

pub fn block_to_str(blk: &ast::Block) -> ~str {
    let wr = ~MemWriter::new();
    let mut s = rust_printer(wr as ~io::Writer);
    // containing cbox, will be closed by print-block at }
    cbox(&mut s, indent_unit).unwrap();
    // head-ibox, will be closed by print-block after {
    ibox(&mut s, 0u).unwrap();
    print_block(&mut s, blk).unwrap();
    eof(&mut s.s).unwrap();
    unsafe {
        get_mem_writer(&mut s.s.out)
    }
}

pub fn meta_item_to_str(mi: &ast::MetaItem) -> ~str {
    to_str(mi, print_meta_item)
}

pub fn attribute_to_str(attr: &ast::Attribute) -> ~str {
    to_str(attr, print_attribute)
}

pub fn variant_to_str(var: &ast::Variant) -> ~str {
    to_str(var, print_variant)
}

pub fn cbox(s: &mut State, u: uint) -> io::IoResult<()> {
    {
        let mut boxes = s.boxes.borrow_mut();
        boxes.get().push(pp::Consistent);
    }
    pp::cbox(&mut s.s, u)
}

// "raw box"
pub fn rbox(s: &mut State, u: uint, b: pp::Breaks) -> io::IoResult<()> {
    {
        let mut boxes = s.boxes.borrow_mut();
        boxes.get().push(b);
    }
    pp::rbox(&mut s.s, u, b)
}

pub fn nbsp(s: &mut State) -> io::IoResult<()> { word(&mut s.s, " ") }

pub fn word_nbsp(s: &mut State, w: &str) -> io::IoResult<()> {
    try!(word(&mut s.s, w));
    nbsp(s)
}

pub fn word_space(s: &mut State, w: &str) -> io::IoResult<()> {
    try!(word(&mut s.s, w));
    space(&mut s.s)
}

pub fn popen(s: &mut State) -> io::IoResult<()> { word(&mut s.s, "(") }

pub fn pclose(s: &mut State) -> io::IoResult<()> { word(&mut s.s, ")") }

pub fn head(s: &mut State, w: &str) -> io::IoResult<()> {
    // outer-box is consistent
    try!(cbox(s, indent_unit));
    // head-box is inconsistent
    try!(ibox(s, w.len() + 1));
    // keyword that starts the head
    if !w.is_empty() {
        try!(word_nbsp(s, w));
    }
    Ok(())
}

pub fn bopen(s: &mut State) -> io::IoResult<()> {
    try!(word(&mut s.s, "{"));
    try!(end(s)); // close the head-box
    Ok(())
}

pub fn bclose_(s: &mut State, span: codemap::Span,
               indented: uint) -> io::IoResult<()> {
    bclose_maybe_open(s, span, indented, true)
}
pub fn bclose_maybe_open (s: &mut State, span: codemap::Span,
                          indented: uint, close_box: bool) -> io::IoResult<()> {
    try!(maybe_print_comment(s, span.hi));
    try!(break_offset_if_not_bol(s, 1u, -(indented as int)));
    try!(word(&mut s.s, "}"));
    if close_box {
        try!(end(s)); // close the outer-box
    }
    Ok(())
}
pub fn bclose(s: &mut State, span: codemap::Span) -> io::IoResult<()> {
    bclose_(s, span, indent_unit)
}

pub fn is_begin(s: &mut State) -> bool {
    match s.s.last_token() { pp::Begin(_) => true, _ => false }
}

pub fn is_end(s: &mut State) -> bool {
    match s.s.last_token() { pp::End => true, _ => false }
}

pub fn is_bol(s: &mut State) -> bool {
    return s.s.last_token().is_eof() || s.s.last_token().is_hardbreak_tok();
}

pub fn in_cbox(s: &mut State) -> bool {
    let boxes = s.boxes.borrow();
    let len = boxes.get().len();
    if len == 0u { return false; }
    return *boxes.get().get(len - 1u) == pp::Consistent;
}

pub fn hardbreak_if_not_bol(s: &mut State) -> io::IoResult<()> {
    if !is_bol(s) {
        try!(hardbreak(&mut s.s))
    }
    Ok(())
}
pub fn space_if_not_bol(s: &mut State) -> io::IoResult<()> {
    if !is_bol(s) { try!(space(&mut s.s)); }
    Ok(())
}
pub fn break_offset_if_not_bol(s: &mut State, n: uint,
                               off: int) -> io::IoResult<()> {
    if !is_bol(s) {
        try!(break_offset(&mut s.s, n, off));
    } else {
        if off != 0 && s.s.last_token().is_hardbreak_tok() {
            // We do something pretty sketchy here: tuck the nonzero
            // offset-adjustment we were going to deposit along with the
            // break into the previous hardbreak.
            s.s.replace_last_token(pp::hardbreak_tok_offset(off));
        }
    }
    Ok(())
}

// Synthesizes a comment that was not textually present in the original source
// file.
pub fn synth_comment(s: &mut State, text: ~str) -> io::IoResult<()> {
    try!(word(&mut s.s, "/*"));
    try!(space(&mut s.s));
    try!(word(&mut s.s, text));
    try!(space(&mut s.s));
    try!(word(&mut s.s, "*/"));
    Ok(())
}

pub fn commasep<T>(s: &mut State, b: Breaks, elts: &[T],
                   op: |&mut State, &T| -> io::IoResult<()>)
    -> io::IoResult<()>
{
    try!(rbox(s, 0u, b));
    let mut first = true;
    for elt in elts.iter() {
        if first { first = false; } else { try!(word_space(s, ",")); }
        try!(op(s, elt));
    }
    end(s)
}


pub fn commasep_cmnt<T>(
                     s: &mut State,
                     b: Breaks,
                     elts: &[T],
                     op: |&mut State, &T| -> io::IoResult<()>,
                     get_span: |&T| -> codemap::Span) -> io::IoResult<()> {
    try!(rbox(s, 0u, b));
    let len = elts.len();
    let mut i = 0u;
    for elt in elts.iter() {
        try!(maybe_print_comment(s, get_span(elt).hi));
        try!(op(s, elt));
        i += 1u;
        if i < len {
            try!(word(&mut s.s, ","));
            try!(maybe_print_trailing_comment(s, get_span(elt),
                                                Some(get_span(&elts[i]).hi)));
            try!(space_if_not_bol(s));
        }
    }
    end(s)
}

pub fn commasep_exprs(s: &mut State, b: Breaks,
                      exprs: &[@ast::Expr]) -> io::IoResult<()> {
    commasep_cmnt(s, b, exprs, |p, &e| print_expr(p, e), |e| e.span)
}

pub fn print_mod(s: &mut State, _mod: &ast::Mod,
                 attrs: &[ast::Attribute]) -> io::IoResult<()> {
    try!(print_inner_attributes(s, attrs));
    for vitem in _mod.view_items.iter() {
        try!(print_view_item(s, vitem));
    }
    for item in _mod.items.iter() {
        try!(print_item(s, *item));
    }
    Ok(())
}

pub fn print_foreign_mod(s: &mut State, nmod: &ast::ForeignMod,
                         attrs: &[ast::Attribute]) -> io::IoResult<()> {
    try!(print_inner_attributes(s, attrs));
    for vitem in nmod.view_items.iter() {
        try!(print_view_item(s, vitem));
    }
    for item in nmod.items.iter() {
        try!(print_foreign_item(s, *item));
    }
    Ok(())
}

pub fn print_opt_lifetime(s: &mut State,
                          lifetime: &Option<ast::Lifetime>) -> io::IoResult<()> {
    for l in lifetime.iter() {
        try!(print_lifetime(s, l));
        try!(nbsp(s));
    }
    Ok(())
}

pub fn print_type(s: &mut State, ty: &ast::Ty) -> io::IoResult<()> {
    try!(maybe_print_comment(s, ty.span.lo));
    try!(ibox(s, 0u));
    match ty.node {
        ast::TyNil => try!(word(&mut s.s, "()")),
        ast::TyBot => try!(word(&mut s.s, "!")),
        ast::TyBox(ty) => {
            try!(word(&mut s.s, "@"));
            try!(print_type(s, ty));
        }
        ast::TyUniq(ty) => {
            try!(word(&mut s.s, "~"));
            try!(print_type(s, ty));
        }
        ast::TyVec(ty) => {
            try!(word(&mut s.s, "["));
            try!(print_type(s, ty));
            try!(word(&mut s.s, "]"));
        }
        ast::TyPtr(ref mt) => {
            try!(word(&mut s.s, "*"));
            try!(print_mt(s, mt));
        }
        ast::TyRptr(ref lifetime, ref mt) => {
            try!(word(&mut s.s, "&"));
            try!(print_opt_lifetime(s, lifetime));
            try!(print_mt(s, mt));
        }
        ast::TyTup(ref elts) => {
            try!(popen(s));
            try!(commasep(s, Inconsistent, elts.as_slice(), print_type_ref));
            if elts.len() == 1 {
                try!(word(&mut s.s, ","));
            }
            try!(pclose(s));
        }
        ast::TyBareFn(f) => {
            let generics = ast::Generics {
                lifetimes: f.lifetimes.clone(),
                ty_params: opt_vec::Empty
            };
            try!(print_ty_fn(s, Some(f.abis), None, &None,
                               f.purity, ast::Many, f.decl, None, &None,
                               Some(&generics), None));
        }
        ast::TyClosure(f) => {
            let generics = ast::Generics {
                lifetimes: f.lifetimes.clone(),
                ty_params: opt_vec::Empty
            };
            try!(print_ty_fn(s, None, Some(f.sigil), &f.region,
                               f.purity, f.onceness, f.decl, None, &f.bounds,
                               Some(&generics), None));
        }
        ast::TyPath(ref path, ref bounds, _) => {
            try!(print_bounded_path(s, path, bounds));
        }
        ast::TyFixedLengthVec(ty, v) => {
            try!(word(&mut s.s, "["));
            try!(print_type(s, ty));
            try!(word(&mut s.s, ", .."));
            try!(print_expr(s, v));
            try!(word(&mut s.s, "]"));
        }
        ast::TyTypeof(e) => {
            try!(word(&mut s.s, "typeof("));
            try!(print_expr(s, e));
            try!(word(&mut s.s, ")"));
        }
        ast::TyInfer => {
            try!(word(&mut s.s, "_"));
        }
    }
    end(s)
}

pub fn print_type_ref(s: &mut State, ty: &P<ast::Ty>) -> io::IoResult<()> {
    print_type(s, *ty)
}

pub fn print_foreign_item(s: &mut State,
                          item: &ast::ForeignItem) -> io::IoResult<()> {
    try!(hardbreak_if_not_bol(s));
    try!(maybe_print_comment(s, item.span.lo));
    try!(print_outer_attributes(s, item.attrs.as_slice()));
    match item.node {
        ast::ForeignItemFn(decl, ref generics) => {
            try!(print_fn(s, decl, None, AbiSet::Rust(), item.ident, generics,
            None, item.vis));
            try!(end(s)); // end head-ibox
            try!(word(&mut s.s, ";"));
            try!(end(s)); // end the outer fn box
        }
        ast::ForeignItemStatic(t, m) => {
            try!(head(s, visibility_qualified(item.vis, "static")));
            if m {
                try!(word_space(s, "mut"));
            }
            try!(print_ident(s, item.ident));
            try!(word_space(s, ":"));
            try!(print_type(s, t));
            try!(word(&mut s.s, ";"));
            try!(end(s)); // end the head-ibox
            try!(end(s)); // end the outer cbox
        }
    }
    Ok(())
}

pub fn print_item(s: &mut State, item: &ast::Item) -> io::IoResult<()> {
    try!(hardbreak_if_not_bol(s));
    try!(maybe_print_comment(s, item.span.lo));
    try!(print_outer_attributes(s, item.attrs.as_slice()));
    {
        let ann_node = NodeItem(s, item);
        try!(s.ann.pre(ann_node));
    }
    match item.node {
      ast::ItemStatic(ty, m, expr) => {
        try!(head(s, visibility_qualified(item.vis, "static")));
        if m == ast::MutMutable {
            try!(word_space(s, "mut"));
        }
        try!(print_ident(s, item.ident));
        try!(word_space(s, ":"));
        try!(print_type(s, ty));
        try!(space(&mut s.s));
        try!(end(s)); // end the head-ibox

        try!(word_space(s, "="));
        try!(print_expr(s, expr));
        try!(word(&mut s.s, ";"));
        try!(end(s)); // end the outer cbox

      }
      ast::ItemFn(decl, purity, abi, ref typarams, body) => {
        try!(print_fn(
            s,
            decl,
            Some(purity),
            abi,
            item.ident,
            typarams,
            None,
            item.vis
        ));
        try!(word(&mut s.s, " "));
        try!(print_block_with_attrs(s, body, item.attrs.as_slice()));
      }
      ast::ItemMod(ref _mod) => {
        try!(head(s, visibility_qualified(item.vis, "mod")));
        try!(print_ident(s, item.ident));
        try!(nbsp(s));
        try!(bopen(s));
        try!(print_mod(s, _mod, item.attrs.as_slice()));
        try!(bclose(s, item.span));
      }
      ast::ItemForeignMod(ref nmod) => {
        try!(head(s, "extern"));
        try!(word_nbsp(s, nmod.abis.to_str()));
        try!(bopen(s));
        try!(print_foreign_mod(s, nmod, item.attrs.as_slice()));
        try!(bclose(s, item.span));
      }
      ast::ItemTy(ty, ref params) => {
        try!(ibox(s, indent_unit));
        try!(ibox(s, 0u));
        try!(word_nbsp(s, visibility_qualified(item.vis, "type")));
        try!(print_ident(s, item.ident));
        try!(print_generics(s, params));
        try!(end(s)); // end the inner ibox

        try!(space(&mut s.s));
        try!(word_space(s, "="));
        try!(print_type(s, ty));
        try!(word(&mut s.s, ";"));
        try!(end(s)); // end the outer ibox
      }
      ast::ItemEnum(ref enum_definition, ref params) => {
        try!(print_enum_def(
            s,
            enum_definition,
            params,
            item.ident,
            item.span,
            item.vis
        ));
      }
      ast::ItemStruct(struct_def, ref generics) => {
          try!(head(s, visibility_qualified(item.vis, "struct")));
          try!(print_struct(s, struct_def, generics, item.ident, item.span));
      }

      ast::ItemImpl(ref generics, ref opt_trait, ty, ref methods) => {
        try!(head(s, visibility_qualified(item.vis, "impl")));
        if generics.is_parameterized() {
            try!(print_generics(s, generics));
            try!(space(&mut s.s));
        }

        match opt_trait {
            &Some(ref t) => {
                try!(print_trait_ref(s, t));
                try!(space(&mut s.s));
                try!(word_space(s, "for"));
            }
            &None => ()
        };

        try!(print_type(s, ty));

        try!(space(&mut s.s));
        try!(bopen(s));
        try!(print_inner_attributes(s, item.attrs.as_slice()));
        for meth in methods.iter() {
           try!(print_method(s, *meth));
        }
        try!(bclose(s, item.span));
      }
      ast::ItemTrait(ref generics, ref traits, ref methods) => {
        try!(head(s, visibility_qualified(item.vis, "trait")));
        try!(print_ident(s, item.ident));
        try!(print_generics(s, generics));
        if traits.len() != 0u {
            try!(word(&mut s.s, ":"));
            for (i, trait_) in traits.iter().enumerate() {
                try!(nbsp(s));
                if i != 0 {
                    try!(word_space(s, "+"));
                }
                try!(print_path(s, &trait_.path, false));
            }
        }
        try!(word(&mut s.s, " "));
        try!(bopen(s));
        for meth in methods.iter() {
            try!(print_trait_method(s, meth));
        }
        try!(bclose(s, item.span));
      }
      // I think it's reasonable to hide the context here:
      ast::ItemMac(codemap::Spanned { node: ast::MacInvocTT(ref pth, ref tts, _),
                                   ..}) => {
        try!(print_visibility(s, item.vis));
        try!(print_path(s, pth, false));
        try!(word(&mut s.s, "! "));
        try!(print_ident(s, item.ident));
        try!(cbox(s, indent_unit));
        try!(popen(s));
        try!(print_tts(s, &(tts.as_slice())));
        try!(pclose(s));
        try!(end(s));
      }
    }
    {
        let ann_node = NodeItem(s, item);
        try!(s.ann.post(ann_node));
    }
    Ok(())
}

fn print_trait_ref(s: &mut State, t: &ast::TraitRef) -> io::IoResult<()> {
    print_path(s, &t.path, false)
}

pub fn print_enum_def(s: &mut State, enum_definition: &ast::EnumDef,
                      generics: &ast::Generics, ident: ast::Ident,
                      span: codemap::Span,
                      visibility: ast::Visibility) -> io::IoResult<()> {
    try!(head(s, visibility_qualified(visibility, "enum")));
    try!(print_ident(s, ident));
    try!(print_generics(s, generics));
    try!(space(&mut s.s));
    try!(print_variants(s, enum_definition.variants.as_slice(), span));
    Ok(())
}

pub fn print_variants(s: &mut State,
                      variants: &[P<ast::Variant>],
                      span: codemap::Span) -> io::IoResult<()> {
    try!(bopen(s));
    for &v in variants.iter() {
        try!(space_if_not_bol(s));
        try!(maybe_print_comment(s, v.span.lo));
        try!(print_outer_attributes(s, v.node.attrs.as_slice()));
        try!(ibox(s, indent_unit));
        try!(print_variant(s, v));
        try!(word(&mut s.s, ","));
        try!(end(s));
        try!(maybe_print_trailing_comment(s, v.span, None));
    }
    bclose(s, span)
}

pub fn visibility_to_str(vis: ast::Visibility) -> ~str {
    match vis {
        ast::Private => ~"priv",
        ast::Public => ~"pub",
        ast::Inherited => ~""
    }
}

pub fn visibility_qualified(vis: ast::Visibility, s: &str) -> ~str {
    match vis {
        ast::Private | ast::Public => visibility_to_str(vis) + " " + s,
        ast::Inherited => s.to_owned()
    }
}

pub fn print_visibility(s: &mut State, vis: ast::Visibility) -> io::IoResult<()> {
    match vis {
        ast::Private | ast::Public =>
            try!(word_nbsp(s, visibility_to_str(vis))),
        ast::Inherited => ()
    }
    Ok(())
}

pub fn print_struct(s: &mut State,
                    struct_def: &ast::StructDef,
                    generics: &ast::Generics,
                    ident: ast::Ident,
                    span: codemap::Span) -> io::IoResult<()> {
    try!(print_ident(s, ident));
    try!(print_generics(s, generics));
    if ast_util::struct_def_is_tuple_like(struct_def) {
        if !struct_def.fields.is_empty() {
            try!(popen(s));
            try!(commasep(s,
                          Inconsistent,
                          struct_def.fields.as_slice(),
                          |s, field| {
                match field.node.kind {
                    ast::NamedField(..) => fail!("unexpected named field"),
                    ast::UnnamedField => {
                        try!(maybe_print_comment(s, field.span.lo));
                        try!(print_type(s, field.node.ty));
                    }
                }
                Ok(())
            }));
            try!(pclose(s));
        }
        try!(word(&mut s.s, ";"));
        try!(end(s));
        end(s) // close the outer-box
    } else {
        try!(nbsp(s));
        try!(bopen(s));
        try!(hardbreak_if_not_bol(s));

        for field in struct_def.fields.iter() {
            match field.node.kind {
                ast::UnnamedField => fail!("unexpected unnamed field"),
                ast::NamedField(ident, visibility) => {
                    try!(hardbreak_if_not_bol(s));
                    try!(maybe_print_comment(s, field.span.lo));
                    try!(print_outer_attributes(s,
                                                field.node.attrs.as_slice()));
                    try!(print_visibility(s, visibility));
                    try!(print_ident(s, ident));
                    try!(word_nbsp(s, ":"));
                    try!(print_type(s, field.node.ty));
                    try!(word(&mut s.s, ","));
                }
            }
        }

        bclose(s, span)
    }
}

/// This doesn't deserve to be called "pretty" printing, but it should be
/// meaning-preserving. A quick hack that might help would be to look at the
/// spans embedded in the TTs to decide where to put spaces and newlines.
/// But it'd be better to parse these according to the grammar of the
/// appropriate macro, transcribe back into the grammar we just parsed from,
/// and then pretty-print the resulting AST nodes (so, e.g., we print
/// expression arguments as expressions). It can be done! I think.
pub fn print_tt(s: &mut State, tt: &ast::TokenTree) -> io::IoResult<()> {
    match *tt {
        ast::TTDelim(ref tts) => print_tts(s, &(tts.as_slice())),
        ast::TTTok(_, ref tk) => {
            word(&mut s.s, parse::token::to_str(tk))
        }
        ast::TTSeq(_, ref tts, ref sep, zerok) => {
            try!(word(&mut s.s, "$("));
            for tt_elt in (*tts).iter() {
                try!(print_tt(s, tt_elt));
            }
            try!(word(&mut s.s, ")"));
            match *sep {
                Some(ref tk) => {
                    try!(word(&mut s.s, parse::token::to_str(tk)));
                }
                None => ()
            }
            word(&mut s.s, if zerok { "*" } else { "+" })
        }
        ast::TTNonterminal(_, name) => {
            try!(word(&mut s.s, "$"));
            print_ident(s, name)
        }
    }
}

pub fn print_tts(s: &mut State, tts: & &[ast::TokenTree]) -> io::IoResult<()> {
    try!(ibox(s, 0));
    for (i, tt) in tts.iter().enumerate() {
        if i != 0 {
            try!(space(&mut s.s));
        }
        try!(print_tt(s, tt));
    }
    end(s)
}

pub fn print_variant(s: &mut State, v: &ast::Variant) -> io::IoResult<()> {
    try!(print_visibility(s, v.node.vis));
    match v.node.kind {
        ast::TupleVariantKind(ref args) => {
            try!(print_ident(s, v.node.name));
            if !args.is_empty() {
                try!(popen(s));
                fn print_variant_arg(s: &mut State,
                                     arg: &ast::VariantArg) -> io::IoResult<()> {
                    print_type(s, arg.ty)
                }
                try!(commasep(s,
                              Consistent,
                              args.as_slice(),
                              print_variant_arg));
                try!(pclose(s));
            }
        }
        ast::StructVariantKind(struct_def) => {
            try!(head(s, ""));
            let generics = ast_util::empty_generics();
            try!(print_struct(s, struct_def, &generics, v.node.name, v.span));
        }
    }
    match v.node.disr_expr {
      Some(d) => {
        try!(space(&mut s.s));
        try!(word_space(s, "="));
        try!(print_expr(s, d));
      }
      _ => ()
    }
    Ok(())
}

pub fn print_ty_method(s: &mut State, m: &ast::TypeMethod) -> io::IoResult<()> {
    try!(hardbreak_if_not_bol(s));
    try!(maybe_print_comment(s, m.span.lo));
    try!(print_outer_attributes(s, m.attrs.as_slice()));
    try!(print_ty_fn(s,
                       None,
                       None,
                       &None,
                       m.purity,
                       ast::Many,
                       m.decl,
                       Some(m.ident),
                       &None,
                       Some(&m.generics),
                       Some(m.explicit_self.node)));
    word(&mut s.s, ";")
}

pub fn print_trait_method(s: &mut State,
                          m: &ast::TraitMethod) -> io::IoResult<()> {
    match *m {
        Required(ref ty_m) => print_ty_method(s, ty_m),
        Provided(m) => print_method(s, m)
    }
}

pub fn print_method(s: &mut State, meth: &ast::Method) -> io::IoResult<()> {
    try!(hardbreak_if_not_bol(s));
    try!(maybe_print_comment(s, meth.span.lo));
    try!(print_outer_attributes(s, meth.attrs.as_slice()));
    try!(print_fn(s, meth.decl, Some(meth.purity), AbiSet::Rust(),
                    meth.ident, &meth.generics, Some(meth.explicit_self.node),
                    meth.vis));
    try!(word(&mut s.s, " "));
    print_block_with_attrs(s, meth.body, meth.attrs.as_slice())
}

pub fn print_outer_attributes(s: &mut State,
                              attrs: &[ast::Attribute]) -> io::IoResult<()> {
    let mut count = 0;
    for attr in attrs.iter() {
        match attr.node.style {
          ast::AttrOuter => {
              try!(print_attribute(s, attr));
              count += 1;
          }
          _ => {/* fallthrough */ }
        }
    }
    if count > 0 {
        try!(hardbreak_if_not_bol(s));
    }
    Ok(())
}

pub fn print_inner_attributes(s: &mut State,
                              attrs: &[ast::Attribute]) -> io::IoResult<()> {
    let mut count = 0;
    for attr in attrs.iter() {
        match attr.node.style {
          ast::AttrInner => {
            try!(print_attribute(s, attr));
            if !attr.node.is_sugared_doc {
                try!(word(&mut s.s, ";"));
            }
            count += 1;
          }
          _ => {/* fallthrough */ }
        }
    }
    if count > 0 {
        try!(hardbreak_if_not_bol(s));
    }
    Ok(())
}

pub fn print_attribute(s: &mut State, attr: &ast::Attribute) -> io::IoResult<()> {
    try!(hardbreak_if_not_bol(s));
    try!(maybe_print_comment(s, attr.span.lo));
    if attr.node.is_sugared_doc {
        let comment = attr.value_str().unwrap();
        try!(word(&mut s.s, comment.get()));
    } else {
        try!(word(&mut s.s, "#["));
        try!(print_meta_item(s, attr.meta()));
        try!(word(&mut s.s, "]"));
    }
    Ok(())
}


pub fn print_stmt(s: &mut State, st: &ast::Stmt) -> io::IoResult<()> {
    try!(maybe_print_comment(s, st.span.lo));
    match st.node {
      ast::StmtDecl(decl, _) => {
        try!(print_decl(s, decl));
      }
      ast::StmtExpr(expr, _) => {
        try!(space_if_not_bol(s));
        try!(print_expr(s, expr));
      }
      ast::StmtSemi(expr, _) => {
        try!(space_if_not_bol(s));
        try!(print_expr(s, expr));
        try!(word(&mut s.s, ";"));
      }
      ast::StmtMac(ref mac, semi) => {
        try!(space_if_not_bol(s));
        try!(print_mac(s, mac));
        if semi {
            try!(word(&mut s.s, ";"));
        }
      }
    }
    if parse::classify::stmt_ends_with_semi(st) {
        try!(word(&mut s.s, ";"));
    }
    maybe_print_trailing_comment(s, st.span, None)
}

pub fn print_block(s: &mut State, blk: &ast::Block) -> io::IoResult<()> {
    print_possibly_embedded_block(s, blk, BlockNormal, indent_unit)
}

pub fn print_block_unclosed(s: &mut State, blk: &ast::Block) -> io::IoResult<()> {
    print_possibly_embedded_block_(s, blk, BlockNormal, indent_unit, &[],
                                   false)
}

pub fn print_block_unclosed_indent(s: &mut State, blk: &ast::Block,
                                   indented: uint) -> io::IoResult<()> {
    print_possibly_embedded_block_(s, blk, BlockNormal, indented, &[], false)
}

pub fn print_block_with_attrs(s: &mut State,
                              blk: &ast::Block,
                              attrs: &[ast::Attribute]) -> io::IoResult<()> {
    print_possibly_embedded_block_(s, blk, BlockNormal, indent_unit, attrs,
                                  true)
}

pub enum EmbedType {
    BlockBlockFn,
    BlockNormal,
}

pub fn print_possibly_embedded_block(s: &mut State,
                                     blk: &ast::Block,
                                     embedded: EmbedType,
                                     indented: uint) -> io::IoResult<()> {
    print_possibly_embedded_block_(
        s, blk, embedded, indented, &[], true)
}

pub fn print_possibly_embedded_block_(s: &mut State,
                                      blk: &ast::Block,
                                      embedded: EmbedType,
                                      indented: uint,
                                      attrs: &[ast::Attribute],
                                      close_box: bool) -> io::IoResult<()> {
    match blk.rules {
      ast::UnsafeBlock(..) => try!(word_space(s, "unsafe")),
      ast::DefaultBlock => ()
    }
    try!(maybe_print_comment(s, blk.span.lo));
    {
        let ann_node = NodeBlock(s, blk);
        try!(s.ann.pre(ann_node));
    }
    try!(match embedded {
        BlockBlockFn => end(s),
        BlockNormal => bopen(s)
    });

    try!(print_inner_attributes(s, attrs));

    for vi in blk.view_items.iter() {
        try!(print_view_item(s, vi));
    }
    for st in blk.stmts.iter() {
        try!(print_stmt(s, *st));
    }
    match blk.expr {
      Some(expr) => {
        try!(space_if_not_bol(s));
        try!(print_expr(s, expr));
        try!(maybe_print_trailing_comment(s, expr.span, Some(blk.span.hi)));
      }
      _ => ()
    }
    try!(bclose_maybe_open(s, blk.span, indented, close_box));
    {
        let ann_node = NodeBlock(s, blk);
        try!(s.ann.post(ann_node));
    }
    Ok(())
}

pub fn print_if(s: &mut State, test: &ast::Expr, blk: &ast::Block,
                elseopt: Option<@ast::Expr>, chk: bool) -> io::IoResult<()> {
    try!(head(s, "if"));
    if chk { try!(word_nbsp(s, "check")); }
    try!(print_expr(s, test));
    try!(space(&mut s.s));
    try!(print_block(s, blk));
    fn do_else(s: &mut State, els: Option<@ast::Expr>) -> io::IoResult<()> {
        match els {
            Some(_else) => {
                match _else.node {
                    // "another else-if"
                    ast::ExprIf(i, t, e) => {
                        try!(cbox(s, indent_unit - 1u));
                        try!(ibox(s, 0u));
                        try!(word(&mut s.s, " else if "));
                        try!(print_expr(s, i));
                        try!(space(&mut s.s));
                        try!(print_block(s, t));
                        try!(do_else(s, e));
                    }
                    // "final else"
                    ast::ExprBlock(b) => {
                        try!(cbox(s, indent_unit - 1u));
                        try!(ibox(s, 0u));
                        try!(word(&mut s.s, " else "));
                        try!(print_block(s, b));
                    }
                    // BLEAH, constraints would be great here
                    _ => {
                        fail!("print_if saw if with weird alternative");
                    }
                }
            }
            _ => {/* fall through */ }
        }
        Ok(())
    }
    do_else(s, elseopt)
}

pub fn print_mac(s: &mut State, m: &ast::Mac) -> io::IoResult<()> {
    match m.node {
      // I think it's reasonable to hide the ctxt here:
      ast::MacInvocTT(ref pth, ref tts, _) => {
        try!(print_path(s, pth, false));
        try!(word(&mut s.s, "!"));
        try!(popen(s));
        try!(print_tts(s, &tts.as_slice()));
        pclose(s)
      }
    }
}

pub fn print_expr_vstore(s: &mut State, t: ast::ExprVstore) -> io::IoResult<()> {
    match t {
      ast::ExprVstoreUniq => word(&mut s.s, "~"),
      ast::ExprVstoreSlice => word(&mut s.s, "&"),
      ast::ExprVstoreMutSlice => {
        try!(word(&mut s.s, "&"));
        word(&mut s.s, "mut")
      }
    }
}

fn print_call_post(s: &mut State, args: &[@ast::Expr]) -> io::IoResult<()> {
    try!(popen(s));
    try!(commasep_exprs(s, Inconsistent, args));
    try!(pclose(s));
    Ok(())
}

pub fn print_expr(s: &mut State, expr: &ast::Expr) -> io::IoResult<()> {
    fn print_field(s: &mut State, field: &ast::Field) -> io::IoResult<()> {
        try!(ibox(s, indent_unit));
        try!(print_ident(s, field.ident.node));
        try!(word_space(s, ":"));
        try!(print_expr(s, field.expr));
        try!(end(s));
        Ok(())
    }
    fn get_span(field: &ast::Field) -> codemap::Span { return field.span; }

    try!(maybe_print_comment(s, expr.span.lo));
    try!(ibox(s, indent_unit));
    {
        let ann_node = NodeExpr(s, expr);
        try!(s.ann.pre(ann_node));
    }
    match expr.node {
        ast::ExprVstore(e, v) => {
            try!(print_expr_vstore(s, v));
            try!(print_expr(s, e));
        },
        ast::ExprBox(p, e) => {
            try!(word(&mut s.s, "box"));
            try!(word(&mut s.s, "("));
            try!(print_expr(s, p));
            try!(word_space(s, ")"));
            try!(print_expr(s, e));
        }
      ast::ExprVec(ref exprs, mutbl) => {
        try!(ibox(s, indent_unit));
        try!(word(&mut s.s, "["));
        if mutbl == ast::MutMutable {
            try!(word(&mut s.s, "mut"));
            if exprs.len() > 0u { try!(nbsp(s)); }
        }
        try!(commasep_exprs(s, Inconsistent, exprs.as_slice()));
        try!(word(&mut s.s, "]"));
        try!(end(s));
      }

      ast::ExprRepeat(element, count, mutbl) => {
        try!(ibox(s, indent_unit));
        try!(word(&mut s.s, "["));
        if mutbl == ast::MutMutable {
            try!(word(&mut s.s, "mut"));
            try!(nbsp(s));
        }
        try!(print_expr(s, element));
        try!(word(&mut s.s, ","));
        try!(word(&mut s.s, ".."));
        try!(print_expr(s, count));
        try!(word(&mut s.s, "]"));
        try!(end(s));
      }

      ast::ExprStruct(ref path, ref fields, wth) => {
        try!(print_path(s, path, true));
        try!(word(&mut s.s, "{"));
        try!(commasep_cmnt(s,
                           Consistent,
                           fields.as_slice(),
                           print_field,
                           get_span));
        match wth {
            Some(expr) => {
                try!(ibox(s, indent_unit));
                if !fields.is_empty() {
                    try!(word(&mut s.s, ","));
                    try!(space(&mut s.s));
                }
                try!(word(&mut s.s, ".."));
                try!(print_expr(s, expr));
                try!(end(s));
            }
            _ => try!(word(&mut s.s, ","))
        }
        try!(word(&mut s.s, "}"));
      }
      ast::ExprTup(ref exprs) => {
        try!(popen(s));
        try!(commasep_exprs(s, Inconsistent, exprs.as_slice()));
        if exprs.len() == 1 {
            try!(word(&mut s.s, ","));
        }
        try!(pclose(s));
      }
      ast::ExprCall(func, ref args) => {
        try!(print_expr(s, func));
        try!(print_call_post(s, args.as_slice()));
      }
      ast::ExprMethodCall(ident, ref tys, ref args) => {
        let base_args = args.slice_from(1);
        try!(print_expr(s, *args.get(0)));
        try!(word(&mut s.s, "."));
        try!(print_ident(s, ident));
        if tys.len() > 0u {
            try!(word(&mut s.s, "::<"));
            try!(commasep(s, Inconsistent, tys.as_slice(), print_type_ref));
            try!(word(&mut s.s, ">"));
        }
        try!(print_call_post(s, base_args));
      }
      ast::ExprBinary(op, lhs, rhs) => {
        try!(print_expr(s, lhs));
        try!(space(&mut s.s));
        try!(word_space(s, ast_util::binop_to_str(op)));
        try!(print_expr(s, rhs));
      }
      ast::ExprUnary(op, expr) => {
        try!(word(&mut s.s, ast_util::unop_to_str(op)));
        try!(print_expr(s, expr));
      }
      ast::ExprAddrOf(m, expr) => {
        try!(word(&mut s.s, "&"));
        try!(print_mutability(s, m));
        // Avoid `& &e` => `&&e`.
        match (m, &expr.node) {
            (ast::MutImmutable, &ast::ExprAddrOf(..)) => try!(space(&mut s.s)),
            _ => { }
        }
        try!(print_expr(s, expr));
      }
      ast::ExprLit(lit) => try!(print_literal(s, lit)),
      ast::ExprCast(expr, ty) => {
        try!(print_expr(s, expr));
        try!(space(&mut s.s));
        try!(word_space(s, "as"));
        try!(print_type(s, ty));
      }
      ast::ExprIf(test, blk, elseopt) => {
        try!(print_if(s, test, blk, elseopt, false));
      }
      ast::ExprWhile(test, blk) => {
        try!(head(s, "while"));
        try!(print_expr(s, test));
        try!(space(&mut s.s));
        try!(print_block(s, blk));
      }
      ast::ExprForLoop(pat, iter, blk, opt_ident) => {
        for ident in opt_ident.iter() {
            try!(word(&mut s.s, "'"));
            try!(print_ident(s, *ident));
            try!(word_space(s, ":"));
        }
        try!(head(s, "for"));
        try!(print_pat(s, pat));
        try!(space(&mut s.s));
        try!(word_space(s, "in"));
        try!(print_expr(s, iter));
        try!(space(&mut s.s));
        try!(print_block(s, blk));
      }
      ast::ExprLoop(blk, opt_ident) => {
        for ident in opt_ident.iter() {
            try!(word(&mut s.s, "'"));
            try!(print_ident(s, *ident));
            try!(word_space(s, ":"));
        }
        try!(head(s, "loop"));
        try!(space(&mut s.s));
        try!(print_block(s, blk));
      }
      ast::ExprMatch(expr, ref arms) => {
        try!(cbox(s, indent_unit));
        try!(ibox(s, 4));
        try!(word_nbsp(s, "match"));
        try!(print_expr(s, expr));
        try!(space(&mut s.s));
        try!(bopen(s));
        let len = arms.len();
        for (i, arm) in arms.iter().enumerate() {
            try!(space(&mut s.s));
            try!(cbox(s, indent_unit));
            try!(ibox(s, 0u));
            let mut first = true;
            for p in arm.pats.iter() {
                if first {
                    first = false;
                } else {
                    try!(space(&mut s.s));
                    try!(word_space(s, "|"));
                }
                try!(print_pat(s, *p));
            }
            try!(space(&mut s.s));
            match arm.guard {
              Some(e) => {
                try!(word_space(s, "if"));
                try!(print_expr(s, e));
                try!(space(&mut s.s));
              }
              None => ()
            }
            try!(word_space(s, "=>"));

            match arm.body.node {
                ast::ExprBlock(blk) => {
                    // the block will close the pattern's ibox
                    try!(print_block_unclosed_indent(
                                s, blk, indent_unit));
                }
                _ => {
                    try!(end(s)); // close the ibox for the pattern
                    try!(print_expr(s, arm.body));
                }
            }
            if !expr_is_simple_block(expr)
                && i < len - 1 {
                try!(word(&mut s.s, ","));
            }
            try!(end(s)); // close enclosing cbox
        }
        try!(bclose_(s, expr.span, indent_unit));
      }
      ast::ExprFnBlock(decl, body) => {
        // in do/for blocks we don't want to show an empty
        // argument list, but at this point we don't know which
        // we are inside.
        //
        // if !decl.inputs.is_empty() {
        try!(print_fn_block_args(s, decl));
        try!(space(&mut s.s));
        // }
        assert!(body.stmts.is_empty());
        assert!(body.expr.is_some());
        // we extract the block, so as not to create another set of boxes
        match body.expr.unwrap().node {
            ast::ExprBlock(blk) => {
                try!(print_block_unclosed(s, blk));
            }
            _ => {
                // this is a bare expression
                try!(print_expr(s, body.expr.unwrap()));
                try!(end(s)); // need to close a box
            }
        }
        // a box will be closed by print_expr, but we didn't want an overall
        // wrapper so we closed the corresponding opening. so create an
        // empty box to satisfy the close.
        try!(ibox(s, 0));
      }
      ast::ExprProc(decl, body) => {
        // in do/for blocks we don't want to show an empty
        // argument list, but at this point we don't know which
        // we are inside.
        //
        // if !decl.inputs.is_empty() {
        try!(print_proc_args(s, decl));
        try!(space(&mut s.s));
        // }
        assert!(body.stmts.is_empty());
        assert!(body.expr.is_some());
        // we extract the block, so as not to create another set of boxes
        match body.expr.unwrap().node {
            ast::ExprBlock(blk) => {
                try!(print_block_unclosed(s, blk));
            }
            _ => {
                // this is a bare expression
                try!(print_expr(s, body.expr.unwrap()));
                try!(end(s)); // need to close a box
            }
        }
        // a box will be closed by print_expr, but we didn't want an overall
        // wrapper so we closed the corresponding opening. so create an
        // empty box to satisfy the close.
        try!(ibox(s, 0));
      }
      ast::ExprBlock(blk) => {
        // containing cbox, will be closed by print-block at }
        try!(cbox(s, indent_unit));
        // head-box, will be closed by print-block after {
        try!(ibox(s, 0u));
        try!(print_block(s, blk));
      }
      ast::ExprAssign(lhs, rhs) => {
        try!(print_expr(s, lhs));
        try!(space(&mut s.s));
        try!(word_space(s, "="));
        try!(print_expr(s, rhs));
      }
      ast::ExprAssignOp(op, lhs, rhs) => {
        try!(print_expr(s, lhs));
        try!(space(&mut s.s));
        try!(word(&mut s.s, ast_util::binop_to_str(op)));
        try!(word_space(s, "="));
        try!(print_expr(s, rhs));
      }
      ast::ExprField(expr, id, ref tys) => {
        try!(print_expr(s, expr));
        try!(word(&mut s.s, "."));
        try!(print_ident(s, id));
        if tys.len() > 0u {
            try!(word(&mut s.s, "::<"));
            try!(commasep(s, Inconsistent, tys.as_slice(), print_type_ref));
            try!(word(&mut s.s, ">"));
        }
      }
      ast::ExprIndex(expr, index) => {
        try!(print_expr(s, expr));
        try!(word(&mut s.s, "["));
        try!(print_expr(s, index));
        try!(word(&mut s.s, "]"));
      }
      ast::ExprPath(ref path) => try!(print_path(s, path, true)),
      ast::ExprBreak(opt_ident) => {
        try!(word(&mut s.s, "break"));
        try!(space(&mut s.s));
        for ident in opt_ident.iter() {
            try!(word(&mut s.s, "'"));
            try!(print_ident(s, *ident));
            try!(space(&mut s.s));
        }
      }
      ast::ExprAgain(opt_ident) => {
        try!(word(&mut s.s, "continue"));
        try!(space(&mut s.s));
        for ident in opt_ident.iter() {
            try!(word(&mut s.s, "'"));
            try!(print_ident(s, *ident));
            try!(space(&mut s.s))
        }
      }
      ast::ExprRet(result) => {
        try!(word(&mut s.s, "return"));
        match result {
          Some(expr) => {
              try!(word(&mut s.s, " "));
              try!(print_expr(s, expr));
          }
          _ => ()
        }
      }
      ast::ExprLogLevel => {
        try!(word(&mut s.s, "__log_level"));
        try!(popen(s));
        try!(pclose(s));
      }
      ast::ExprInlineAsm(ref a) => {
        if a.volatile {
            try!(word(&mut s.s, "__volatile__ asm!"));
        } else {
            try!(word(&mut s.s, "asm!"));
        }
        try!(popen(s));
        try!(print_string(s, a.asm.get(), a.asm_str_style));
        try!(word_space(s, ":"));
        for &(ref co, o) in a.outputs.iter() {
            try!(print_string(s, co.get(), ast::CookedStr));
            try!(popen(s));
            try!(print_expr(s, o));
            try!(pclose(s));
            try!(word_space(s, ","));
        }
        try!(word_space(s, ":"));
        for &(ref co, o) in a.inputs.iter() {
            try!(print_string(s, co.get(), ast::CookedStr));
            try!(popen(s));
            try!(print_expr(s, o));
            try!(pclose(s));
            try!(word_space(s, ","));
        }
        try!(word_space(s, ":"));
        try!(print_string(s, a.clobbers.get(), ast::CookedStr));
        try!(pclose(s));
      }
      ast::ExprMac(ref m) => try!(print_mac(s, m)),
      ast::ExprParen(e) => {
          try!(popen(s));
          try!(print_expr(s, e));
          try!(pclose(s));
      }
    }
    {
        let ann_node = NodeExpr(s, expr);
        try!(s.ann.post(ann_node));
    }
    end(s)
}

pub fn print_local_decl(s: &mut State, loc: &ast::Local) -> io::IoResult<()> {
    try!(print_pat(s, loc.pat));
    match loc.ty.node {
        ast::TyInfer => {}
        _ => {
            try!(word_space(s, ":"));
            try!(print_type(s, loc.ty));
        }
    }
    Ok(())
}

pub fn print_decl(s: &mut State, decl: &ast::Decl) -> io::IoResult<()> {
    try!(maybe_print_comment(s, decl.span.lo));
    match decl.node {
      ast::DeclLocal(ref loc) => {
        try!(space_if_not_bol(s));
        try!(ibox(s, indent_unit));
        try!(word_nbsp(s, "let"));

        fn print_local(s: &mut State, loc: &ast::Local) -> io::IoResult<()> {
            try!(ibox(s, indent_unit));
            try!(print_local_decl(s, loc));
            try!(end(s));
            match loc.init {
              Some(init) => {
                try!(nbsp(s));
                try!(word_space(s, "="));
                try!(print_expr(s, init));
              }
              _ => ()
            }
            Ok(())
        }

        try!(print_local(s, *loc));
        end(s)
      }
      ast::DeclItem(item) => print_item(s, item)
    }
}

pub fn print_ident(s: &mut State, ident: ast::Ident) -> io::IoResult<()> {
    word(&mut s.s, token::get_ident(ident).get())
}

pub fn print_name(s: &mut State, name: ast::Name) -> io::IoResult<()> {
    word(&mut s.s, token::get_name(name).get())
}

pub fn print_for_decl(s: &mut State, loc: &ast::Local,
                      coll: &ast::Expr) -> io::IoResult<()> {
    try!(print_local_decl(s, loc));
    try!(space(&mut s.s));
    try!(word_space(s, "in"));
    print_expr(s, coll)
}

fn print_path_(s: &mut State,
               path: &ast::Path,
               colons_before_params: bool,
               opt_bounds: &Option<OptVec<ast::TyParamBound>>)
    -> io::IoResult<()>
{
    try!(maybe_print_comment(s, path.span.lo));
    if path.global {
        try!(word(&mut s.s, "::"));
    }

    let mut first = true;
    for (i, segment) in path.segments.iter().enumerate() {
        if first {
            first = false
        } else {
            try!(word(&mut s.s, "::"))
        }

        try!(print_ident(s, segment.identifier));

        // If this is the last segment, print the bounds.
        if i == path.segments.len() - 1 {
            match *opt_bounds {
                None => {}
                Some(ref bounds) => try!(print_bounds(s, bounds, true)),
            }
        }

        if !segment.lifetimes.is_empty() || !segment.types.is_empty() {
            if colons_before_params {
                try!(word(&mut s.s, "::"))
            }
            try!(word(&mut s.s, "<"));

            let mut comma = false;
            for lifetime in segment.lifetimes.iter() {
                if comma {
                    try!(word_space(s, ","))
                }
                try!(print_lifetime(s, lifetime));
                comma = true;
            }

            if !segment.types.is_empty() {
                if comma {
                    try!(word_space(s, ","))
                }
                try!(commasep(s,
                                Inconsistent,
                                segment.types.map_to_vec(|&t| t).as_slice(),
                                print_type_ref));
            }

            try!(word(&mut s.s, ">"))
        }
    }
    Ok(())
}

fn print_path(s: &mut State, path: &ast::Path,
              colons_before_params: bool) -> io::IoResult<()> {
    print_path_(s, path, colons_before_params, &None)
}

fn print_bounded_path(s: &mut State, path: &ast::Path,
                      bounds: &Option<OptVec<ast::TyParamBound>>)
    -> io::IoResult<()> {
    print_path_(s, path, false, bounds)
}

pub fn print_pat(s: &mut State, pat: &ast::Pat) -> io::IoResult<()> {
    try!(maybe_print_comment(s, pat.span.lo));
    {
        let ann_node = NodePat(s, pat);
        try!(s.ann.pre(ann_node));
    }
    /* Pat isn't normalized, but the beauty of it
     is that it doesn't matter */
    match pat.node {
      ast::PatWild => try!(word(&mut s.s, "_")),
      ast::PatWildMulti => try!(word(&mut s.s, "..")),
      ast::PatIdent(binding_mode, ref path, sub) => {
          match binding_mode {
              ast::BindByRef(mutbl) => {
                  try!(word_nbsp(s, "ref"));
                  try!(print_mutability(s, mutbl));
              }
              ast::BindByValue(ast::MutImmutable) => {}
              ast::BindByValue(ast::MutMutable) => {
                  try!(word_nbsp(s, "mut"));
              }
          }
          try!(print_path(s, path, true));
          match sub {
              Some(p) => {
                  try!(word(&mut s.s, "@"));
                  try!(print_pat(s, p));
              }
              None => ()
          }
      }
      ast::PatEnum(ref path, ref args_) => {
        try!(print_path(s, path, true));
        match *args_ {
          None => try!(word(&mut s.s, "(..)")),
          Some(ref args) => {
            if !args.is_empty() {
              try!(popen(s));
              try!(commasep(s, Inconsistent, args.as_slice(),
                              |s, &p| print_pat(s, p)));
              try!(pclose(s));
            } else { }
          }
        }
      }
      ast::PatStruct(ref path, ref fields, etc) => {
        try!(print_path(s, path, true));
        try!(word(&mut s.s, "{"));
        fn print_field(s: &mut State, f: &ast::FieldPat) -> io::IoResult<()> {
            try!(cbox(s, indent_unit));
            try!(print_ident(s, f.ident));
            try!(word_space(s, ":"));
            try!(print_pat(s, f.pat));
            try!(end(s));
            Ok(())
        }
        fn get_span(f: &ast::FieldPat) -> codemap::Span { return f.pat.span; }
        try!(commasep_cmnt(s, Consistent, fields.as_slice(),
                             |s, f| print_field(s,f),
                             get_span));
        if etc {
            if fields.len() != 0u { try!(word_space(s, ",")); }
            try!(word(&mut s.s, ".."));
        }
        try!(word(&mut s.s, "}"));
      }
      ast::PatTup(ref elts) => {
        try!(popen(s));
        try!(commasep(s,
                      Inconsistent,
                      elts.as_slice(),
                      |s, &p| print_pat(s, p)));
        if elts.len() == 1 {
            try!(word(&mut s.s, ","));
        }
        try!(pclose(s));
      }
      ast::PatUniq(inner) => {
          try!(word(&mut s.s, "~"));
          try!(print_pat(s, inner));
      }
      ast::PatRegion(inner) => {
          try!(word(&mut s.s, "&"));
          try!(print_pat(s, inner));
      }
      ast::PatLit(e) => try!(print_expr(s, e)),
      ast::PatRange(begin, end) => {
        try!(print_expr(s, begin));
        try!(space(&mut s.s));
        try!(word(&mut s.s, ".."));
        try!(print_expr(s, end));
      }
      ast::PatVec(ref before, slice, ref after) => {
        try!(word(&mut s.s, "["));
        try!(commasep(s,
                      Inconsistent,
                      before.as_slice(),
                      |s, &p| print_pat(s, p)));
        for &p in slice.iter() {
            if !before.is_empty() { try!(word_space(s, ",")); }
            match *p {
                ast::Pat { node: ast::PatWildMulti, .. } => {
                    // this case is handled by print_pat
                }
                _ => try!(word(&mut s.s, "..")),
            }
            try!(print_pat(s, p));
            if !after.is_empty() { try!(word_space(s, ",")); }
        }
        try!(commasep(s,
                      Inconsistent,
                      after.as_slice(),
                      |s, &p| print_pat(s, p)));
        try!(word(&mut s.s, "]"));
      }
    }
    {
        let ann_node = NodePat(s, pat);
        try!(s.ann.post(ann_node));
    }
    Ok(())
}

pub fn explicit_self_to_str(explicit_self: &ast::ExplicitSelf_) -> ~str {
    to_str(explicit_self, |a, &b| {
        print_explicit_self(a, b, ast::MutImmutable).map(|_| ())
    })
}

// Returns whether it printed anything
fn print_explicit_self(s: &mut State,
                       explicit_self: ast::ExplicitSelf_,
                       mutbl: ast::Mutability) -> io::IoResult<bool> {
    try!(print_mutability(s, mutbl));
    match explicit_self {
        ast::SelfStatic => { return Ok(false); }
        ast::SelfValue => {
            try!(word(&mut s.s, "self"));
        }
        ast::SelfUniq => {
            try!(word(&mut s.s, "~self"));
        }
        ast::SelfRegion(ref lt, m) => {
            try!(word(&mut s.s, "&"));
            try!(print_opt_lifetime(s, lt));
            try!(print_mutability(s, m));
            try!(word(&mut s.s, "self"));
        }
    }
    return Ok(true);
}

pub fn print_fn(s: &mut State,
                decl: &ast::FnDecl,
                purity: Option<ast::Purity>,
                abis: AbiSet,
                name: ast::Ident,
                generics: &ast::Generics,
                opt_explicit_self: Option<ast::ExplicitSelf_>,
                vis: ast::Visibility) -> io::IoResult<()> {
    try!(head(s, ""));
    try!(print_fn_header_info(s, opt_explicit_self, purity, abis,
                                ast::Many, None, vis));
    try!(nbsp(s));
    try!(print_ident(s, name));
    try!(print_generics(s, generics));
    try!(print_fn_args_and_ret(s, decl, opt_explicit_self));
    Ok(())
}

pub fn print_fn_args(s: &mut State, decl: &ast::FnDecl,
                     opt_explicit_self: Option<ast::ExplicitSelf_>)
    -> io::IoResult<()>
{
    // It is unfortunate to duplicate the commasep logic, but we want the
    // self type and the args all in the same box.
    try!(rbox(s, 0u, Inconsistent));
    let mut first = true;
    for &explicit_self in opt_explicit_self.iter() {
        let m = match explicit_self {
            ast::SelfStatic => ast::MutImmutable,
            _ => match decl.inputs.get(0).pat.node {
                ast::PatIdent(ast::BindByValue(m), _, _) => m,
                _ => ast::MutImmutable
            }
        };
        first = !try!(print_explicit_self(s, explicit_self, m));
    }

    // HACK(eddyb) ignore the separately printed self argument.
    let args = if first {
        decl.inputs.as_slice()
    } else {
        decl.inputs.slice_from(1)
    };

    for arg in args.iter() {
        if first { first = false; } else { try!(word_space(s, ",")); }
        try!(print_arg(s, arg));
    }

    end(s)
}

pub fn print_fn_args_and_ret(s: &mut State, decl: &ast::FnDecl,
                             opt_explicit_self: Option<ast::ExplicitSelf_>)
    -> io::IoResult<()>
{
    try!(popen(s));
    try!(print_fn_args(s, decl, opt_explicit_self));
    if decl.variadic {
        try!(word(&mut s.s, ", ..."));
    }
    try!(pclose(s));

    try!(maybe_print_comment(s, decl.output.span.lo));
    match decl.output.node {
        ast::TyNil => {}
        _ => {
            try!(space_if_not_bol(s));
            try!(word_space(s, "->"));
            try!(print_type(s, decl.output));
        }
    }
    Ok(())
}

pub fn print_fn_block_args(s: &mut State,
                           decl: &ast::FnDecl) -> io::IoResult<()> {
    try!(word(&mut s.s, "|"));
    try!(print_fn_args(s, decl, None));
    try!(word(&mut s.s, "|"));

    match decl.output.node {
        ast::TyInfer => {}
        _ => {
            try!(space_if_not_bol(s));
            try!(word_space(s, "->"));
            try!(print_type(s, decl.output));
        }
    }

    maybe_print_comment(s, decl.output.span.lo)
}

pub fn print_proc_args(s: &mut State, decl: &ast::FnDecl) -> io::IoResult<()> {
    try!(word(&mut s.s, "proc"));
    try!(word(&mut s.s, "("));
    try!(print_fn_args(s, decl, None));
    try!(word(&mut s.s, ")"));

    match decl.output.node {
        ast::TyInfer => {}
        _ => {
            try!(space_if_not_bol(s));
            try!(word_space(s, "->"));
            try!(print_type(s, decl.output));
        }
    }

    maybe_print_comment(s, decl.output.span.lo)
}

pub fn print_bounds(s: &mut State, bounds: &OptVec<ast::TyParamBound>,
                    print_colon_anyway: bool) -> io::IoResult<()> {
    if !bounds.is_empty() {
        try!(word(&mut s.s, ":"));
        let mut first = true;
        for bound in bounds.iter() {
            try!(nbsp(s));
            if first {
                first = false;
            } else {
                try!(word_space(s, "+"));
            }

            try!(match *bound {
                TraitTyParamBound(ref tref) => print_trait_ref(s, tref),
                RegionTyParamBound => word(&mut s.s, "'static"),
            })
        }
    } else if print_colon_anyway {
        try!(word(&mut s.s, ":"));
    }
    Ok(())
}

pub fn print_lifetime(s: &mut State,
                      lifetime: &ast::Lifetime) -> io::IoResult<()> {
    try!(word(&mut s.s, "'"));
    print_name(s, lifetime.name)
}

pub fn print_generics(s: &mut State,
                      generics: &ast::Generics) -> io::IoResult<()> {
    let total = generics.lifetimes.len() + generics.ty_params.len();
    if total > 0 {
        try!(word(&mut s.s, "<"));
        fn print_item(s: &mut State, generics: &ast::Generics,
                      idx: uint) -> io::IoResult<()> {
            if idx < generics.lifetimes.len() {
                let lifetime = generics.lifetimes.get(idx);
                print_lifetime(s, lifetime)
            } else {
                let idx = idx - generics.lifetimes.len();
                let param = generics.ty_params.get(idx);
                try!(print_ident(s, param.ident));
                try!(print_bounds(s, &param.bounds, false));
                match param.default {
                    Some(default) => {
                        try!(space(&mut s.s));
                        try!(word_space(s, "="));
                        try!(print_type(s, default));
                    }
                    _ => {}
                }
                Ok(())
            }
        }

        let mut ints = Vec::new();
        for i in range(0u, total) {
            ints.push(i);
        }

        try!(commasep(s, Inconsistent, ints.as_slice(),
                        |s, &i| print_item(s, generics, i)));
        try!(word(&mut s.s, ">"));
    }
    Ok(())
}

pub fn print_meta_item(s: &mut State, item: &ast::MetaItem) -> io::IoResult<()> {
    try!(ibox(s, indent_unit));
    match item.node {
        ast::MetaWord(ref name) => {
            try!(word(&mut s.s, name.get()));
        }
        ast::MetaNameValue(ref name, ref value) => {
            try!(word_space(s, name.get()));
            try!(word_space(s, "="));
            try!(print_literal(s, value));
        }
        ast::MetaList(ref name, ref items) => {
            try!(word(&mut s.s, name.get()));
            try!(popen(s));
            try!(commasep(s,
                            Consistent,
                            items.as_slice(),
                            |p, &i| print_meta_item(p, i)));
            try!(pclose(s));
        }
    }
    end(s)
}

pub fn print_view_path(s: &mut State, vp: &ast::ViewPath) -> io::IoResult<()> {
    match vp.node {
      ast::ViewPathSimple(ident, ref path, _) => {
        // FIXME(#6993) can't compare identifiers directly here
        if path.segments.last().unwrap().identifier.name != ident.name {
            try!(print_ident(s, ident));
            try!(space(&mut s.s));
            try!(word_space(s, "="));
        }
        print_path(s, path, false)
      }

      ast::ViewPathGlob(ref path, _) => {
        try!(print_path(s, path, false));
        word(&mut s.s, "::*")
      }

      ast::ViewPathList(ref path, ref idents, _) => {
        if path.segments.is_empty() {
            try!(word(&mut s.s, "{"));
        } else {
            try!(print_path(s, path, false));
            try!(word(&mut s.s, "::{"));
        }
        try!(commasep(s, Inconsistent, idents.as_slice(), |s, w| {
            print_ident(s, w.node.name)
        }));
        word(&mut s.s, "}")
      }
    }
}

pub fn print_view_paths(s: &mut State,
                        vps: &[@ast::ViewPath]) -> io::IoResult<()> {
    commasep(s, Inconsistent, vps, |p, &vp| print_view_path(p, vp))
}

pub fn print_view_item(s: &mut State, item: &ast::ViewItem) -> io::IoResult<()> {
    try!(hardbreak_if_not_bol(s));
    try!(maybe_print_comment(s, item.span.lo));
    try!(print_outer_attributes(s, item.attrs.as_slice()));
    try!(print_visibility(s, item.vis));
    match item.node {
        ast::ViewItemExternCrate(id, ref optional_path, _) => {
            try!(head(s, "extern crate"));
            try!(print_ident(s, id));
            for &(ref p, style) in optional_path.iter() {
                try!(space(&mut s.s));
                try!(word(&mut s.s, "="));
                try!(space(&mut s.s));
                try!(print_string(s, p.get(), style));
            }
        }

        ast::ViewItemUse(ref vps) => {
            try!(head(s, "use"));
            try!(print_view_paths(s, vps.as_slice()));
        }
    }
    try!(word(&mut s.s, ";"));
    try!(end(s)); // end inner head-block
    try!(end(s)); // end outer head-block
    Ok(())
}

pub fn print_mutability(s: &mut State,
                        mutbl: ast::Mutability) -> io::IoResult<()> {
    match mutbl {
      ast::MutMutable => word_nbsp(s, "mut"),
      ast::MutImmutable => Ok(()),
    }
}

pub fn print_mt(s: &mut State, mt: &ast::MutTy) -> io::IoResult<()> {
    try!(print_mutability(s, mt.mutbl));
    print_type(s, mt.ty)
}

pub fn print_arg(s: &mut State, input: &ast::Arg) -> io::IoResult<()> {
    try!(ibox(s, indent_unit));
    match input.ty.node {
        ast::TyInfer => try!(print_pat(s, input.pat)),
        _ => {
            match input.pat.node {
                ast::PatIdent(_, ref path, _) if
                    path.segments.len() == 1 &&
                    path.segments.get(0).identifier.name ==
                        parse::token::special_idents::invalid.name => {
                    // Do nothing.
                }
                _ => {
                    try!(print_pat(s, input.pat));
                    try!(word(&mut s.s, ":"));
                    try!(space(&mut s.s));
                }
            }
            try!(print_type(s, input.ty));
        }
    }
    end(s)
}

pub fn print_ty_fn(s: &mut State,
                   opt_abis: Option<AbiSet>,
                   opt_sigil: Option<ast::Sigil>,
                   opt_region: &Option<ast::Lifetime>,
                   purity: ast::Purity,
                   onceness: ast::Onceness,
                   decl: &ast::FnDecl,
                   id: Option<ast::Ident>,
                   opt_bounds: &Option<OptVec<ast::TyParamBound>>,
                   generics: Option<&ast::Generics>,
                   opt_explicit_self: Option<ast::ExplicitSelf_>)
    -> io::IoResult<()>
{
    try!(ibox(s, indent_unit));

    // Duplicates the logic in `print_fn_header_info()`.  This is because that
    // function prints the sigil in the wrong place.  That should be fixed.
    if opt_sigil == Some(ast::OwnedSigil) && onceness == ast::Once {
        try!(word(&mut s.s, "proc"));
    } else if opt_sigil == Some(ast::BorrowedSigil) {
        try!(print_extern_opt_abis(s, opt_abis));
        for lifetime in opt_region.iter() {
            try!(print_lifetime(s, lifetime));
        }
        try!(print_purity(s, purity));
        try!(print_onceness(s, onceness));
    } else {
        try!(print_opt_abis_and_extern_if_nondefault(s, opt_abis));
        try!(print_opt_sigil(s, opt_sigil));
        try!(print_opt_lifetime(s, opt_region));
        try!(print_purity(s, purity));
        try!(print_onceness(s, onceness));
        try!(word(&mut s.s, "fn"));
    }

    match id {
        Some(id) => {
            try!(word(&mut s.s, " "));
            try!(print_ident(s, id));
        }
        _ => ()
    }

    if opt_sigil != Some(ast::BorrowedSigil) {
        opt_bounds.as_ref().map(|bounds| print_bounds(s, bounds, true));
    }

    match generics { Some(g) => try!(print_generics(s, g)), _ => () }
    try!(zerobreak(&mut s.s));

    if opt_sigil == Some(ast::BorrowedSigil) {
        try!(word(&mut s.s, "|"));
    } else {
        try!(popen(s));
    }

    try!(print_fn_args(s, decl, opt_explicit_self));

    if opt_sigil == Some(ast::BorrowedSigil) {
        try!(word(&mut s.s, "|"));

        opt_bounds.as_ref().map(|bounds| print_bounds(s, bounds, true));
    } else {
        if decl.variadic {
            try!(word(&mut s.s, ", ..."));
        }
        try!(pclose(s));
    }

    try!(maybe_print_comment(s, decl.output.span.lo));

    match decl.output.node {
        ast::TyNil => {}
        _ => {
            try!(space_if_not_bol(s));
            try!(ibox(s, indent_unit));
            try!(word_space(s, "->"));
            if decl.cf == ast::NoReturn {
                try!(word_nbsp(s, "!"));
            } else {
                try!(print_type(s, decl.output));
            }
            try!(end(s));
        }
    }

    end(s)
}

pub fn maybe_print_trailing_comment(s: &mut State, span: codemap::Span,
                                    next_pos: Option<BytePos>)
    -> io::IoResult<()>
{
    let cm;
    match s.cm { Some(ccm) => cm = ccm, _ => return Ok(()) }
    match next_comment(s) {
        Some(ref cmnt) => {
            if (*cmnt).style != comments::Trailing { return Ok(()) }
            let span_line = cm.lookup_char_pos(span.hi);
            let comment_line = cm.lookup_char_pos((*cmnt).pos);
            let mut next = (*cmnt).pos + BytePos(1);
            match next_pos { None => (), Some(p) => next = p }
            if span.hi < (*cmnt).pos && (*cmnt).pos < next &&
                span_line.line == comment_line.line {
                    try!(print_comment(s, cmnt));
                    s.cur_cmnt_and_lit.cur_cmnt += 1u;
                }
        }
        _ => ()
    }
    Ok(())
}

pub fn print_remaining_comments(s: &mut State) -> io::IoResult<()> {
    // If there aren't any remaining comments, then we need to manually
    // make sure there is a line break at the end.
    if next_comment(s).is_none() {
        try!(hardbreak(&mut s.s));
    }
    loop {
        match next_comment(s) {
            Some(ref cmnt) => {
                try!(print_comment(s, cmnt));
                s.cur_cmnt_and_lit.cur_cmnt += 1u;
            }
            _ => break
        }
    }
    Ok(())
}

pub fn print_literal(s: &mut State, lit: &ast::Lit) -> io::IoResult<()> {
    try!(maybe_print_comment(s, lit.span.lo));
    match next_lit(s, lit.span.lo) {
      Some(ref ltrl) => {
        return word(&mut s.s, (*ltrl).lit);
      }
      _ => ()
    }
    match lit.node {
      ast::LitStr(ref st, style) => print_string(s, st.get(), style),
      ast::LitChar(ch) => {
          let mut res = ~"'";
          char::from_u32(ch).unwrap().escape_default(|c| res.push_char(c));
          res.push_char('\'');
          word(&mut s.s, res)
      }
      ast::LitInt(i, t) => {
        word(&mut s.s, format!("{}{}", i, ast_util::int_ty_to_str(t)))
      }
      ast::LitUint(u, t) => {
        word(&mut s.s, format!("{}{}", u, ast_util::uint_ty_to_str(t)))
      }
      ast::LitIntUnsuffixed(i) => {
        word(&mut s.s, format!("{}", i))
      }
      ast::LitFloat(ref f, t) => {
        word(&mut s.s, f.get() + ast_util::float_ty_to_str(t))
      }
      ast::LitFloatUnsuffixed(ref f) => word(&mut s.s, f.get()),
      ast::LitNil => word(&mut s.s, "()"),
      ast::LitBool(val) => {
        if val { word(&mut s.s, "true") } else { word(&mut s.s, "false") }
      }
      ast::LitBinary(ref arr) => {
        try!(ibox(s, indent_unit));
        try!(word(&mut s.s, "["));
        try!(commasep_cmnt(s, Inconsistent, arr.deref().as_slice(),
                             |s, u| word(&mut s.s, format!("{}", *u)),
                             |_| lit.span));
        try!(word(&mut s.s, "]"));
        end(s)
      }
    }
}

pub fn lit_to_str(l: &ast::Lit) -> ~str {
    to_str(l, print_literal)
}

pub fn next_lit(s: &mut State, pos: BytePos) -> Option<comments::Literal> {
    match s.literals {
      Some(ref lits) => {
        while s.cur_cmnt_and_lit.cur_lit < lits.len() {
            let ltrl = (*(*lits).get(s.cur_cmnt_and_lit.cur_lit)).clone();
            if ltrl.pos > pos { return None; }
            s.cur_cmnt_and_lit.cur_lit += 1u;
            if ltrl.pos == pos { return Some(ltrl); }
        }
        return None;
      }
      _ => return None
    }
}

pub fn maybe_print_comment(s: &mut State, pos: BytePos) -> io::IoResult<()> {
    loop {
        match next_comment(s) {
          Some(ref cmnt) => {
            if (*cmnt).pos < pos {
                try!(print_comment(s, cmnt));
                s.cur_cmnt_and_lit.cur_cmnt += 1u;
            } else { break; }
          }
          _ => break
        }
    }
    Ok(())
}

pub fn print_comment(s: &mut State,
                     cmnt: &comments::Comment) -> io::IoResult<()> {
    match cmnt.style {
        comments::Mixed => {
            assert_eq!(cmnt.lines.len(), 1u);
            try!(zerobreak(&mut s.s));
            try!(word(&mut s.s, *cmnt.lines.get(0)));
            try!(zerobreak(&mut s.s));
        }
        comments::Isolated => {
            try!(pprust::hardbreak_if_not_bol(s));
            for line in cmnt.lines.iter() {
                // Don't print empty lines because they will end up as trailing
                // whitespace
                if !line.is_empty() {
                    try!(word(&mut s.s, *line));
                }
                try!(hardbreak(&mut s.s));
            }
        }
        comments::Trailing => {
            try!(word(&mut s.s, " "));
            if cmnt.lines.len() == 1u {
                try!(word(&mut s.s, *cmnt.lines.get(0)));
                try!(hardbreak(&mut s.s));
            } else {
                try!(ibox(s, 0u));
                for line in cmnt.lines.iter() {
                    if !line.is_empty() {
                        try!(word(&mut s.s, *line));
                    }
                    try!(hardbreak(&mut s.s));
                }
                try!(end(s));
            }
        }
        comments::BlankLine => {
            // We need to do at least one, possibly two hardbreaks.
            let is_semi = match s.s.last_token() {
                pp::String(s, _) => ";" == s,
                _ => false
            };
            if is_semi || is_begin(s) || is_end(s) {
                try!(hardbreak(&mut s.s));
            }
            try!(hardbreak(&mut s.s));
        }
    }
    Ok(())
}

pub fn print_string(s: &mut State, st: &str,
                    style: ast::StrStyle) -> io::IoResult<()> {
    let st = match style {
        ast::CookedStr => format!("\"{}\"", st.escape_default()),
        ast::RawStr(n) => format!("r{delim}\"{string}\"{delim}",
                                  delim="#".repeat(n), string=st)
    };
    word(&mut s.s, st)
}

// FIXME(pcwalton): A nasty function to extract the string from an `io::Writer`
// that we "know" to be a `MemWriter` that works around the lack of checked
// downcasts.
unsafe fn get_mem_writer(writer: &mut ~io::Writer) -> ~str {
    let (_, wr): (uint, ~MemWriter) = cast::transmute_copy(writer);
    let result = str::from_utf8_owned(wr.get_ref().to_owned()).unwrap();
    cast::forget(wr);
    result
}

pub fn to_str<T>(t: &T, f: |&mut State, &T| -> io::IoResult<()>) -> ~str {
    let wr = ~MemWriter::new();
    let mut s = rust_printer(wr as ~io::Writer);
    f(&mut s, t).unwrap();
    eof(&mut s.s).unwrap();
    unsafe {
        get_mem_writer(&mut s.s.out)
    }
}

pub fn next_comment(s: &mut State) -> Option<comments::Comment> {
    match s.comments {
        Some(ref cmnts) => {
            if s.cur_cmnt_and_lit.cur_cmnt < cmnts.len() {
                Some((*cmnts.get(s.cur_cmnt_and_lit.cur_cmnt)).clone())
            } else {
                None
            }
        }
        _ => None
    }
}

pub fn print_opt_purity(s: &mut State,
                        opt_purity: Option<ast::Purity>) -> io::IoResult<()> {
    match opt_purity {
        Some(ast::ImpureFn) => { }
        Some(purity) => {
            try!(word_nbsp(s, purity_to_str(purity)));
        }
        None => {}
    }
    Ok(())
}

pub fn print_opt_abis_and_extern_if_nondefault(s: &mut State,
                                               opt_abis: Option<AbiSet>)
    -> io::IoResult<()>
{
    match opt_abis {
        Some(abis) if !abis.is_rust() => {
            try!(word_nbsp(s, "extern"));
            try!(word_nbsp(s, abis.to_str()));
        }
        Some(_) | None => {}
    };
    Ok(())
}

pub fn print_extern_opt_abis(s: &mut State,
                             opt_abis: Option<AbiSet>) -> io::IoResult<()> {
    match opt_abis {
        Some(abis) => {
            try!(word_nbsp(s, "extern"));
            try!(word_nbsp(s, abis.to_str()));
        }
        None => {}
    }
    Ok(())
}

pub fn print_opt_sigil(s: &mut State,
                       opt_sigil: Option<ast::Sigil>) -> io::IoResult<()> {
    match opt_sigil {
        Some(ast::BorrowedSigil) => word(&mut s.s, "&"),
        Some(ast::OwnedSigil) => word(&mut s.s, "~"),
        Some(ast::ManagedSigil) => word(&mut s.s, "@"),
        None => Ok(())
    }
}

pub fn print_fn_header_info(s: &mut State,
                            _opt_explicit_self: Option<ast::ExplicitSelf_>,
                            opt_purity: Option<ast::Purity>,
                            abis: AbiSet,
                            onceness: ast::Onceness,
                            opt_sigil: Option<ast::Sigil>,
                            vis: ast::Visibility) -> io::IoResult<()> {
    try!(word(&mut s.s, visibility_qualified(vis, "")));

    if abis != AbiSet::Rust() {
        try!(word_nbsp(s, "extern"));
        try!(word_nbsp(s, abis.to_str()));

        if opt_purity != Some(ast::ExternFn) {
            try!(print_opt_purity(s, opt_purity));
        }
    } else {
        try!(print_opt_purity(s, opt_purity));
    }

    try!(print_onceness(s, onceness));
    try!(word(&mut s.s, "fn"));
    try!(print_opt_sigil(s, opt_sigil));
    Ok(())
}

pub fn purity_to_str(p: ast::Purity) -> &'static str {
    match p {
      ast::ImpureFn => "impure",
      ast::UnsafeFn => "unsafe",
      ast::ExternFn => "extern"
    }
}

pub fn onceness_to_str(o: ast::Onceness) -> &'static str {
    match o {
        ast::Once => "once",
        ast::Many => "many"
    }
}

pub fn print_purity(s: &mut State, p: ast::Purity) -> io::IoResult<()> {
    match p {
      ast::ImpureFn => Ok(()),
      _ => word_nbsp(s, purity_to_str(p))
    }
}

pub fn print_onceness(s: &mut State, o: ast::Onceness) -> io::IoResult<()> {
    match o {
        ast::Once => word_nbsp(s, "once"),
        ast::Many => Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ast;
    use ast_util;
    use codemap;
    use parse::token;

    use std::vec_ng::Vec;

    #[test]
    fn test_fun_to_str() {
        let abba_ident = token::str_to_ident("abba");

        let decl = ast::FnDecl {
            inputs: Vec::new(),
            output: ast::P(ast::Ty {id: 0,
                                    node: ast::TyNil,
                                    span: codemap::DUMMY_SP}),
            cf: ast::Return,
            variadic: false
        };
        let generics = ast_util::empty_generics();
        assert_eq!(&fun_to_str(&decl, ast::ImpureFn, abba_ident,
                               None, &generics),
                   &~"fn abba()");
    }

    #[test]
    fn test_variant_to_str() {
        let ident = token::str_to_ident("principal_skinner");

        let var = codemap::respan(codemap::DUMMY_SP, ast::Variant_ {
            name: ident,
            attrs: Vec::new(),
            // making this up as I go.... ?
            kind: ast::TupleVariantKind(Vec::new()),
            id: 0,
            disr_expr: None,
            vis: ast::Public,
        });

        let varstr = variant_to_str(&var);
        assert_eq!(&varstr,&~"pub principal_skinner");
    }
}
