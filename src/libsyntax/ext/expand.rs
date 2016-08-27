// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, Ident, Mac_, PatKind};
use ast::{MacStmtStyle, Stmt, StmtKind, ItemKind};
use ast;
use ext::hygiene::Mark;
use attr::{self, HasAttrs};
use codemap::{dummy_spanned, ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
use syntax_pos::{self, Span, ExpnId};
use config::StripUnconfigured;
use ext::base::*;
use feature_gate::{self, Features};
use fold;
use fold::*;
use parse::token::{intern, keywords};
use ptr::P;
use tokenstream::TokenTree;
use util::small_vector::SmallVector;
use visit;
use visit::Visitor;
use std_inject;

use std::path::PathBuf;

macro_rules! expansions {
    ($($kind:ident: $ty:ty, $kind_name:expr, .$make:ident,
            $(.$fold:ident)*  $(lift .$fold_elt:ident)*,
            $(.$visit:ident)* $(lift .$visit_elt:ident)*;)*) => {
        #[derive(Copy, Clone)]
        enum ExpansionKind { OptExpr, $( $kind, )*  }
        enum Expansion { OptExpr(Option<P<ast::Expr>>), $( $kind($ty), )* }

        impl ExpansionKind {
            fn name(self) -> &'static str {
                match self {
                    ExpansionKind::OptExpr => "expression",
                    $( ExpansionKind::$kind => $kind_name, )*
                }
            }

            fn make_from<'a>(self, result: Box<MacResult + 'a>) -> Option<Expansion> {
                match self {
                    ExpansionKind::OptExpr => result.make_expr().map(Some).map(Expansion::OptExpr),
                    $( ExpansionKind::$kind => result.$make().map(Expansion::$kind), )*
                }
            }
        }

        impl Expansion {
            fn make_opt_expr(self) -> Option<P<ast::Expr>> {
                match self {
                    Expansion::OptExpr(expr) => expr,
                    _ => panic!("Expansion::make_* called on the wrong kind of expansion"),
                }
            }
            $( fn $make(self) -> $ty {
                match self {
                    Expansion::$kind(ast) => ast,
                    _ => panic!("Expansion::make_* called on the wrong kind of expansion"),
                }
            } )*

            fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
                use self::Expansion::*;
                match self {
                    OptExpr(expr) => OptExpr(expr.and_then(|expr| folder.fold_opt_expr(expr))),
                    $($( $kind(ast) => $kind(folder.$fold(ast)), )*)*
                    $($( $kind(ast) => {
                        $kind(ast.into_iter().flat_map(|ast| folder.$fold_elt(ast)).collect())
                    }, )*)*
                }
            }

            fn visit_with<V: Visitor>(&self, visitor: &mut V) {
                match *self {
                    Expansion::OptExpr(Some(ref expr)) => visitor.visit_expr(expr),
                    $($( Expansion::$kind(ref ast) => visitor.$visit(ast), )*)*
                    $($( Expansion::$kind(ref ast) => for ast in ast.as_slice() {
                        visitor.$visit_elt(ast);
                    }, )*)*
                    _ => {}
                }
            }
        }
    }
}

expansions! {
    Expr: P<ast::Expr>, "expression", .make_expr, .fold_expr, .visit_expr;
    Pat: P<ast::Pat>,   "pattern",    .make_pat,  .fold_pat,  .visit_pat;
    Ty: P<ast::Ty>,     "type",       .make_ty,   .fold_ty,   .visit_ty;
    Stmts: SmallVector<ast::Stmt>, "statement", .make_stmts, lift .fold_stmt, lift .visit_stmt;
    Items: SmallVector<P<ast::Item>>, "item",   .make_items, lift .fold_item, lift .visit_item;
    TraitItems: SmallVector<ast::TraitItem>,
        "trait item", .make_trait_items, lift .fold_trait_item, lift .visit_trait_item;
    ImplItems: SmallVector<ast::ImplItem>,
        "impl item",  .make_impl_items,  lift .fold_impl_item,  lift .visit_impl_item;
}

impl ExpansionKind {
    fn dummy(self, span: Span) -> Expansion {
        self.make_from(DummyResult::any(span)).unwrap()
    }
}

pub struct Invocation {
    span: Span,
    attrs: Vec<ast::Attribute>,
    mac: ast::Mac,
    ident: Option<Ident>,
    mark: Mark,
    kind: ExpansionKind,
}

pub fn expand_expr(expr: ast::Expr, fld: &mut MacroExpander) -> P<ast::Expr> {
    if let ast::ExprKind::Mac(mac) = expr.node {
        let invoc = fld.new_invoc(mac, expr.attrs.into(), expr.span, ExpansionKind::Expr);
        expand_mac_invoc(invoc, fld).make_expr()
    } else {
        P(noop_fold_expr(expr, fld))
    }
}

/// Expand a macro invocation. Returns the result of expansion.
fn expand_mac_invoc(invoc: Invocation, fld: &mut MacroExpander) -> Expansion {
    let Invocation { span, attrs, mac, ident, mark, kind } = invoc;
    let Mac_ { path, tts, .. } = mac.node;

    // Detect use of feature-gated or invalid attributes on macro invoations
    // since they will not be detected after macro expansion.
    for attr in attrs.iter() {
        feature_gate::check_attribute(&attr, &fld.cx.parse_sess.span_diagnostic,
                                      &fld.cx.parse_sess.codemap(),
                                      &fld.cx.ecfg.features.unwrap());
    }

    if path.segments.len() > 1 || path.global || !path.segments[0].parameters.is_empty() {
        fld.cx.span_err(path.span, "expected macro name without module separators");
        return kind.dummy(span);
    }

    let extname = path.segments[0].identifier.name;
    let extension = if let Some(extension) = fld.cx.syntax_env.find(extname) {
        extension
    } else {
        let mut err =
            fld.cx.struct_span_err(path.span, &format!("macro undefined: '{}!'", &extname));
        fld.cx.suggest_macro_name(&extname.as_str(), &mut err);
        err.emit();
        return kind.dummy(span);
    };

    let ident = ident.unwrap_or(keywords::Invalid.ident());
    let marked_tts = mark_tts(&tts, mark);
    let opt_expanded = match *extension {
        NormalTT(ref expandfun, exp_span, allow_internal_unstable) => {
            if ident.name != keywords::Invalid.name() {
                let msg =
                    format!("macro {}! expects no ident argument, given '{}'", extname, ident);
                fld.cx.span_err(path.span, &msg);
                return kind.dummy(span);
            }

            fld.cx.bt_push(ExpnInfo {
                call_site: span,
                callee: NameAndSpan {
                    format: MacroBang(extname),
                    span: exp_span,
                    allow_internal_unstable: allow_internal_unstable,
                },
            });

            kind.make_from(expandfun.expand(fld.cx, span, &marked_tts))
        }

        IdentTT(ref expander, tt_span, allow_internal_unstable) => {
            if ident.name == keywords::Invalid.name() {
                fld.cx.span_err(path.span,
                                &format!("macro {}! expects an ident argument", extname));
                return kind.dummy(span);
            };

            fld.cx.bt_push(ExpnInfo {
                call_site: span,
                callee: NameAndSpan {
                    format: MacroBang(extname),
                    span: tt_span,
                    allow_internal_unstable: allow_internal_unstable,
                }
            });

            kind.make_from(expander.expand(fld.cx, span, ident, marked_tts))
        }

        MacroRulesTT => {
            if ident.name == keywords::Invalid.name() {
                fld.cx.span_err(path.span,
                                &format!("macro {}! expects an ident argument", extname));
                return kind.dummy(span);
            };

            fld.cx.bt_push(ExpnInfo {
                call_site: span,
                callee: NameAndSpan {
                    format: MacroBang(extname),
                    span: None,
                    // `macro_rules!` doesn't directly allow unstable
                    // (this is orthogonal to whether the macro it creates allows it)
                    allow_internal_unstable: false,
                }
            });

            let def = ast::MacroDef {
                ident: ident,
                id: ast::DUMMY_NODE_ID,
                span: span,
                imported_from: None,
                use_locally: true,
                body: marked_tts,
                export: attr::contains_name(&attrs, "macro_export"),
                allow_internal_unstable: attr::contains_name(&attrs, "allow_internal_unstable"),
                attrs: attrs,
            };

            fld.cx.insert_macro(def.clone());

            // If keep_macs is true, expands to a MacEager::items instead.
            if fld.keep_macs {
                Some(reconstruct_macro_rules(&def, &path))
            } else {
                Some(macro_scope_placeholder())
            }
        }

        MultiDecorator(..) | MultiModifier(..) => {
            fld.cx.span_err(path.span,
                            &format!("`{}` can only be used in attributes", extname));
            return kind.dummy(span);
        }
    };

    let expanded = if let Some(expanded) = opt_expanded {
        expanded
    } else {
        let msg = format!("non-{kind} macro in {kind} position: {name}",
                          name = path.segments[0].identifier.name, kind = kind.name());
        fld.cx.span_err(path.span, &msg);
        return kind.dummy(span);
    };

    let marked = expanded.fold_with(&mut Marker { mark: mark, expn_id: Some(fld.cx.backtrace()) });
    let configured = marked.fold_with(&mut fld.strip_unconfigured());
    fld.load_macros(&configured);

    let fully_expanded = if fld.single_step {
        configured
    } else {
        configured.fold_with(fld)
    };

    fld.cx.bt_pop();
    fully_expanded
}

// When we enter a module, record it, for the sake of `module!`
pub fn expand_item(it: P<ast::Item>, fld: &mut MacroExpander)
                   -> SmallVector<P<ast::Item>> {
    expand_annotatable(Annotatable::Item(it), fld)
        .into_iter().map(|i| i.expect_item()).collect()
}

// does this attribute list contain "macro_use" ?
fn contains_macro_use(fld: &mut MacroExpander, attrs: &[ast::Attribute]) -> bool {
    for attr in attrs {
        let mut is_use = attr.check_name("macro_use");
        if attr.check_name("macro_escape") {
            let mut err =
                fld.cx.struct_span_warn(attr.span,
                                        "macro_escape is a deprecated synonym for macro_use");
            is_use = true;
            if let ast::AttrStyle::Inner = attr.node.style {
                err.help("consider an outer attribute, \
                          #[macro_use] mod ...").emit();
            } else {
                err.emit();
            }
        };

        if is_use {
            if !attr.is_word() {
              fld.cx.span_err(attr.span, "arguments to macro_use are not allowed here");
            }
            return true;
        }
    }
    false
}

/// Expand a stmt
fn expand_stmt(stmt: Stmt, fld: &mut MacroExpander) -> SmallVector<Stmt> {
    let (mac, style, attrs) = match stmt.node {
        StmtKind::Mac(mac) => mac.unwrap(),
        _ => return noop_fold_stmt(stmt, fld)
    };

    let invoc = fld.new_invoc(mac, attrs.into(), stmt.span, ExpansionKind::Stmts);
    let mut fully_expanded = expand_mac_invoc(invoc, fld).make_stmts();

    // If this is a macro invocation with a semicolon, then apply that
    // semicolon to the final statement produced by expansion.
    if style == MacStmtStyle::Semicolon {
        if let Some(stmt) = fully_expanded.pop() {
            fully_expanded.push(stmt.add_trailing_semicolon());
        }
    }

    fully_expanded
}

fn expand_pat(p: P<ast::Pat>, fld: &mut MacroExpander) -> P<ast::Pat> {
    match p.node {
        PatKind::Mac(_) => {}
        _ => return noop_fold_pat(p, fld)
    }
    p.and_then(|p| match p.node {
        PatKind::Mac(mac) => {
            let invoc = fld.new_invoc(mac, Vec::new(), p.span, ExpansionKind::Pat);
            expand_mac_invoc(invoc, fld).make_pat()
        }
        _ => unreachable!(),
    })
}

fn expand_multi_modified(a: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    match a {
        Annotatable::Item(it) => match it.node {
            ast::ItemKind::Mac(..) => {
                if match it.node {
                    ItemKind::Mac(ref mac) => mac.node.path.segments.is_empty(),
                    _ => unreachable!(),
                } {
                    return SmallVector::one(Annotatable::Item(it));
                }
                it.and_then(|it| match it.node {
                    ItemKind::Mac(mac) => {
                        let mut invoc = fld.new_invoc(mac, it.attrs, it.span, ExpansionKind::Items);
                        invoc.ident = Some(it.ident);
                        expand_mac_invoc(invoc, fld).make_items()
                    }
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(_) | ast::ItemKind::ForeignMod(_) => {
                let valid_ident =
                    it.ident.name != keywords::Invalid.name();

                if valid_ident {
                    fld.cx.mod_push(it.ident);
                }
                let macro_use = contains_macro_use(fld, &it.attrs);
                let result = fld.with_exts_frame(macro_use, |fld| noop_fold_item(it, fld));
                if valid_ident {
                    fld.cx.mod_pop();
                }
                result
            },
            _ => noop_fold_item(it, fld),
        }.into_iter().map(|i| Annotatable::Item(i)).collect(),

        Annotatable::TraitItem(it) => {
            expand_trait_item(it.unwrap(), fld).into_iter().
                map(|it| Annotatable::TraitItem(P(it))).collect()
        }

        Annotatable::ImplItem(ii) => {
            expand_impl_item(ii.unwrap(), fld).into_iter().
                map(|ii| Annotatable::ImplItem(P(ii))).collect()
        }
    }
}

fn expand_annotatable(mut item: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    let mut multi_modifier = None;
    item = item.map_attrs(|mut attrs| {
        for i in 0..attrs.len() {
            if let Some(extension) = fld.cx.syntax_env.find(intern(&attrs[i].name())) {
                match *extension {
                    MultiModifier(..) | MultiDecorator(..) => {
                        multi_modifier = Some((attrs.remove(i), extension));
                        break;
                    }
                    _ => {}
                }
            }
        }
        attrs
    });

    match multi_modifier {
        None => expand_multi_modified(item, fld),
        Some((attr, extension)) => {
            attr::mark_used(&attr);
            fld.cx.bt_push(ExpnInfo {
                call_site: attr.span,
                callee: NameAndSpan {
                    format: MacroAttribute(intern(&attr.name())),
                    span: Some(attr.span),
                    allow_internal_unstable: false,
                }
            });

            let modified = match *extension {
                MultiModifier(ref mac) => mac.expand(fld.cx, attr.span, &attr.node.value, item),
                MultiDecorator(ref mac) => {
                    let mut items = Vec::new();
                    mac.expand(fld.cx, attr.span, &attr.node.value, &item,
                               &mut |item| items.push(item));
                    items.push(item);
                    items
                }
                _ => unreachable!(),
            };

            fld.cx.bt_pop();
            let configured = modified.into_iter().flat_map(|it| {
                it.fold_with(&mut fld.strip_unconfigured())
            }).collect::<SmallVector<_>>();

            configured.into_iter().flat_map(|it| expand_annotatable(it, fld)).collect()
        }
    }
}

fn expand_impl_item(ii: ast::ImplItem, fld: &mut MacroExpander)
                 -> SmallVector<ast::ImplItem> {
    match ii.node {
        ast::ImplItemKind::Macro(mac) => {
            let invoc = fld.new_invoc(mac, ii.attrs, ii.span, ExpansionKind::ImplItems);
            expand_mac_invoc(invoc, fld).make_impl_items()
        }
        _ => fold::noop_fold_impl_item(ii, fld)
    }
}

fn expand_trait_item(ti: ast::TraitItem, fld: &mut MacroExpander)
                     -> SmallVector<ast::TraitItem> {
    match ti.node {
        ast::TraitItemKind::Macro(mac) => {
            let invoc = fld.new_invoc(mac, ti.attrs, ti.span, ExpansionKind::TraitItems);
            expand_mac_invoc(invoc, fld).make_trait_items()
        }
        _ => fold::noop_fold_trait_item(ti, fld)
    }
}

pub fn expand_type(t: P<ast::Ty>, fld: &mut MacroExpander) -> P<ast::Ty> {
    let t = match t.node {
        ast::TyKind::Mac(_) => t.unwrap(),
        _ => return fold::noop_fold_ty(t, fld),
    };

    match t.node {
        ast::TyKind::Mac(mac) => {
            let invoc = fld.new_invoc(mac, Vec::new(), t.span, ExpansionKind::Ty);
            expand_mac_invoc(invoc, fld).make_ty()
        }
        _ => unreachable!(),
    }
}

/// A tree-folder that performs macro expansion
pub struct MacroExpander<'a, 'b:'a> {
    pub cx: &'a mut ExtCtxt<'b>,
    pub single_step: bool,
    pub keep_macs: bool,
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>,
               single_step: bool,
               keep_macs: bool) -> MacroExpander<'a, 'b> {
        MacroExpander {
            cx: cx,
            single_step: single_step,
            keep_macs: keep_macs
        }
    }

    fn strip_unconfigured(&mut self) -> StripUnconfigured {
        StripUnconfigured {
            config: &self.cx.cfg,
            should_test: self.cx.ecfg.should_test,
            sess: self.cx.parse_sess,
            features: self.cx.ecfg.features,
        }
    }

    fn load_macros(&mut self, node: &Expansion) {
        struct MacroLoadingVisitor<'a, 'b: 'a>{
            cx: &'a mut ExtCtxt<'b>,
            at_crate_root: bool,
        }

        impl<'a, 'b> Visitor for MacroLoadingVisitor<'a, 'b> {
            fn visit_mac(&mut self, _: &ast::Mac) {}
            fn visit_item(&mut self, item: &ast::Item) {
                if let ast::ItemKind::ExternCrate(..) = item.node {
                    // We need to error on `#[macro_use] extern crate` when it isn't at the
                    // crate root, because `$crate` won't work properly.
                    for def in self.cx.loader.load_crate(item, self.at_crate_root) {
                        match def {
                            LoadedMacro::Def(def) => self.cx.insert_macro(def),
                            LoadedMacro::CustomDerive(name, ext) => {
                                self.cx.insert_custom_derive(&name, ext, item.span);
                            }
                        }
                    }
                } else {
                    let at_crate_root = ::std::mem::replace(&mut self.at_crate_root, false);
                    visit::walk_item(self, item);
                    self.at_crate_root = at_crate_root;
                }
            }
            fn visit_block(&mut self, block: &ast::Block) {
                let at_crate_root = ::std::mem::replace(&mut self.at_crate_root, false);
                visit::walk_block(self, block);
                self.at_crate_root = at_crate_root;
            }
        }

        node.visit_with(&mut MacroLoadingVisitor {
            at_crate_root: self.cx.syntax_env.is_crate_root(),
            cx: self.cx,
        });
    }

    fn new_invoc(&self, mac: ast::Mac, attrs: Vec<ast::Attribute>, span: Span, kind: ExpansionKind)
                 -> Invocation {
        let mark = Mark::fresh();
        Invocation { span: span, attrs: attrs, mac: mac, mark: mark, kind: kind, ident: None }
    }

    fn with_exts_frame<T, F: FnOnce(&mut Self) -> T>(&mut self, macros_escape: bool, f: F) -> T {
        self.cx.syntax_env.push_frame();
        self.cx.syntax_env.info().macros_escape = macros_escape;
        let result = f(self);
        self.cx.syntax_env.pop_frame();
        result
    }
}

impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
    fn fold_crate(&mut self, c: Crate) -> Crate {
        let mut directory = PathBuf::from(self.cx.parse_sess.codemap().span_to_filename(c.span));
        directory.pop();
        self.cx.directory = directory;
        noop_fold_crate(c, self)
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        expr.and_then(|expr| expand_expr(expr, self))
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        expr.and_then(|expr| match expr.node {
            ast::ExprKind::Mac(mac) => {
                let invoc =
                    self.new_invoc(mac, expr.attrs.into(), expr.span, ExpansionKind::OptExpr);
                expand_mac_invoc(invoc, self).make_opt_expr()
            }
            _ => Some(expand_expr(expr, self)),
        })
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        expand_pat(pat, self)
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        use std::mem::replace;
        let result;
        if let ast::ItemKind::Mod(ast::Mod { inner, .. }) = item.node {
            if item.span.contains(inner) {
                let directory = self.cx.directory.clone();
                self.cx.directory.push(&*{
                    ::attr::first_attr_value_str_by_name(&item.attrs, "path")
                        .unwrap_or(item.ident.name.as_str())
                });
                result = expand_item(item, self);
                self.cx.directory = directory;
            } else {
                let mut directory = match inner {
                    syntax_pos::DUMMY_SP => PathBuf::new(),
                    _ => PathBuf::from(self.cx.parse_sess.codemap().span_to_filename(inner)),
                };
                directory.pop();
                let directory = replace(&mut self.cx.directory, directory);
                result = expand_item(item, self);
                self.cx.directory = directory;
            }
        } else {
            result = expand_item(item, self);
        }
        result
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        expand_stmt(stmt, self)
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let was_in_block = ::std::mem::replace(&mut self.cx.in_block, true);
        let result = self.with_exts_frame(false, |this| noop_fold_block(block, this));
        self.cx.in_block = was_in_block;
        result
    }

    fn fold_trait_item(&mut self, i: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        expand_annotatable(Annotatable::TraitItem(P(i)), self)
            .into_iter().map(|i| i.expect_trait_item()).collect()
    }

    fn fold_impl_item(&mut self, i: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        expand_annotatable(Annotatable::ImplItem(P(i)), self)
            .into_iter().map(|i| i.expect_impl_item()).collect()
    }

    fn fold_ty(&mut self, ty: P<ast::Ty>) -> P<ast::Ty> {
        expand_type(ty, self)
    }
}

fn macro_scope_placeholder() -> Expansion {
    Expansion::Items(SmallVector::one(P(ast::Item {
        ident: keywords::Invalid.ident(),
        attrs: Vec::new(),
        id: ast::DUMMY_NODE_ID,
        node: ast::ItemKind::Mac(dummy_spanned(ast::Mac_ {
            path: ast::Path { span: syntax_pos::DUMMY_SP, global: false, segments: Vec::new() },
            tts: Vec::new(),
        })),
        vis: ast::Visibility::Inherited,
        span: syntax_pos::DUMMY_SP,
    })))
}

fn reconstruct_macro_rules(def: &ast::MacroDef, path: &ast::Path) -> Expansion {
    Expansion::Items(SmallVector::one(P(ast::Item {
        ident: def.ident,
        attrs: def.attrs.clone(),
        id: ast::DUMMY_NODE_ID,
        node: ast::ItemKind::Mac(ast::Mac {
            span: def.span,
            node: ast::Mac_ {
                path: path.clone(),
                tts: def.body.clone(),
            }
        }),
        vis: ast::Visibility::Inherited,
        span: def.span,
    })))
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: String,
    pub features: Option<&'feat Features>,
    pub recursion_limit: usize,
    pub trace_mac: bool,
    pub should_test: bool, // If false, strip `#[test]` nodes
}

macro_rules! feature_tests {
    ($( fn $getter:ident = $field:ident, )*) => {
        $(
            pub fn $getter(&self) -> bool {
                match self.features {
                    Some(&Features { $field: true, .. }) => true,
                    _ => false,
                }
            }
        )*
    }
}

impl<'feat> ExpansionConfig<'feat> {
    pub fn default(crate_name: String) -> ExpansionConfig<'static> {
        ExpansionConfig {
            crate_name: crate_name,
            features: None,
            recursion_limit: 64,
            trace_mac: false,
            should_test: false,
        }
    }

    feature_tests! {
        fn enable_quotes = quote,
        fn enable_asm = asm,
        fn enable_log_syntax = log_syntax,
        fn enable_concat_idents = concat_idents,
        fn enable_trace_macros = trace_macros,
        fn enable_allow_internal_unstable = allow_internal_unstable,
        fn enable_custom_derive = custom_derive,
        fn enable_pushpop_unsafe = pushpop_unsafe,
        fn enable_rustc_macro = rustc_macro,
    }
}

pub fn expand_crate(cx: &mut ExtCtxt,
                    user_exts: Vec<NamedSyntaxExtension>,
                    c: Crate) -> Crate {
    let mut expander = MacroExpander::new(cx, false, false);
    expand_crate_with_expander(&mut expander, user_exts, c)
}

// Expands crate using supplied MacroExpander - allows for
// non-standard expansion behaviour (e.g. step-wise).
pub fn expand_crate_with_expander(expander: &mut MacroExpander,
                                  user_exts: Vec<NamedSyntaxExtension>,
                                  mut c: Crate) -> Crate {
    if std_inject::no_core(&c) {
        expander.cx.crate_root = None;
    } else if std_inject::no_std(&c) {
        expander.cx.crate_root = Some("core");
    } else {
        expander.cx.crate_root = Some("std");
    }

    // User extensions must be added before expander.load_macros is called,
    // so that macros from external crates shadow user defined extensions.
    for (name, extension) in user_exts {
        expander.cx.syntax_env.insert(name, extension);
    }

    let items = Expansion::Items(SmallVector::many(c.module.items));
    let configured = items.fold_with(&mut expander.strip_unconfigured());
    expander.load_macros(&configured);
    c.module.items = configured.make_items().into();

    let err_count = expander.cx.parse_sess.span_diagnostic.err_count();
    let mut ret = expander.fold_crate(c);
    ret.exported_macros = expander.cx.exported_macros.clone();

    if expander.cx.parse_sess.span_diagnostic.err_count() > err_count {
        expander.cx.parse_sess.span_diagnostic.abort_if_errors();
    }

    ret
}

// A Marker adds the given mark to the syntax context and
// sets spans' `expn_id` to the given expn_id (unless it is `None`).
struct Marker { mark: Mark, expn_id: Option<ExpnId> }

impl Folder for Marker {
    fn fold_ident(&mut self, mut ident: Ident) -> Ident {
        ident.ctxt = ident.ctxt.apply_mark(self.mark);
        ident
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        noop_fold_mac(mac, self)
    }

    fn new_span(&mut self, mut span: Span) -> Span {
        if let Some(expn_id) = self.expn_id {
            span.expn_id = expn_id;
        }
        span
    }
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts: &[TokenTree], m: Mark) -> Vec<TokenTree> {
    noop_fold_tts(tts, &mut Marker{mark:m, expn_id: None})
}


#[cfg(test)]
mod tests {
    use super::{expand_crate, ExpansionConfig};
    use ast;
    use ext::base::{ExtCtxt, DummyMacroLoader};
    use parse;
    use util::parser_testing::{string_to_parser};
    use visit;
    use visit::Visitor;

    // a visitor that extracts the paths
    // from a given thingy and puts them in a mutable
    // array (passed in to the traversal)
    #[derive(Clone)]
    struct PathExprFinderContext {
        path_accumulator: Vec<ast::Path> ,
    }

    impl Visitor for PathExprFinderContext {
        fn visit_expr(&mut self, expr: &ast::Expr) {
            if let ast::ExprKind::Path(None, ref p) = expr.node {
                self.path_accumulator.push(p.clone());
            }
            visit::walk_expr(self, expr);
        }
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    fn test_ecfg() -> ExpansionConfig<'static> {
        ExpansionConfig::default("test".to_string())
    }

    // make sure that macros can't escape fns
    #[should_panic]
    #[test] fn macros_cant_escape_fns_test () {
        let src = "fn bogus() {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        // should fail:
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast);
    }

    // make sure that macros can't escape modules
    #[should_panic]
    #[test] fn macros_cant_escape_mods_test () {
        let src = "mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast);
    }

    // macro_use modules should allow macros to escape
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = "#[macro_use] mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast);
    }

    fn expand_crate_str(crate_str: String) -> ast::Crate {
        let ps = parse::ParseSess::new();
        let crate_ast = panictry!(string_to_parser(&ps, crate_str).parse_crate_mod());
        // the cfg argument actually does matter, here...
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&ps, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast)
    }

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(
            "macro_rules! m((a)=>(13)) ;fn main(){m!(a);}".to_string());
    }

    // should be able to use a bound identifier as a literal in a macro definition:
    #[test] fn self_macro_parsing(){
        expand_crate_str(
            "macro_rules! foo ((zz) => (287;));
            fn f(zz: i32) {foo!(zz);}".to_string()
            );
    }

    // create a really evil test case where a $x appears inside a binding of $x
    // but *shouldn't* bind because it was inserted by a different macro....
    // can't write this test case until we have macro-generating macros.
}
