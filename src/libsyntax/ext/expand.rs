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
use ast::{MacStmtStyle, StmtKind, ItemKind};
use ast;
use ext::hygiene::Mark;
use ext::placeholders::{self, placeholder, PlaceholderExpander};
use attr::{self, HasAttrs};
use codemap::{ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
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

use std::mem;
use std::path::PathBuf;
use std::rc::Rc;

macro_rules! expansions {
    ($($kind:ident: $ty:ty [$($vec:ident, $ty_elt:ty)*], $kind_name:expr, .$make:ident,
            $(.$fold:ident)*  $(lift .$fold_elt:ident)*;)*) => {
        #[derive(Copy, Clone)]
        pub enum ExpansionKind { OptExpr, $( $kind, )*  }
        pub enum Expansion { OptExpr(Option<P<ast::Expr>>), $( $kind($ty), )* }

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
            pub fn make_opt_expr(self) -> Option<P<ast::Expr>> {
                match self {
                    Expansion::OptExpr(expr) => expr,
                    _ => panic!("Expansion::make_* called on the wrong kind of expansion"),
                }
            }
            $( pub fn $make(self) -> $ty {
                match self {
                    Expansion::$kind(ast) => ast,
                    _ => panic!("Expansion::make_* called on the wrong kind of expansion"),
                }
            } )*

            pub fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
                use self::Expansion::*;
                match self {
                    OptExpr(expr) => OptExpr(expr.and_then(|expr| folder.fold_opt_expr(expr))),
                    $($( $kind(ast) => $kind(folder.$fold(ast)), )*)*
                    $($( $kind(ast) => {
                        $kind(ast.into_iter().flat_map(|ast| folder.$fold_elt(ast)).collect())
                    }, )*)*
                }
            }
        }

        impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
            fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
                self.expand(Expansion::OptExpr(Some(expr))).make_opt_expr()
            }
            $($(fn $fold(&mut self, node: $ty) -> $ty {
                self.expand(Expansion::$kind(node)).$make()
            })*)*
            $($(fn $fold_elt(&mut self, node: $ty_elt) -> $ty {
                self.expand(Expansion::$kind(SmallVector::one(node))).$make()
            })*)*
        }
    }
}

expansions! {
    Expr: P<ast::Expr> [], "expression", .make_expr, .fold_expr;
    Pat: P<ast::Pat>   [], "pattern",    .make_pat,  .fold_pat;
    Ty: P<ast::Ty>     [], "type",       .make_ty,   .fold_ty;
    Stmts: SmallVector<ast::Stmt> [SmallVector, ast::Stmt],
        "statement",  .make_stmts,       lift .fold_stmt;
    Items: SmallVector<P<ast::Item>> [SmallVector, P<ast::Item>],
        "item",       .make_items,       lift .fold_item;
    TraitItems: SmallVector<ast::TraitItem> [SmallVector, ast::TraitItem],
        "trait item", .make_trait_items, lift .fold_trait_item;
    ImplItems: SmallVector<ast::ImplItem> [SmallVector, ast::ImplItem],
        "impl item",  .make_impl_items,  lift .fold_impl_item;
}

impl ExpansionKind {
    fn dummy(self, span: Span) -> Expansion {
        self.make_from(DummyResult::any(span)).unwrap()
    }

    fn expect_from_annotatables<I: IntoIterator<Item = Annotatable>>(self, items: I) -> Expansion {
        let items = items.into_iter();
        match self {
            ExpansionKind::Items =>
                Expansion::Items(items.map(Annotatable::expect_item).collect()),
            ExpansionKind::ImplItems =>
                Expansion::ImplItems(items.map(Annotatable::expect_impl_item).collect()),
            ExpansionKind::TraitItems =>
                Expansion::TraitItems(items.map(Annotatable::expect_trait_item).collect()),
            _ => unreachable!(),
        }
    }
}

pub struct Invocation {
    kind: InvocationKind,
    expansion_kind: ExpansionKind,
    mark: Mark,
    module: Module,
    backtrace: ExpnId,
    depth: usize,
}

enum InvocationKind {
    Bang {
        attrs: Vec<ast::Attribute>,
        mac: ast::Mac,
        ident: Option<Ident>,
        span: Span,
    },
    Attr {
        attr: ast::Attribute,
        item: Annotatable,
    },
}

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

    fn expand_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        let err_count = self.cx.parse_sess.span_diagnostic.err_count();

        let items = Expansion::Items(SmallVector::many(krate.module.items));
        krate.module.items = self.expand(items).make_items().into();
        krate.exported_macros = self.cx.exported_macros.clone();

        if self.cx.parse_sess.span_diagnostic.err_count() > err_count {
            self.cx.parse_sess.span_diagnostic.abort_if_errors();
        }

        krate
    }

    // Fully expand all the invocations in `expansion`.
    fn expand(&mut self, expansion: Expansion) -> Expansion {
        self.cx.recursion_count = 0;
        let (expansion, mut invocations) = self.collect_invocations(expansion);
        invocations.reverse();

        let mut expansions = vec![vec![(0, expansion)]];
        while let Some(invoc) = invocations.pop() {
            let Invocation { mark, module, depth, backtrace, .. } = invoc;
            self.cx.syntax_env.current_module = module;
            self.cx.recursion_count = depth;
            self.cx.backtrace = backtrace;

            let expansion = self.expand_invoc(invoc);

            self.cx.syntax_env.current_module = module;
            self.cx.recursion_count = depth + 1;
            let (expansion, new_invocations) = self.collect_invocations(expansion);

            if expansions.len() == depth {
                expansions.push(Vec::new());
            }
            expansions[depth].push((mark.as_u32(), expansion));
            if !self.single_step {
                invocations.extend(new_invocations.into_iter().rev());
            }
        }

        let mut placeholder_expander = PlaceholderExpander::new();
        while let Some(expansions) = expansions.pop() {
            for (mark, expansion) in expansions.into_iter().rev() {
                let expansion = expansion.fold_with(&mut placeholder_expander);
                placeholder_expander.add(mark, expansion);
            }
        }

        placeholder_expander.remove(0)
    }

    fn collect_invocations(&mut self, expansion: Expansion) -> (Expansion, Vec<Invocation>) {
        let crate_config = mem::replace(&mut self.cx.cfg, Vec::new());
        let result = {
            let mut collector = InvocationCollector {
                cfg: StripUnconfigured {
                    config: &crate_config,
                    should_test: self.cx.ecfg.should_test,
                    sess: self.cx.parse_sess,
                    features: self.cx.ecfg.features,
                },
                cx: self.cx,
                invocations: Vec::new(),
            };
            (expansion.fold_with(&mut collector), collector.invocations)
        };

        self.cx.cfg = crate_config;
        result
    }

    fn expand_invoc(&mut self, invoc: Invocation) -> Expansion {
        match invoc.kind {
            InvocationKind::Bang { .. } => self.expand_bang_invoc(invoc),
            InvocationKind::Attr { .. } => self.expand_attr_invoc(invoc),
        }
    }

    fn expand_attr_invoc(&mut self, invoc: Invocation) -> Expansion {
        let Invocation { expansion_kind: kind, .. } = invoc;
        let (attr, item) = match invoc.kind {
            InvocationKind::Attr { attr, item } => (attr, item),
            _ => unreachable!(),
        };

        let extension = match self.cx.syntax_env.find(intern(&attr.name())) {
            Some(extension) => extension,
            None => unreachable!(),
        };

        attr::mark_used(&attr);
        self.cx.bt_push(ExpnInfo {
            call_site: attr.span,
            callee: NameAndSpan {
                format: MacroAttribute(intern(&attr.name())),
                span: Some(attr.span),
                allow_internal_unstable: false,
            }
        });

        match *extension {
            MultiModifier(ref mac) => {
                let item = mac.expand(self.cx, attr.span, &attr.node.value, item);
                kind.expect_from_annotatables(item)
            }
            MultiDecorator(ref mac) => {
                let mut items = Vec::new();
                mac.expand(self.cx, attr.span, &attr.node.value, &item,
                           &mut |item| items.push(item));
                items.push(item);
                kind.expect_from_annotatables(items)
            }
            _ => unreachable!(),
        }
    }

    /// Expand a macro invocation. Returns the result of expansion.
    fn expand_bang_invoc(&mut self, invoc: Invocation) -> Expansion {
        let Invocation { mark, expansion_kind: kind, .. } = invoc;
        let (attrs, mac, ident, span) = match invoc.kind {
            InvocationKind::Bang { attrs, mac, ident, span } => (attrs, mac, ident, span),
            _ => unreachable!(),
        };
        let Mac_ { path, tts, .. } = mac.node;

        // Detect use of feature-gated or invalid attributes on macro invoations
        // since they will not be detected after macro expansion.
        for attr in attrs.iter() {
            feature_gate::check_attribute(&attr, &self.cx.parse_sess.span_diagnostic,
                                          &self.cx.parse_sess.codemap(),
                                          &self.cx.ecfg.features.unwrap());
        }

        if path.segments.len() > 1 || path.global || !path.segments[0].parameters.is_empty() {
            self.cx.span_err(path.span, "expected macro name without module separators");
            return kind.dummy(span);
        }

        let extname = path.segments[0].identifier.name;
        let extension = if let Some(extension) = self.cx.syntax_env.find(extname) {
            extension
        } else {
            let mut err =
                self.cx.struct_span_err(path.span, &format!("macro undefined: '{}!'", &extname));
            self.cx.suggest_macro_name(&extname.as_str(), &mut err);
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
                    self.cx.span_err(path.span, &msg);
                    return kind.dummy(span);
                }

                self.cx.bt_push(ExpnInfo {
                    call_site: span,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: exp_span,
                        allow_internal_unstable: allow_internal_unstable,
                    },
                });

                kind.make_from(expandfun.expand(self.cx, span, &marked_tts))
            }

            IdentTT(ref expander, tt_span, allow_internal_unstable) => {
                if ident.name == keywords::Invalid.name() {
                    self.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", extname));
                    return kind.dummy(span);
                };

                self.cx.bt_push(ExpnInfo {
                    call_site: span,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: tt_span,
                        allow_internal_unstable: allow_internal_unstable,
                    }
                });

                kind.make_from(expander.expand(self.cx, span, ident, marked_tts))
            }

            MacroRulesTT => {
                if ident.name == keywords::Invalid.name() {
                    self.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", extname));
                    return kind.dummy(span);
                };

                self.cx.bt_push(ExpnInfo {
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

                self.cx.insert_macro(def.clone());

                // If keep_macs is true, expands to a MacEager::items instead.
                if self.keep_macs {
                    Some(placeholders::reconstructed_macro_rules(&def, &path))
                } else {
                    Some(placeholders::macro_scope_placeholder())
                }
            }

            MultiDecorator(..) | MultiModifier(..) => {
                self.cx.span_err(path.span,
                                 &format!("`{}` can only be used in attributes", extname));
                return kind.dummy(span);
            }
        };

        let expanded = if let Some(expanded) = opt_expanded {
            expanded
        } else {
            let msg = format!("non-{kind} macro in {kind} position: {name}",
                              name = path.segments[0].identifier.name, kind = kind.name());
            self.cx.span_err(path.span, &msg);
            return kind.dummy(span);
        };

        expanded.fold_with(&mut Marker {
            mark: mark,
            expn_id: Some(self.cx.backtrace()),
        })
    }
}

struct InvocationCollector<'a, 'b: 'a> {
    cx: &'a mut ExtCtxt<'b>,
    cfg: StripUnconfigured<'a>,
    invocations: Vec<Invocation>,
}

macro_rules! fully_configure {
    ($this:ident, $node:ident, $noop_fold:ident) => {
        match $noop_fold($node, &mut $this.cfg).pop() {
            Some(node) => node,
            None => return SmallVector::zero(),
        }
    }
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn collect(&mut self, expansion_kind: ExpansionKind, kind: InvocationKind) -> Expansion {
        let mark = Mark::fresh();
        self.invocations.push(Invocation {
            kind: kind,
            expansion_kind: expansion_kind,
            mark: mark,
            module: self.cx.syntax_env.current_module,
            backtrace: self.cx.backtrace,
            depth: self.cx.recursion_count,
        });
        placeholder(expansion_kind, mark.as_u32())
    }

    fn collect_bang(
        &mut self, mac: ast::Mac, attrs: Vec<ast::Attribute>, span: Span, kind: ExpansionKind,
    ) -> Expansion {
        self.collect(kind, InvocationKind::Bang { attrs: attrs, mac: mac, ident: None, span: span })
    }

    fn collect_attr(&mut self, attr: ast::Attribute, item: Annotatable, kind: ExpansionKind)
                    -> Expansion {
        self.collect(kind, InvocationKind::Attr { attr: attr, item: item })
    }

    // If `item` is an attr invocation, remove and return the macro attribute.
    fn classify_item<T: HasAttrs>(&self, mut item: T) -> (T, Option<ast::Attribute>) {
        let mut attr = None;
        item = item.map_attrs(|mut attrs| {
            for i in 0..attrs.len() {
                if let Some(extension) = self.cx.syntax_env.find(intern(&attrs[i].name())) {
                    match *extension {
                        MultiModifier(..) | MultiDecorator(..) => {
                            attr = Some(attrs.remove(i));
                            break;
                        }
                        _ => {}
                    }
                }
            }
            attrs
        });
        (item, attr)
    }

    // does this attribute list contain "macro_use" ?
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            let mut is_use = attr.check_name("macro_use");
            if attr.check_name("macro_escape") {
                let msg = "macro_escape is a deprecated synonym for macro_use";
                let mut err = self.cx.struct_span_warn(attr.span, msg);
                is_use = true;
                if let ast::AttrStyle::Inner = attr.node.style {
                    err.help("consider an outer attribute, #[macro_use] mod ...").emit();
                } else {
                    err.emit();
                }
            };

            if is_use {
                if !attr.is_word() {
                    self.cx.span_err(attr.span, "arguments to macro_use are not allowed here");
                }
                return true;
            }
        }
        false
    }

    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }
}

impl<'a, 'b> Folder for InvocationCollector<'a, 'b> {
    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        let mut expr = self.cfg.configure_expr(expr).unwrap();
        expr.node = self.cfg.configure_expr_kind(expr.node);

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.collect_bang(mac, expr.attrs.into(), expr.span, ExpansionKind::Expr).make_expr()
        } else {
            P(noop_fold_expr(expr, self))
        }
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr).unwrap();
        expr.node = self.cfg.configure_expr_kind(expr.node);

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.collect_bang(mac, expr.attrs.into(), expr.span, ExpansionKind::OptExpr)
                .make_opt_expr()
        } else {
            Some(P(noop_fold_expr(expr, self)))
        }
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        match pat.node {
            PatKind::Mac(_) => {}
            _ => return noop_fold_pat(pat, self),
        }

        pat.and_then(|pat| match pat.node {
            PatKind::Mac(mac) =>
                self.collect_bang(mac, Vec::new(), pat.span, ExpansionKind::Pat).make_pat(),
            _ => unreachable!(),
        })
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        let stmt = match self.cfg.configure_stmt(stmt) {
            Some(stmt) => stmt,
            None => return SmallVector::zero(),
        };

        let (mac, style, attrs) = match stmt.node {
            StmtKind::Mac(mac) => mac.unwrap(),
            _ => return noop_fold_stmt(stmt, self),
        };

        let mut placeholder =
            self.collect_bang(mac, attrs.into(), stmt.span, ExpansionKind::Stmts).make_stmts();

        // If this is a macro invocation with a semicolon, then apply that
        // semicolon to the final statement produced by expansion.
        if style == MacStmtStyle::Semicolon {
            if let Some(stmt) = placeholder.pop() {
                placeholder.push(stmt.add_trailing_semicolon());
            }
        }

        placeholder
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let paths = self.cx.syntax_env.paths();
        let module = self.cx.syntax_env.add_module(false, true, paths);
        let orig_module = mem::replace(&mut self.cx.syntax_env.current_module, module);
        let result = noop_fold_block(block, self);
        self.cx.syntax_env.current_module = orig_module;
        result
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        let item = configure!(self, item);

        let (item, attr) = self.classify_item(item);
        if let Some(attr) = attr {
            let item = Annotatable::Item(fully_configure!(self, item, noop_fold_item));
            return self.collect_attr(attr, item, ExpansionKind::Items).make_items();
        }

        match item.node {
            ast::ItemKind::Mac(..) => {
                if match item.node {
                    ItemKind::Mac(ref mac) => mac.node.path.segments.is_empty(),
                    _ => unreachable!(),
                } {
                    return SmallVector::one(item);
                }

                item.and_then(|item| match item.node {
                    ItemKind::Mac(mac) => {
                        self.collect(ExpansionKind::Items, InvocationKind::Bang {
                            mac: mac,
                            attrs: item.attrs,
                            ident: Some(item.ident),
                            span: item.span,
                        }).make_items()
                    }
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(ast::Mod { inner, .. }) => {
                let mut paths = (*self.cx.syntax_env.paths()).clone();
                paths.mod_path.push(item.ident);

                // Detect if this is an inline module (`mod m { ... }` as opposed to `mod m;`).
                // In the non-inline case, `inner` is never the dummy span (c.f. `parse_item_mod`).
                // Thus, if `inner` is the dummy span, we know the module is inline.
                let inline_module = item.span.contains(inner) || inner == syntax_pos::DUMMY_SP;

                if inline_module {
                    paths.directory.push(&*{
                        ::attr::first_attr_value_str_by_name(&item.attrs, "path")
                            .unwrap_or(item.ident.name.as_str())
                    });
                } else {
                    paths.directory =
                        PathBuf::from(self.cx.parse_sess.codemap().span_to_filename(inner));
                    paths.directory.pop();
                }

                let macro_use = self.contains_macro_use(&item.attrs);
                let in_block = self.cx.syntax_env.in_block();
                let module = self.cx.syntax_env.add_module(macro_use, in_block, Rc::new(paths));
                let module = mem::replace(&mut self.cx.syntax_env.current_module, module);
                let result = noop_fold_item(item, self);
                self.cx.syntax_env.current_module = module;
                result
            },
            ast::ItemKind::ExternCrate(..) => {
                // We need to error on `#[macro_use] extern crate` when it isn't at the
                // crate root, because `$crate` won't work properly.
                let is_crate_root = self.cx.syntax_env.is_crate_root();
                for def in self.cx.loader.load_crate(&*item, is_crate_root) {
                    match def {
                        LoadedMacro::Def(def) => self.cx.insert_macro(def),
                        LoadedMacro::CustomDerive(name, ext) => {
                            self.cx.insert_custom_derive(&name, ext, item.span);
                        }
                    }
                }
                SmallVector::one(item)
            },
            _ => noop_fold_item(item, self),
        }
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        let item = configure!(self, item);

        let (item, attr) = self.classify_item(item);
        if let Some(attr) = attr {
            let item =
                Annotatable::TraitItem(P(fully_configure!(self, item, noop_fold_trait_item)));
            return self.collect_attr(attr, item, ExpansionKind::TraitItems).make_trait_items()
        }

        match item.node {
            ast::TraitItemKind::Macro(mac) => {
                let ast::TraitItem { attrs, span, .. } = item;
                self.collect_bang(mac, attrs, span, ExpansionKind::TraitItems).make_trait_items()
            }
            _ => fold::noop_fold_trait_item(item, self),
        }
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        let item = configure!(self, item);

        let (item, attr) = self.classify_item(item);
        if let Some(attr) = attr {
            let item = Annotatable::ImplItem(P(fully_configure!(self, item, noop_fold_impl_item)));
            return self.collect_attr(attr, item, ExpansionKind::ImplItems).make_impl_items();
        }

        match item.node {
            ast::ImplItemKind::Macro(mac) => {
                let ast::ImplItem { attrs, span, .. } = item;
                self.collect_bang(mac, attrs, span, ExpansionKind::ImplItems).make_impl_items()
            }
            _ => fold::noop_fold_impl_item(item, self),
        }
    }

    fn fold_ty(&mut self, ty: P<ast::Ty>) -> P<ast::Ty> {
        let ty = match ty.node {
            ast::TyKind::Mac(_) => ty.unwrap(),
            _ => return fold::noop_fold_ty(ty, self),
        };

        match ty.node {
            ast::TyKind::Mac(mac) =>
                self.collect_bang(mac, Vec::new(), ty.span, ExpansionKind::Ty).make_ty(),
            _ => unreachable!(),
        }
    }

    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        noop_fold_foreign_mod(self.cfg.configure_foreign_mod(foreign_mod), self)
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        noop_fold_item_kind(self.cfg.configure_item_kind(item), self)
    }
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
    cx.initialize(user_exts, &c);
    cx.expander().expand_crate(c)
}

// Expands crate using supplied MacroExpander - allows for
// non-standard expansion behaviour (e.g. step-wise).
pub fn expand_crate_with_expander(expander: &mut MacroExpander,
                                  user_exts: Vec<NamedSyntaxExtension>,
                                  c: Crate) -> Crate {
    expander.cx.initialize(user_exts, &c);
    expander.expand_crate(c)
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
