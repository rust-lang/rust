// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, Block, Ident, NodeId, PatKind, Path};
use ast::{MacStmtStyle, StmtKind, ItemKind};
use attr::{self, HasAttrs};
use codemap::{ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
use config::{is_test_or_bench, StripUnconfigured};
use errors::FatalError;
use ext::base::*;
use ext::derive::{add_derived_markers, collect_derives};
use ext::hygiene::Mark;
use ext::placeholders::{placeholder, PlaceholderExpander};
use feature_gate::{self, Features, is_builtin_attr};
use fold;
use fold::*;
use parse::{filemap_to_stream, ParseSess, DirectoryOwnership, PResult, token};
use parse::parser::Parser;
use print::pprust;
use ptr::P;
use std_inject;
use symbol::Symbol;
use symbol::keywords;
use syntax_pos::{Span, DUMMY_SP};
use tokenstream::TokenStream;
use util::small_vector::SmallVector;
use visit::Visitor;

use std::collections::HashMap;
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;

macro_rules! expansions {
    ($($kind:ident: $ty:ty [$($vec:ident, $ty_elt:ty)*], $kind_name:expr, .$make:ident,
            $(.$fold:ident)*  $(lift .$fold_elt:ident)*,
            $(.$visit:ident)*  $(lift .$visit_elt:ident)*;)*) => {
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub enum ExpansionKind { OptExpr, $( $kind, )*  }
        pub enum Expansion { OptExpr(Option<P<ast::Expr>>), $( $kind($ty), )* }

        impl ExpansionKind {
            pub fn name(self) -> &'static str {
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

            pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) {
                match *self {
                    Expansion::OptExpr(Some(ref expr)) => visitor.visit_expr(expr),
                    Expansion::OptExpr(None) => {}
                    $($( Expansion::$kind(ref ast) => visitor.$visit(ast), )*)*
                    $($( Expansion::$kind(ref ast) => for ast in &ast[..] {
                        visitor.$visit_elt(ast);
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

        impl<'a> MacResult for ::ext::tt::macro_rules::ParserAnyMacro<'a> {
            $(fn $make(self: Box<::ext::tt::macro_rules::ParserAnyMacro<'a>>) -> Option<$ty> {
                Some(self.make(ExpansionKind::$kind).$make())
            })*
        }
    }
}

expansions! {
    Expr: P<ast::Expr> [], "expression", .make_expr, .fold_expr, .visit_expr;
    Pat: P<ast::Pat>   [], "pattern",    .make_pat,  .fold_pat,  .visit_pat;
    Ty: P<ast::Ty>     [], "type",       .make_ty,   .fold_ty,   .visit_ty;
    Stmts: SmallVector<ast::Stmt> [SmallVector, ast::Stmt],
        "statement",  .make_stmts,       lift .fold_stmt, lift .visit_stmt;
    Items: SmallVector<P<ast::Item>> [SmallVector, P<ast::Item>],
        "item",       .make_items,       lift .fold_item, lift .visit_item;
    TraitItems: SmallVector<ast::TraitItem> [SmallVector, ast::TraitItem],
        "trait item", .make_trait_items, lift .fold_trait_item, lift .visit_trait_item;
    ImplItems: SmallVector<ast::ImplItem> [SmallVector, ast::ImplItem],
        "impl item",  .make_impl_items,  lift .fold_impl_item,  lift .visit_impl_item;
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
    pub kind: InvocationKind,
    expansion_kind: ExpansionKind,
    pub expansion_data: ExpansionData,
}

pub enum InvocationKind {
    Bang {
        mac: ast::Mac,
        ident: Option<Ident>,
        span: Span,
    },
    Attr {
        attr: Option<ast::Attribute>,
        traits: Vec<Path>,
        item: Annotatable,
    },
    Derive {
        path: Path,
        item: Annotatable,
    },
}

impl Invocation {
    fn span(&self) -> Span {
        match self.kind {
            InvocationKind::Bang { span, .. } => span,
            InvocationKind::Attr { attr: Some(ref attr), .. } => attr.span,
            InvocationKind::Attr { attr: None, .. } => DUMMY_SP,
            InvocationKind::Derive { ref path, .. } => path.span,
        }
    }
}

pub struct MacroExpander<'a, 'b:'a> {
    pub cx: &'a mut ExtCtxt<'b>,
    monotonic: bool, // c.f. `cx.monotonic_expander()`
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>, monotonic: bool) -> Self {
        MacroExpander { cx: cx, monotonic: monotonic }
    }

    pub fn expand_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        self.cx.crate_root = std_inject::injected_crate_name(&krate);
        let mut module = ModuleData {
            mod_path: vec![Ident::from_str(&self.cx.ecfg.crate_name)],
            directory: PathBuf::from(self.cx.codemap().span_to_filename(krate.span)),
        };
        module.directory.pop();
        self.cx.current_expansion.module = Rc::new(module);

        let orig_mod_span = krate.module.inner;

        let krate_item = Expansion::Items(SmallVector::one(P(ast::Item {
            attrs: krate.attrs,
            span: krate.span,
            node: ast::ItemKind::Mod(krate.module),
            ident: keywords::Invalid.ident(),
            id: ast::DUMMY_NODE_ID,
            vis: ast::Visibility::Public,
        })));

        match self.expand(krate_item).make_items().pop().map(P::unwrap) {
            Some(ast::Item { attrs, node: ast::ItemKind::Mod(module), .. }) => {
                krate.attrs = attrs;
                krate.module = module;
            },
            None => {
                // Resolution failed so we return an empty expansion
                krate.attrs = vec![];
                krate.module = ast::Mod {
                    inner: orig_mod_span,
                    items: vec![],
                };
            },
            _ => unreachable!(),
        };
        self.cx.trace_macros_diag();
        krate
    }

    // Fully expand all the invocations in `expansion`.
    fn expand(&mut self, expansion: Expansion) -> Expansion {
        let orig_expansion_data = self.cx.current_expansion.clone();
        self.cx.current_expansion.depth = 0;

        let (expansion, mut invocations) = self.collect_invocations(expansion, &[]);
        self.resolve_imports();
        invocations.reverse();

        let mut expansions = Vec::new();
        let mut derives = HashMap::new();
        let mut undetermined_invocations = Vec::new();
        let (mut progress, mut force) = (false, !self.monotonic);
        loop {
            let mut invoc = if let Some(invoc) = invocations.pop() {
                invoc
            } else {
                self.resolve_imports();
                if undetermined_invocations.is_empty() { break }
                invocations = mem::replace(&mut undetermined_invocations, Vec::new());
                force = !mem::replace(&mut progress, false);
                continue
            };

            let scope =
                if self.monotonic { invoc.expansion_data.mark } else { orig_expansion_data.mark };
            let ext = match self.cx.resolver.resolve_invoc(&mut invoc, scope, force) {
                Ok(ext) => Some(ext),
                Err(Determinacy::Determined) => None,
                Err(Determinacy::Undetermined) => {
                    undetermined_invocations.push(invoc);
                    continue
                }
            };

            progress = true;
            let ExpansionData { depth, mark, .. } = invoc.expansion_data;
            self.cx.current_expansion = invoc.expansion_data.clone();

            self.cx.current_expansion.mark = scope;
            // FIXME(jseyfried): Refactor out the following logic
            let (expansion, new_invocations) = if let Some(ext) = ext {
                if let Some(ext) = ext {
                    let expansion = self.expand_invoc(invoc, ext);
                    self.collect_invocations(expansion, &[])
                } else if let InvocationKind::Attr { attr: None, traits, item } = invoc.kind {
                    let item = item
                        .map_attrs(|mut attrs| { attrs.retain(|a| a.path != "derive"); attrs });
                    let item_with_markers =
                        add_derived_markers(&mut self.cx, item.span(), &traits, item.clone());
                    let derives = derives.entry(invoc.expansion_data.mark).or_insert_with(Vec::new);

                    for path in &traits {
                        let mark = Mark::fresh(self.cx.current_expansion.mark);
                        derives.push(mark);
                        let item = match self.cx.resolver.resolve_macro(
                                Mark::root(), path, MacroKind::Derive, false) {
                            Ok(ext) => match *ext {
                                SyntaxExtension::BuiltinDerive(..) => item_with_markers.clone(),
                                _ => item.clone(),
                            },
                            _ => item.clone(),
                        };
                        invocations.push(Invocation {
                            kind: InvocationKind::Derive { path: path.clone(), item: item },
                            expansion_kind: invoc.expansion_kind,
                            expansion_data: ExpansionData {
                                mark: mark,
                                ..invoc.expansion_data.clone()
                            },
                        });
                    }
                    let expansion = invoc.expansion_kind
                        .expect_from_annotatables(::std::iter::once(item_with_markers));
                    self.collect_invocations(expansion, derives)
                } else {
                    unreachable!()
                }
            } else {
                self.collect_invocations(invoc.expansion_kind.dummy(invoc.span()), &[])
            };

            if expansions.len() < depth {
                expansions.push(Vec::new());
            }
            expansions[depth - 1].push((mark, expansion));
            if !self.cx.ecfg.single_step {
                invocations.extend(new_invocations.into_iter().rev());
            }
        }

        self.cx.current_expansion = orig_expansion_data;

        let mut placeholder_expander = PlaceholderExpander::new(self.cx, self.monotonic);
        while let Some(expansions) = expansions.pop() {
            for (mark, expansion) in expansions.into_iter().rev() {
                let derives = derives.remove(&mark).unwrap_or_else(Vec::new);
                placeholder_expander.add(NodeId::placeholder_from_mark(mark), expansion, derives);
            }
        }

        expansion.fold_with(&mut placeholder_expander)
    }

    fn resolve_imports(&mut self) {
        if self.monotonic {
            let err_count = self.cx.parse_sess.span_diagnostic.err_count();
            self.cx.resolver.resolve_imports();
            self.cx.resolve_err_count += self.cx.parse_sess.span_diagnostic.err_count() - err_count;
        }
    }

    fn collect_invocations(&mut self, expansion: Expansion, derives: &[Mark])
                           -> (Expansion, Vec<Invocation>) {
        let result = {
            let mut collector = InvocationCollector {
                cfg: StripUnconfigured {
                    should_test: self.cx.ecfg.should_test,
                    sess: self.cx.parse_sess,
                    features: self.cx.ecfg.features,
                },
                cx: self.cx,
                invocations: Vec::new(),
                monotonic: self.monotonic,
            };
            (expansion.fold_with(&mut collector), collector.invocations)
        };

        if self.monotonic {
            let err_count = self.cx.parse_sess.span_diagnostic.err_count();
            let mark = self.cx.current_expansion.mark;
            self.cx.resolver.visit_expansion(mark, &result.0, derives);
            self.cx.resolve_err_count += self.cx.parse_sess.span_diagnostic.err_count() - err_count;
        }

        result
    }

    fn expand_invoc(&mut self, invoc: Invocation, ext: Rc<SyntaxExtension>) -> Expansion {
        let result = match invoc.kind {
            InvocationKind::Bang { .. } => self.expand_bang_invoc(invoc, ext),
            InvocationKind::Attr { .. } => self.expand_attr_invoc(invoc, ext),
            InvocationKind::Derive { .. } => self.expand_derive_invoc(invoc, ext),
        };

        if self.cx.current_expansion.depth > self.cx.ecfg.recursion_limit {
            let info = self.cx.current_expansion.mark.expn_info().unwrap();
            let suggested_limit = self.cx.ecfg.recursion_limit * 2;
            let mut err = self.cx.struct_span_fatal(info.call_site,
                &format!("recursion limit reached while expanding the macro `{}`",
                         info.callee.name()));
            err.help(&format!(
                "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                suggested_limit));
            err.emit();
            panic!(FatalError);
        }

        result
    }

    fn expand_attr_invoc(&mut self, invoc: Invocation, ext: Rc<SyntaxExtension>) -> Expansion {
        let Invocation { expansion_kind: kind, .. } = invoc;
        let (attr, item) = match invoc.kind {
            InvocationKind::Attr { attr, item, .. } => (attr.unwrap(), item),
            _ => unreachable!(),
        };

        attr::mark_used(&attr);
        invoc.expansion_data.mark.set_expn_info(ExpnInfo {
            call_site: attr.span,
            callee: NameAndSpan {
                format: MacroAttribute(Symbol::intern(&format!("{}", attr.path))),
                span: None,
                allow_internal_unstable: false,
            }
        });

        match *ext {
            MultiModifier(ref mac) => {
                let meta = panictry!(attr.parse_meta(self.cx.parse_sess));
                let item = mac.expand(self.cx, attr.span, &meta, item);
                kind.expect_from_annotatables(item)
            }
            MultiDecorator(ref mac) => {
                let mut items = Vec::new();
                let meta = panictry!(attr.parse_meta(self.cx.parse_sess));
                mac.expand(self.cx, attr.span, &meta, &item, &mut |item| items.push(item));
                items.push(item);
                kind.expect_from_annotatables(items)
            }
            SyntaxExtension::AttrProcMacro(ref mac) => {
                let item_toks = stream_for_item(&item, self.cx.parse_sess);

                let span = Span { ctxt: self.cx.backtrace(), ..attr.span };
                let tok_result = mac.expand(self.cx, attr.span, attr.tokens, item_toks);
                self.parse_expansion(tok_result, kind, &attr.path, span)
            }
            SyntaxExtension::ProcMacroDerive(..) | SyntaxExtension::BuiltinDerive(..) => {
                self.cx.span_err(attr.span, &format!("`{}` is a derive mode", attr.path));
                kind.dummy(attr.span)
            }
            _ => {
                let msg = &format!("macro `{}` may not be used in attributes", attr.path);
                self.cx.span_err(attr.span, msg);
                kind.dummy(attr.span)
            }
        }
    }

    /// Expand a macro invocation. Returns the result of expansion.
    fn expand_bang_invoc(&mut self, invoc: Invocation, ext: Rc<SyntaxExtension>) -> Expansion {
        let (mark, kind) = (invoc.expansion_data.mark, invoc.expansion_kind);
        let (mac, ident, span) = match invoc.kind {
            InvocationKind::Bang { mac, ident, span } => (mac, ident, span),
            _ => unreachable!(),
        };
        let path = &mac.node.path;

        let ident = ident.unwrap_or_else(|| keywords::Invalid.ident());
        let validate_and_set_expn_info = |def_site_span, allow_internal_unstable| {
            if ident.name != keywords::Invalid.name() {
                return Err(format!("macro {}! expects no ident argument, given '{}'", path, ident));
            }
            mark.set_expn_info(ExpnInfo {
                call_site: span,
                callee: NameAndSpan {
                    format: MacroBang(Symbol::intern(&format!("{}", path))),
                    span: def_site_span,
                    allow_internal_unstable: allow_internal_unstable,
                },
            });
            Ok(())
        };

        let marked_tts = noop_fold_tts(mac.node.stream(), &mut Marker(mark));
        let opt_expanded = match *ext {
            SyntaxExtension::DeclMacro(ref expand, def_site_span) => {
                if let Err(msg) = validate_and_set_expn_info(def_site_span, false) {
                    self.cx.span_err(path.span, &msg);
                    return kind.dummy(span);
                }
                kind.make_from(expand.expand(self.cx, span, marked_tts))
            }

            NormalTT(ref expandfun, def_info, allow_internal_unstable) => {
                if let Err(msg) = validate_and_set_expn_info(def_info.map(|(_, s)| s),
                                                             allow_internal_unstable) {
                    self.cx.span_err(path.span, &msg);
                    return kind.dummy(span);
                }
                kind.make_from(expandfun.expand(self.cx, span, marked_tts))
            }

            IdentTT(ref expander, tt_span, allow_internal_unstable) => {
                if ident.name == keywords::Invalid.name() {
                    self.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", path));
                    return kind.dummy(span);
                };

                invoc.expansion_data.mark.set_expn_info(ExpnInfo {
                    call_site: span,
                    callee: NameAndSpan {
                        format: MacroBang(Symbol::intern(&format!("{}", path))),
                        span: tt_span,
                        allow_internal_unstable: allow_internal_unstable,
                    }
                });

                let input: Vec<_> = marked_tts.into_trees().collect();
                kind.make_from(expander.expand(self.cx, span, ident, input))
            }

            MultiDecorator(..) | MultiModifier(..) | SyntaxExtension::AttrProcMacro(..) => {
                self.cx.span_err(path.span,
                                 &format!("`{}` can only be used in attributes", path));
                return kind.dummy(span);
            }

            SyntaxExtension::ProcMacroDerive(..) | SyntaxExtension::BuiltinDerive(..) => {
                self.cx.span_err(path.span, &format!("`{}` is a derive mode", path));
                return kind.dummy(span);
            }

            SyntaxExtension::ProcMacro(ref expandfun) => {
                if ident.name != keywords::Invalid.name() {
                    let msg =
                        format!("macro {}! expects no ident argument, given '{}'", path, ident);
                    self.cx.span_err(path.span, &msg);
                    return kind.dummy(span);
                }

                invoc.expansion_data.mark.set_expn_info(ExpnInfo {
                    call_site: span,
                    callee: NameAndSpan {
                        format: MacroBang(Symbol::intern(&format!("{}", path))),
                        // FIXME procedural macros do not have proper span info
                        // yet, when they do, we should use it here.
                        span: None,
                        // FIXME probably want to follow macro_rules macros here.
                        allow_internal_unstable: false,
                    },
                });

                let tok_result = expandfun.expand(self.cx, span, marked_tts);
                Some(self.parse_expansion(tok_result, kind, path, span))
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

        expanded.fold_with(&mut Marker(mark))
    }

    /// Expand a derive invocation. Returns the result of expansion.
    fn expand_derive_invoc(&mut self, invoc: Invocation, ext: Rc<SyntaxExtension>) -> Expansion {
        let Invocation { expansion_kind: kind, .. } = invoc;
        let (path, item) = match invoc.kind {
            InvocationKind::Derive { path, item } => (path, item),
            _ => unreachable!(),
        };

        let pretty_name = Symbol::intern(&format!("derive({})", path));
        let span = path.span;
        let attr = ast::Attribute {
            path: path, tokens: TokenStream::empty(), span: span,
            // irrelevant:
            id: ast::AttrId(0), style: ast::AttrStyle::Outer, is_sugared_doc: false,
        };

        let mut expn_info = ExpnInfo {
            call_site: span,
            callee: NameAndSpan {
                format: MacroAttribute(pretty_name),
                span: None,
                allow_internal_unstable: false,
            }
        };

        match *ext {
            SyntaxExtension::ProcMacroDerive(ref ext, _) => {
                invoc.expansion_data.mark.set_expn_info(expn_info);
                let span = Span { ctxt: self.cx.backtrace(), ..span };
                let dummy = ast::MetaItem { // FIXME(jseyfried) avoid this
                    name: keywords::Invalid.name(),
                    span: DUMMY_SP,
                    node: ast::MetaItemKind::Word,
                };
                kind.expect_from_annotatables(ext.expand(self.cx, span, &dummy, item))
            }
            SyntaxExtension::BuiltinDerive(func) => {
                expn_info.callee.allow_internal_unstable = true;
                invoc.expansion_data.mark.set_expn_info(expn_info);
                let span = Span { ctxt: self.cx.backtrace(), ..span };
                let mut items = Vec::new();
                func(self.cx, span, &attr.meta().unwrap(), &item, &mut |a| items.push(a));
                kind.expect_from_annotatables(items)
            }
            _ => {
                let msg = &format!("macro `{}` may not be used for derive attributes", attr.path);
                self.cx.span_err(span, msg);
                kind.dummy(span)
            }
        }
    }

    fn parse_expansion(&mut self, toks: TokenStream, kind: ExpansionKind, path: &Path, span: Span)
                       -> Expansion {
        let mut parser = self.cx.new_parser_from_tts(&toks.into_trees().collect::<Vec<_>>());
        let expansion = match parser.parse_expansion(kind, false) {
            Ok(expansion) => expansion,
            Err(mut err) => {
                err.emit();
                return kind.dummy(span);
            }
        };
        parser.ensure_complete_parse(path, kind.name(), span);
        // FIXME better span info
        expansion.fold_with(&mut ChangeSpan { span: span })
    }
}

impl<'a> Parser<'a> {
    pub fn parse_expansion(&mut self, kind: ExpansionKind, macro_legacy_warnings: bool)
                           -> PResult<'a, Expansion> {
        Ok(match kind {
            ExpansionKind::Items => {
                let mut items = SmallVector::new();
                while let Some(item) = self.parse_item()? {
                    items.push(item);
                }
                Expansion::Items(items)
            }
            ExpansionKind::TraitItems => {
                let mut items = SmallVector::new();
                while self.token != token::Eof {
                    items.push(self.parse_trait_item(&mut false)?);
                }
                Expansion::TraitItems(items)
            }
            ExpansionKind::ImplItems => {
                let mut items = SmallVector::new();
                while self.token != token::Eof {
                    items.push(self.parse_impl_item(&mut false)?);
                }
                Expansion::ImplItems(items)
            }
            ExpansionKind::Stmts => {
                let mut stmts = SmallVector::new();
                while self.token != token::Eof &&
                      // won't make progress on a `}`
                      self.token != token::CloseDelim(token::Brace) {
                    if let Some(stmt) = self.parse_full_stmt(macro_legacy_warnings)? {
                        stmts.push(stmt);
                    }
                }
                Expansion::Stmts(stmts)
            }
            ExpansionKind::Expr => Expansion::Expr(self.parse_expr()?),
            ExpansionKind::OptExpr => Expansion::OptExpr(Some(self.parse_expr()?)),
            ExpansionKind::Ty => Expansion::Ty(self.parse_ty()?),
            ExpansionKind::Pat => Expansion::Pat(self.parse_pat()?),
        })
    }

    pub fn ensure_complete_parse(&mut self, macro_path: &Path, kind_name: &str, span: Span) {
        if self.token != token::Eof {
            let msg = format!("macro expansion ignores token `{}` and any following",
                              self.this_token_to_string());
            let mut err = self.diagnostic().struct_span_err(self.span, &msg);
            let msg = format!("caused by the macro expansion here; the usage \
                               of `{}!` is likely invalid in {} context",
                               macro_path, kind_name);
            err.span_note(span, &msg).emit();
        }
    }
}

struct InvocationCollector<'a, 'b: 'a> {
    cx: &'a mut ExtCtxt<'b>,
    cfg: StripUnconfigured<'a>,
    invocations: Vec<Invocation>,
    monotonic: bool,
}

macro_rules! fully_configure {
    ($this:ident, $node:ident, $noop_fold:ident) => {
        match $noop_fold($node, &mut $this.cfg).pop() {
            Some(node) => node,
            None => return SmallVector::new(),
        }
    }
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn collect(&mut self, expansion_kind: ExpansionKind, kind: InvocationKind) -> Expansion {
        let mark = Mark::fresh(self.cx.current_expansion.mark);
        self.invocations.push(Invocation {
            kind: kind,
            expansion_kind: expansion_kind,
            expansion_data: ExpansionData {
                mark: mark,
                depth: self.cx.current_expansion.depth + 1,
                ..self.cx.current_expansion.clone()
            },
        });
        placeholder(expansion_kind, NodeId::placeholder_from_mark(mark))
    }

    fn collect_bang(&mut self, mac: ast::Mac, span: Span, kind: ExpansionKind) -> Expansion {
        self.collect(kind, InvocationKind::Bang { mac: mac, ident: None, span: span })
    }

    fn collect_attr(&mut self,
                    attr: Option<ast::Attribute>,
                    traits: Vec<Path>,
                    item: Annotatable,
                    kind: ExpansionKind)
                    -> Expansion {
        if !traits.is_empty() &&
           (kind == ExpansionKind::TraitItems || kind == ExpansionKind::ImplItems) {
            self.cx.span_err(traits[0].span, "`derive` can be only be applied to items");
            return kind.expect_from_annotatables(::std::iter::once(item));
        }
        self.collect(kind, InvocationKind::Attr { attr: attr, traits: traits, item: item })
    }

    // If `item` is an attr invocation, remove and return the macro attribute.
    fn classify_item<T>(&mut self, mut item: T) -> (Option<ast::Attribute>, Vec<Path>, T)
        where T: HasAttrs,
    {
        let (mut attr, mut traits) = (None, Vec::new());

        item = item.map_attrs(|mut attrs| {
            if let Some(legacy_attr_invoc) = self.cx.resolver.find_legacy_attr_invoc(&mut attrs) {
                attr = Some(legacy_attr_invoc);
                return attrs;
            }

            if self.cx.ecfg.proc_macro_enabled() {
                attr = find_attr_invoc(&mut attrs);
            }
            traits = collect_derives(&mut self.cx, &mut attrs);
            attrs
        });

        (attr, traits, item)
    }

    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }

    // Detect use of feature-gated or invalid attributes on macro invocations
    // since they will not be detected after macro expansion.
    fn check_attributes(&mut self, attrs: &[ast::Attribute]) {
        let features = self.cx.ecfg.features.unwrap();
        for attr in attrs.iter() {
            feature_gate::check_attribute(attr, self.cx.parse_sess, features);
        }
    }
}

pub fn find_attr_invoc(attrs: &mut Vec<ast::Attribute>) -> Option<ast::Attribute> {
    attrs.iter()
         .position(|a| !attr::is_known(a) && !is_builtin_attr(a))
         .map(|i| attrs.remove(i))
}

// These are pretty nasty. Ideally, we would keep the tokens around, linked from
// the AST. However, we don't so we need to create new ones. Since the item might
// have come from a macro expansion (possibly only in part), we can't use the
// existing codemap.
//
// Therefore, we must use the pretty printer (yuck) to turn the AST node into a
// string, which we then re-tokenise (double yuck), but first we have to patch
// the pretty-printed string on to the end of the existing codemap (infinity-yuck).
fn stream_for_item(item: &Annotatable, parse_sess: &ParseSess) -> TokenStream {
    let text = match *item {
        Annotatable::Item(ref i) => pprust::item_to_string(i),
        Annotatable::TraitItem(ref ti) => pprust::trait_item_to_string(ti),
        Annotatable::ImplItem(ref ii) => pprust::impl_item_to_string(ii),
    };
    string_to_stream(text, parse_sess)
}

fn string_to_stream(text: String, parse_sess: &ParseSess) -> TokenStream {
    let filename = String::from("<macro expansion>");
    filemap_to_stream(parse_sess, parse_sess.codemap().new_filemap(filename, text))
}

impl<'a, 'b> Folder for InvocationCollector<'a, 'b> {
    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        let mut expr = self.cfg.configure_expr(expr).unwrap();
        expr.node = self.cfg.configure_expr_kind(expr.node);

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.check_attributes(&expr.attrs);
            self.collect_bang(mac, expr.span, ExpansionKind::Expr).make_expr()
        } else {
            P(noop_fold_expr(expr, self))
        }
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr).unwrap();
        expr.node = self.cfg.configure_expr_kind(expr.node);

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.check_attributes(&expr.attrs);
            self.collect_bang(mac, expr.span, ExpansionKind::OptExpr).make_opt_expr()
        } else {
            Some(P(noop_fold_expr(expr, self)))
        }
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        let pat = self.cfg.configure_pat(pat);
        match pat.node {
            PatKind::Mac(_) => {}
            _ => return noop_fold_pat(pat, self),
        }

        pat.and_then(|pat| match pat.node {
            PatKind::Mac(mac) => self.collect_bang(mac, pat.span, ExpansionKind::Pat).make_pat(),
            _ => unreachable!(),
        })
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        let stmt = match self.cfg.configure_stmt(stmt) {
            Some(stmt) => stmt,
            None => return SmallVector::new(),
        };

        let (mac, style, attrs) = if let StmtKind::Mac(mac) = stmt.node {
            mac.unwrap()
        } else {
            // The placeholder expander gives ids to statements, so we avoid folding the id here.
            let ast::Stmt { id, node, span } = stmt;
            return noop_fold_stmt_kind(node, self).into_iter().map(|node| {
                ast::Stmt { id: id, node: node, span: span }
            }).collect()
        };

        self.check_attributes(&attrs);
        let mut placeholder = self.collect_bang(mac, stmt.span, ExpansionKind::Stmts).make_stmts();

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
        let old_directory_ownership = self.cx.current_expansion.directory_ownership;
        self.cx.current_expansion.directory_ownership = DirectoryOwnership::UnownedViaBlock;
        let result = noop_fold_block(block, self);
        self.cx.current_expansion.directory_ownership = old_directory_ownership;
        result
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        let item = configure!(self, item);

        let (attr, traits, mut item) = self.classify_item(item);
        if attr.is_some() || !traits.is_empty() {
            let item = Annotatable::Item(fully_configure!(self, item, noop_fold_item));
            return self.collect_attr(attr, traits, item, ExpansionKind::Items).make_items();
        }

        match item.node {
            ast::ItemKind::Mac(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.node {
                    ItemKind::Mac(mac) => {
                        self.collect(ExpansionKind::Items, InvocationKind::Bang {
                            mac: mac,
                            ident: Some(item.ident),
                            span: item.span,
                        }).make_items()
                    }
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(ast::Mod { inner, .. }) => {
                if item.ident == keywords::Invalid.ident() {
                    return noop_fold_item(item, self);
                }

                let orig_directory_ownership = self.cx.current_expansion.directory_ownership;
                let mut module = (*self.cx.current_expansion.module).clone();
                module.mod_path.push(item.ident);

                // Detect if this is an inline module (`mod m { ... }` as opposed to `mod m;`).
                // In the non-inline case, `inner` is never the dummy span (c.f. `parse_item_mod`).
                // Thus, if `inner` is the dummy span, we know the module is inline.
                let inline_module = item.span.contains(inner) || inner == DUMMY_SP;

                if inline_module {
                    if let Some(path) = attr::first_attr_value_str_by_name(&item.attrs, "path") {
                        self.cx.current_expansion.directory_ownership = DirectoryOwnership::Owned;
                        module.directory.push(&*path.as_str());
                    } else {
                        module.directory.push(&*item.ident.name.as_str());
                    }
                } else {
                    let mut path =
                        PathBuf::from(self.cx.parse_sess.codemap().span_to_filename(inner));
                    let directory_ownership = match path.file_name().unwrap().to_str() {
                        Some("mod.rs") => DirectoryOwnership::Owned,
                        _ => DirectoryOwnership::UnownedViaMod(false),
                    };
                    path.pop();
                    module.directory = path;
                    self.cx.current_expansion.directory_ownership = directory_ownership;
                }

                let orig_module =
                    mem::replace(&mut self.cx.current_expansion.module, Rc::new(module));
                let result = noop_fold_item(item, self);
                self.cx.current_expansion.module = orig_module;
                self.cx.current_expansion.directory_ownership = orig_directory_ownership;
                result
            }
            // Ensure that test functions are accessible from the test harness.
            ast::ItemKind::Fn(..) if self.cx.ecfg.should_test => {
                if item.attrs.iter().any(|attr| is_test_or_bench(attr)) {
                    item = item.map(|mut item| { item.vis = ast::Visibility::Public; item });
                }
                noop_fold_item(item, self)
            }
            _ => noop_fold_item(item, self),
        }
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        let item = configure!(self, item);

        let (attr, traits, item) = self.classify_item(item);
        if attr.is_some() || !traits.is_empty() {
            let item =
                Annotatable::TraitItem(P(fully_configure!(self, item, noop_fold_trait_item)));
            return self.collect_attr(attr, traits, item, ExpansionKind::TraitItems)
                .make_trait_items()
        }

        match item.node {
            ast::TraitItemKind::Macro(mac) => {
                let ast::TraitItem { attrs, span, .. } = item;
                self.check_attributes(&attrs);
                self.collect_bang(mac, span, ExpansionKind::TraitItems).make_trait_items()
            }
            _ => fold::noop_fold_trait_item(item, self),
        }
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        let item = configure!(self, item);

        let (attr, traits, item) = self.classify_item(item);
        if attr.is_some() || !traits.is_empty() {
            let item = Annotatable::ImplItem(P(fully_configure!(self, item, noop_fold_impl_item)));
            return self.collect_attr(attr, traits, item, ExpansionKind::ImplItems)
                .make_impl_items();
        }

        match item.node {
            ast::ImplItemKind::Macro(mac) => {
                let ast::ImplItem { attrs, span, .. } = item;
                self.check_attributes(&attrs);
                self.collect_bang(mac, span, ExpansionKind::ImplItems).make_impl_items()
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
            ast::TyKind::Mac(mac) => self.collect_bang(mac, ty.span, ExpansionKind::Ty).make_ty(),
            _ => unreachable!(),
        }
    }

    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        noop_fold_foreign_mod(self.cfg.configure_foreign_mod(foreign_mod), self)
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        match item {
            ast::ItemKind::MacroDef(..) => item,
            _ => noop_fold_item_kind(self.cfg.configure_item_kind(item), self),
        }
    }

    fn new_id(&mut self, id: ast::NodeId) -> ast::NodeId {
        if self.monotonic {
            assert_eq!(id, ast::DUMMY_NODE_ID);
            self.cx.resolver.next_node_id()
        } else {
            id
        }
    }
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: String,
    pub features: Option<&'feat Features>,
    pub recursion_limit: usize,
    pub trace_mac: bool,
    pub should_test: bool, // If false, strip `#[test]` nodes
    pub single_step: bool,
    pub keep_macs: bool,
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
            recursion_limit: 1024,
            trace_mac: false,
            should_test: false,
            single_step: false,
            keep_macs: false,
        }
    }

    feature_tests! {
        fn enable_quotes = quote,
        fn enable_asm = asm,
        fn enable_global_asm = global_asm,
        fn enable_log_syntax = log_syntax,
        fn enable_concat_idents = concat_idents,
        fn enable_trace_macros = trace_macros,
        fn enable_allow_internal_unstable = allow_internal_unstable,
        fn enable_custom_derive = custom_derive,
        fn proc_macro_enabled = proc_macro,
    }
}

// A Marker adds the given mark to the syntax context.
struct Marker(Mark);

impl Folder for Marker {
    fn fold_ident(&mut self, mut ident: Ident) -> Ident {
        ident.ctxt = ident.ctxt.apply_mark(self.0);
        ident
    }

    fn new_span(&mut self, mut span: Span) -> Span {
        span.ctxt = span.ctxt.apply_mark(self.0);
        span
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        noop_fold_mac(mac, self)
    }
}
