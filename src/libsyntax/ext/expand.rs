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
use codemap::{ExpnInfo, NameAndSpan, MacroBang, MacroAttribute, dummy_spanned, respan};
use config::{is_test_or_bench, StripUnconfigured};
use errors::FatalError;
use ext::base::*;
use ext::derive::{add_derived_markers, collect_derives};
use ext::hygiene::{Mark, SyntaxContext};
use ext::placeholders::{placeholder, PlaceholderExpander};
use feature_gate::{self, Features, GateIssue, is_builtin_attr, emit_feature_err};
use fold;
use fold::*;
use parse::{DirectoryOwnership, PResult};
use parse::token::{self, Token};
use parse::parser::Parser;
use ptr::P;
use symbol::Symbol;
use symbol::keywords;
use syntax_pos::{Span, DUMMY_SP, FileName};
use syntax_pos::hygiene::ExpnFormat;
use tokenstream::{TokenStream, TokenTree};
use util::small_vector::SmallVector;
use visit::Visitor;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::rc::Rc;
use std::path::PathBuf;

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
    fn dummy(self, span: Span) -> Option<Expansion> {
        self.make_from(DummyResult::any(span))
    }

    fn expect_from_annotatables<I: IntoIterator<Item = Annotatable>>(self, items: I) -> Expansion {
        let mut items = items.into_iter();
        match self {
            ExpansionKind::Items =>
                Expansion::Items(items.map(Annotatable::expect_item).collect()),
            ExpansionKind::ImplItems =>
                Expansion::ImplItems(items.map(Annotatable::expect_impl_item).collect()),
            ExpansionKind::TraitItems =>
                Expansion::TraitItems(items.map(Annotatable::expect_trait_item).collect()),
            ExpansionKind::Stmts => Expansion::Stmts(items.map(Annotatable::expect_stmt).collect()),
            ExpansionKind::Expr => Expansion::Expr(
                items.next().expect("expected exactly one expression").expect_expr()
            ),
            ExpansionKind::OptExpr =>
                Expansion::OptExpr(items.next().map(Annotatable::expect_expr)),
            ExpansionKind::Pat | ExpansionKind::Ty =>
                panic!("patterns and types aren't annotatable"),
        }
    }
}

fn macro_bang_format(path: &ast::Path) -> ExpnFormat {
    // We don't want to format a path using pretty-printing,
    // `format!("{}", path)`, because that tries to insert
    // line-breaks and is slow.
    let mut path_str = String::with_capacity(64);
    for (i, segment) in path.segments.iter().enumerate() {
        if i != 0 {
            path_str.push_str("::");
        }

        if segment.identifier.name != keywords::CrateRoot.name() &&
            segment.identifier.name != keywords::DollarCrate.name()
        {
            path_str.push_str(&segment.identifier.name.as_str())
        }
    }

    MacroBang(Symbol::intern(&path_str))
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
        let mut module = ModuleData {
            mod_path: vec![Ident::from_str(&self.cx.ecfg.crate_name)],
            directory: match self.cx.codemap().span_to_unmapped_path(krate.span) {
                FileName::Real(path) => path,
                other => PathBuf::from(other.to_string()),
            },
        };
        module.directory.pop();
        self.cx.root_path = module.directory.clone();
        self.cx.current_expansion.module = Rc::new(module);
        self.cx.current_expansion.crate_span = Some(krate.span);

        let orig_mod_span = krate.module.inner;

        let krate_item = Expansion::Items(SmallVector::one(P(ast::Item {
            attrs: krate.attrs,
            span: krate.span,
            node: ast::ItemKind::Mod(krate.module),
            ident: keywords::Invalid.ident(),
            id: ast::DUMMY_NODE_ID,
            vis: respan(krate.span.shrink_to_lo(), ast::VisibilityKind::Public),
            tokens: None,
        })));

        match self.expand(krate_item).make_items().pop().map(P::into_inner) {
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
                    let dummy = invoc.expansion_kind.dummy(invoc.span()).unwrap();
                    let expansion = self.expand_invoc(invoc, &*ext).unwrap_or(dummy);
                    self.collect_invocations(expansion, &[])
                } else if let InvocationKind::Attr { attr: None, traits, item } = invoc.kind {
                    if !item.derive_allowed() {
                        let attr = attr::find_by_name(item.attrs(), "derive")
                            .expect("`derive` attribute should exist");
                        let span = attr.span;
                        let mut err = self.cx.mut_span_err(span,
                                                           "`derive` may only be applied to \
                                                            structs, enums and unions");
                        if let ast::AttrStyle::Inner = attr.style {
                            let trait_list = traits.iter()
                                .map(|t| format!("{}", t)).collect::<Vec<_>>();
                            let suggestion = format!("#[derive({})]", trait_list.join(", "));
                            err.span_suggestion(span, "try an outer attribute", suggestion);
                        }
                        err.emit();
                    }

                    let item = self.fully_configure(item)
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
                                BuiltinDerive(..) => item_with_markers.clone(),
                                _ => item.clone(),
                            },
                            _ => item.clone(),
                        };
                        invocations.push(Invocation {
                            kind: InvocationKind::Derive { path: path.clone(), item: item },
                            expansion_kind: invoc.expansion_kind,
                            expansion_data: ExpansionData {
                                mark,
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
                self.collect_invocations(invoc.expansion_kind.dummy(invoc.span()).unwrap(), &[])
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

    fn fully_configure(&mut self, item: Annotatable) -> Annotatable {
        let mut cfg = StripUnconfigured {
            should_test: self.cx.ecfg.should_test,
            sess: self.cx.parse_sess,
            features: self.cx.ecfg.features,
        };
        // Since the item itself has already been configured by the InvocationCollector,
        // we know that fold result vector will contain exactly one element
        match item {
            Annotatable::Item(item) => {
                Annotatable::Item(cfg.fold_item(item).pop().unwrap())
            }
            Annotatable::TraitItem(item) => {
                Annotatable::TraitItem(item.map(|item| cfg.fold_trait_item(item).pop().unwrap()))
            }
            Annotatable::ImplItem(item) => {
                Annotatable::ImplItem(item.map(|item| cfg.fold_impl_item(item).pop().unwrap()))
            }
            Annotatable::Stmt(stmt) => {
                Annotatable::Stmt(stmt.map(|stmt| cfg.fold_stmt(stmt).pop().unwrap()))
            }
            Annotatable::Expr(expr) => {
                Annotatable::Expr(cfg.fold_expr(expr))
            }
        }
    }

    fn expand_invoc(&mut self, invoc: Invocation, ext: &SyntaxExtension) -> Option<Expansion> {
        let result = match invoc.kind {
            InvocationKind::Bang { .. } => self.expand_bang_invoc(invoc, ext)?,
            InvocationKind::Attr { .. } => self.expand_attr_invoc(invoc, ext)?,
            InvocationKind::Derive { .. } => self.expand_derive_invoc(invoc, ext)?,
        };

        if self.cx.current_expansion.depth > self.cx.ecfg.recursion_limit {
            let info = self.cx.current_expansion.mark.expn_info().unwrap();
            let suggested_limit = self.cx.ecfg.recursion_limit * 2;
            let mut err = self.cx.struct_span_err(info.call_site,
                &format!("recursion limit reached while expanding the macro `{}`",
                         info.callee.name()));
            err.help(&format!(
                "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                suggested_limit));
            err.emit();
            self.cx.trace_macros_diag();
            FatalError.raise();
        }

        Some(result)
    }

    fn expand_attr_invoc(&mut self,
                         invoc: Invocation,
                         ext: &SyntaxExtension)
                         -> Option<Expansion> {
        let Invocation { expansion_kind: kind, .. } = invoc;
        let (attr, item) = match invoc.kind {
            InvocationKind::Attr { attr, item, .. } => (attr?, item),
            _ => unreachable!(),
        };

        attr::mark_used(&attr);
        invoc.expansion_data.mark.set_expn_info(ExpnInfo {
            call_site: attr.span,
            callee: NameAndSpan {
                format: MacroAttribute(Symbol::intern(&format!("{}", attr.path))),
                span: None,
                allow_internal_unstable: false,
                allow_internal_unsafe: false,
            }
        });

        match *ext {
            MultiModifier(ref mac) => {
                let meta = attr.parse_meta(self.cx.parse_sess)
                               .map_err(|mut e| { e.emit(); }).ok()?;
                let item = mac.expand(self.cx, attr.span, &meta, item);
                Some(kind.expect_from_annotatables(item))
            }
            MultiDecorator(ref mac) => {
                let mut items = Vec::new();
                let meta = attr.parse_meta(self.cx.parse_sess)
                               .expect("derive meta should already have been parsed");
                mac.expand(self.cx, attr.span, &meta, &item, &mut |item| items.push(item));
                items.push(item);
                Some(kind.expect_from_annotatables(items))
            }
            AttrProcMacro(ref mac) => {
                let item_tok = TokenTree::Token(DUMMY_SP, Token::interpolated(match item {
                    Annotatable::Item(item) => token::NtItem(item),
                    Annotatable::TraitItem(item) => token::NtTraitItem(item.into_inner()),
                    Annotatable::ImplItem(item) => token::NtImplItem(item.into_inner()),
                    Annotatable::Stmt(stmt) => token::NtStmt(stmt.into_inner()),
                    Annotatable::Expr(expr) => token::NtExpr(expr),
                })).into();
                let tok_result = mac.expand(self.cx, attr.span, attr.tokens, item_tok);
                self.parse_expansion(tok_result, kind, &attr.path, attr.span)
            }
            ProcMacroDerive(..) | BuiltinDerive(..) => {
                self.cx.span_err(attr.span, &format!("`{}` is a derive mode", attr.path));
                self.cx.trace_macros_diag();
                kind.dummy(attr.span)
            }
            _ => {
                let msg = &format!("macro `{}` may not be used in attributes", attr.path);
                self.cx.span_err(attr.span, msg);
                self.cx.trace_macros_diag();
                kind.dummy(attr.span)
            }
        }
    }

    /// Expand a macro invocation. Returns the result of expansion.
    fn expand_bang_invoc(&mut self,
                         invoc: Invocation,
                         ext: &SyntaxExtension)
                         -> Option<Expansion> {
        let (mark, kind) = (invoc.expansion_data.mark, invoc.expansion_kind);
        let (mac, ident, span) = match invoc.kind {
            InvocationKind::Bang { mac, ident, span } => (mac, ident, span),
            _ => unreachable!(),
        };
        let path = &mac.node.path;

        let ident = ident.unwrap_or_else(|| keywords::Invalid.ident());
        let validate_and_set_expn_info = |this: &mut Self, // arg instead of capture
                                          def_site_span: Option<Span>,
                                          allow_internal_unstable,
                                          allow_internal_unsafe,
                                          // can't infer this type
                                          unstable_feature: Option<(Symbol, u32)>| {

            // feature-gate the macro invocation
            if let Some((feature, issue)) = unstable_feature {
                let crate_span = this.cx.current_expansion.crate_span.unwrap();
                // don't stability-check macros in the same crate
                // (the only time this is null is for syntax extensions registered as macros)
                if def_site_span.map_or(false, |def_span| !crate_span.contains(def_span))
                    && !span.allows_unstable() && this.cx.ecfg.features.map_or(true, |feats| {
                    // macro features will count as lib features
                    !feats.declared_lib_features.iter().any(|&(feat, _)| feat == feature)
                }) {
                    let explain = format!("macro {}! is unstable", path);
                    emit_feature_err(this.cx.parse_sess, &*feature.as_str(), span,
                                     GateIssue::Library(Some(issue)), &explain);
                    this.cx.trace_macros_diag();
                    return Err(kind.dummy(span));
                }
            }

            if ident.name != keywords::Invalid.name() {
                let msg = format!("macro {}! expects no ident argument, given '{}'", path, ident);
                this.cx.span_err(path.span, &msg);
                this.cx.trace_macros_diag();
                return Err(kind.dummy(span));
            }
            mark.set_expn_info(ExpnInfo {
                call_site: span,
                callee: NameAndSpan {
                    format: macro_bang_format(path),
                    span: def_site_span,
                    allow_internal_unstable,
                    allow_internal_unsafe,
                },
            });
            Ok(())
        };

        let opt_expanded = match *ext {
            DeclMacro(ref expand, def_span) => {
                if let Err(dummy_span) = validate_and_set_expn_info(self, def_span.map(|(_, s)| s),
                                                                    false, false, None) {
                    dummy_span
                } else {
                    kind.make_from(expand.expand(self.cx, span, mac.node.stream()))
                }
            }

            NormalTT {
                ref expander,
                def_info,
                allow_internal_unstable,
                allow_internal_unsafe,
                unstable_feature,
            } => {
                if let Err(dummy_span) = validate_and_set_expn_info(self, def_info.map(|(_, s)| s),
                                                                    allow_internal_unstable,
                                                                    allow_internal_unsafe,
                                                                    unstable_feature) {
                    dummy_span
                } else {
                    kind.make_from(expander.expand(self.cx, span, mac.node.stream()))
                }
            }

            IdentTT(ref expander, tt_span, allow_internal_unstable) => {
                if ident.name == keywords::Invalid.name() {
                    self.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", path));
                    self.cx.trace_macros_diag();
                    kind.dummy(span)
                } else {
                    invoc.expansion_data.mark.set_expn_info(ExpnInfo {
                        call_site: span,
                        callee: NameAndSpan {
                            format: macro_bang_format(path),
                            span: tt_span,
                            allow_internal_unstable,
                            allow_internal_unsafe: false,
                        }
                    });

                    let input: Vec<_> = mac.node.stream().into_trees().collect();
                    kind.make_from(expander.expand(self.cx, span, ident, input))
                }
            }

            MultiDecorator(..) | MultiModifier(..) | AttrProcMacro(..) => {
                self.cx.span_err(path.span,
                                 &format!("`{}` can only be used in attributes", path));
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }

            ProcMacroDerive(..) | BuiltinDerive(..) => {
                self.cx.span_err(path.span, &format!("`{}` is a derive mode", path));
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }

            ProcMacro(ref expandfun) => {
                if ident.name != keywords::Invalid.name() {
                    let msg =
                        format!("macro {}! expects no ident argument, given '{}'", path, ident);
                    self.cx.span_err(path.span, &msg);
                    self.cx.trace_macros_diag();
                    kind.dummy(span)
                } else {
                    invoc.expansion_data.mark.set_expn_info(ExpnInfo {
                        call_site: span,
                        callee: NameAndSpan {
                            format: macro_bang_format(path),
                            // FIXME procedural macros do not have proper span info
                            // yet, when they do, we should use it here.
                            span: None,
                            // FIXME probably want to follow macro_rules macros here.
                            allow_internal_unstable: false,
                            allow_internal_unsafe: false,
                        },
                    });

                    let tok_result = expandfun.expand(self.cx, span, mac.node.stream());
                    self.parse_expansion(tok_result, kind, path, span)
                }
            }
        };

        if opt_expanded.is_some() {
            opt_expanded
        } else {
            let msg = format!("non-{kind} macro in {kind} position: {name}",
                              name = path.segments[0].identifier.name, kind = kind.name());
            self.cx.span_err(path.span, &msg);
            self.cx.trace_macros_diag();
            kind.dummy(span)
        }
    }

    /// Expand a derive invocation. Returns the result of expansion.
    fn expand_derive_invoc(&mut self,
                           invoc: Invocation,
                           ext: &SyntaxExtension)
                           -> Option<Expansion> {
        let Invocation { expansion_kind: kind, .. } = invoc;
        let (path, item) = match invoc.kind {
            InvocationKind::Derive { path, item } => (path, item),
            _ => unreachable!(),
        };
        if !item.derive_allowed() {
            return None;
        }

        let pretty_name = Symbol::intern(&format!("derive({})", path));
        let span = path.span;
        let attr = ast::Attribute {
            path, span,
            tokens: TokenStream::empty(),
            // irrelevant:
            id: ast::AttrId(0), style: ast::AttrStyle::Outer, is_sugared_doc: false,
        };

        let mut expn_info = ExpnInfo {
            call_site: span,
            callee: NameAndSpan {
                format: MacroAttribute(pretty_name),
                span: None,
                allow_internal_unstable: false,
                allow_internal_unsafe: false,
            }
        };

        match *ext {
            ProcMacroDerive(ref ext, _) => {
                invoc.expansion_data.mark.set_expn_info(expn_info);
                let span = span.with_ctxt(self.cx.backtrace());
                let dummy = ast::MetaItem { // FIXME(jseyfried) avoid this
                    name: keywords::Invalid.name(),
                    span: DUMMY_SP,
                    node: ast::MetaItemKind::Word,
                };
                Some(kind.expect_from_annotatables(ext.expand(self.cx, span, &dummy, item)))
            }
            BuiltinDerive(func) => {
                expn_info.callee.allow_internal_unstable = true;
                invoc.expansion_data.mark.set_expn_info(expn_info);
                let span = span.with_ctxt(self.cx.backtrace());
                let mut items = Vec::new();
                func(self.cx, span, &attr.meta()?, &item, &mut |a| items.push(a));
                Some(kind.expect_from_annotatables(items))
            }
            _ => {
                let msg = &format!("macro `{}` may not be used for derive attributes", attr.path);
                self.cx.span_err(span, msg);
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }
        }
    }

    fn parse_expansion(&mut self,
                       toks: TokenStream,
                       kind: ExpansionKind,
                       path: &Path,
                       span: Span)
                       -> Option<Expansion> {
        let mut parser = self.cx.new_parser_from_tts(&toks.into_trees().collect::<Vec<_>>());
        match parser.parse_expansion(kind, false) {
            Ok(expansion) => {
                parser.ensure_complete_parse(path, kind.name(), span);
                Some(expansion)
            }
            Err(mut err) => {
                err.set_span(span);
                err.emit();
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }
        }
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
            ExpansionKind::OptExpr => {
                if self.token != token::Eof {
                    Expansion::OptExpr(Some(self.parse_expr()?))
                } else {
                    Expansion::OptExpr(None)
                }
            },
            ExpansionKind::Ty => Expansion::Ty(self.parse_ty()?),
            ExpansionKind::Pat => Expansion::Pat(self.parse_pat()?),
        })
    }

    pub fn ensure_complete_parse(&mut self, macro_path: &Path, kind_name: &str, span: Span) {
        if self.token != token::Eof {
            let msg = format!("macro expansion ignores token `{}` and any following",
                              self.this_token_to_string());
            // Avoid emitting backtrace info twice.
            let def_site_span = self.span.with_ctxt(SyntaxContext::empty());
            let mut err = self.diagnostic().struct_span_err(def_site_span, &msg);
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

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn collect(&mut self, expansion_kind: ExpansionKind, kind: InvocationKind) -> Expansion {
        let mark = Mark::fresh(self.cx.current_expansion.mark);
        self.invocations.push(Invocation {
            kind,
            expansion_kind,
            expansion_data: ExpansionData {
                mark,
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
        self.collect(kind, InvocationKind::Attr { attr, traits, item })
    }

    /// If `item` is an attr invocation, remove and return the macro attribute and derive traits.
    fn classify_item<T>(&mut self, mut item: T) -> (Option<ast::Attribute>, Vec<Path>, T)
        where T: HasAttrs,
    {
        let (mut attr, mut traits) = (None, Vec::new());

        item = item.map_attrs(|mut attrs| {
            if let Some(legacy_attr_invoc) = self.cx.resolver.find_legacy_attr_invoc(&mut attrs,
                                                                                     true) {
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

    /// Alternative of `classify_item()` that ignores `#[derive]` so invocations fallthrough
    /// to the unused-attributes lint (making it an error on statements and expressions
    /// is a breaking change)
    fn classify_nonitem<T: HasAttrs>(&mut self, mut item: T) -> (Option<ast::Attribute>, T) {
        let mut attr = None;

        item = item.map_attrs(|mut attrs| {
            if let Some(legacy_attr_invoc) = self.cx.resolver.find_legacy_attr_invoc(&mut attrs,
                                                                                     false) {
                attr = Some(legacy_attr_invoc);
                return attrs;
            }

            if self.cx.ecfg.proc_macro_enabled() {
                attr = find_attr_invoc(&mut attrs);
            }
            attrs
        });

        (attr, item)
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

            // macros are expanded before any lint passes so this warning has to be hardcoded
            if attr.path == "derive" {
                self.cx.struct_span_warn(attr.span, "`#[derive]` does nothing on macro invocations")
                    .note("this may become a hard error in a future release")
                    .emit();
            }
        }
    }

    fn check_attribute(&mut self, at: &ast::Attribute) {
        let features = self.cx.ecfg.features.unwrap();
        feature_gate::check_attribute(at, self.cx.parse_sess, features);
    }
}

pub fn find_attr_invoc(attrs: &mut Vec<ast::Attribute>) -> Option<ast::Attribute> {
    attrs.iter()
         .position(|a| !attr::is_known(a) && !is_builtin_attr(a))
         .map(|i| attrs.remove(i))
}

impl<'a, 'b> Folder for InvocationCollector<'a, 'b> {
    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        let mut expr = self.cfg.configure_expr(expr).into_inner();
        expr.node = self.cfg.configure_expr_kind(expr.node);

        // ignore derives so they remain unused
        let (attr, expr) = self.classify_nonitem(expr);

        if attr.is_some() {
            // collect the invoc regardless of whether or not attributes are permitted here
            // expansion will eat the attribute so it won't error later
            attr.as_ref().map(|a| self.cfg.maybe_emit_expr_attr_err(a));

            // ExpansionKind::Expr requires the macro to emit an expression
            return self.collect_attr(attr, vec![], Annotatable::Expr(P(expr)), ExpansionKind::Expr)
                .make_expr();
        }

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.check_attributes(&expr.attrs);
            self.collect_bang(mac, expr.span, ExpansionKind::Expr).make_expr()
        } else {
            P(noop_fold_expr(expr, self))
        }
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let mut expr = configure!(self, expr).into_inner();
        expr.node = self.cfg.configure_expr_kind(expr.node);

        // ignore derives so they remain unused
        let (attr, expr) = self.classify_nonitem(expr);

        if attr.is_some() {
            attr.as_ref().map(|a| self.cfg.maybe_emit_expr_attr_err(a));

            return self.collect_attr(attr, vec![], Annotatable::Expr(P(expr)),
                                     ExpansionKind::OptExpr)
                .make_opt_expr();
        }

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
        let mut stmt = match self.cfg.configure_stmt(stmt) {
            Some(stmt) => stmt,
            None => return SmallVector::new(),
        };

        // we'll expand attributes on expressions separately
        if !stmt.is_expr() {
            let (attr, derives, stmt_) = if stmt.is_item() {
                self.classify_item(stmt)
            } else {
                // ignore derives on non-item statements so it falls through
                // to the unused-attributes lint
                let (attr, stmt) = self.classify_nonitem(stmt);
                (attr, vec![], stmt)
            };

            if attr.is_some() || !derives.is_empty() {
                return self.collect_attr(attr, derives,
                                         Annotatable::Stmt(P(stmt_)), ExpansionKind::Stmts)
                    .make_stmts();
            }

            stmt = stmt_;
        }

        if let StmtKind::Mac(mac) = stmt.node {
            let (mac, style, attrs) = mac.into_inner();
            self.check_attributes(&attrs);
            let mut placeholder = self.collect_bang(mac, stmt.span, ExpansionKind::Stmts)
                                        .make_stmts();

            // If this is a macro invocation with a semicolon, then apply that
            // semicolon to the final statement produced by expansion.
            if style == MacStmtStyle::Semicolon {
                if let Some(stmt) = placeholder.pop() {
                    placeholder.push(stmt.add_trailing_semicolon());
                }
            }

            return placeholder;
        }

        // The placeholder expander gives ids to statements, so we avoid folding the id here.
        let ast::Stmt { id, node, span } = stmt;
        noop_fold_stmt_kind(node, self).into_iter().map(|node| {
            ast::Stmt { id, node, span }
        }).collect()

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
            let item = Annotatable::Item(item);
            return self.collect_attr(attr, traits, item, ExpansionKind::Items).make_items();
        }

        match item.node {
            ast::ItemKind::Mac(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.node {
                    ItemKind::Mac(mac) => {
                        self.collect(ExpansionKind::Items, InvocationKind::Bang {
                            mac,
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
                        self.cx.current_expansion.directory_ownership =
                            DirectoryOwnership::Owned { relative: None };
                        module.directory.push(&*path.as_str());
                    } else {
                        module.directory.push(&*item.ident.name.as_str());
                    }
                } else {
                    let path = self.cx.parse_sess.codemap().span_to_unmapped_path(inner);
                    let mut path = match path {
                        FileName::Real(path) => path,
                        other => PathBuf::from(other.to_string()),
                    };
                    let directory_ownership = match path.file_name().unwrap().to_str() {
                        Some("mod.rs") => DirectoryOwnership::Owned { relative: None },
                        Some(_) => DirectoryOwnership::Owned {
                            relative: Some(item.ident),
                        },
                        None => DirectoryOwnership::UnownedViaMod(false),
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
                    item = item.map(|mut item| {
                        item.vis = respan(item.vis.span, ast::VisibilityKind::Public);
                        item
                    });
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
            let item = Annotatable::TraitItem(P(item));
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
            let item = Annotatable::ImplItem(P(item));
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
            ast::TyKind::Mac(_) => ty.into_inner(),
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

    fn fold_attribute(&mut self, at: ast::Attribute) -> Option<ast::Attribute> {
        // turn `#[doc(include="filename")]` attributes into `#[doc(include(file="filename",
        // contents="file contents")]` attributes
        if !at.check_name("doc") {
            return noop_fold_attribute(at, self);
        }

        if let Some(list) = at.meta_item_list() {
            if !list.iter().any(|it| it.check_name("include")) {
                return noop_fold_attribute(at, self);
            }

            let mut items = vec![];

            for it in list {
                if !it.check_name("include") {
                    items.push(noop_fold_meta_list_item(it, self));
                    continue;
                }

                if let Some(file) = it.value_str() {
                    let err_count = self.cx.parse_sess.span_diagnostic.err_count();
                    self.check_attribute(&at);
                    if self.cx.parse_sess.span_diagnostic.err_count() > err_count {
                        // avoid loading the file if they haven't enabled the feature
                        return noop_fold_attribute(at, self);
                    }

                    let mut buf = vec![];
                    let filename = self.cx.root_path.join(file.to_string());

                    match File::open(&filename).and_then(|mut f| f.read_to_end(&mut buf)) {
                        Ok(..) => {}
                        Err(e) => {
                            self.cx.span_err(at.span,
                                             &format!("couldn't read {}: {}",
                                                      filename.display(),
                                                      e));
                        }
                    }

                    match String::from_utf8(buf) {
                        Ok(src) => {
                            // Add this input file to the code map to make it available as
                            // dependency information
                            self.cx.codemap().new_filemap_and_lines(&filename, &src);

                            let include_info = vec![
                                dummy_spanned(ast::NestedMetaItemKind::MetaItem(
                                        attr::mk_name_value_item_str("file".into(),
                                                                     file))),
                                dummy_spanned(ast::NestedMetaItemKind::MetaItem(
                                        attr::mk_name_value_item_str("contents".into(),
                                                                     (&*src).into()))),
                            ];

                            items.push(dummy_spanned(ast::NestedMetaItemKind::MetaItem(
                                        attr::mk_list_item("include".into(), include_info))));
                        }
                        Err(_) => {
                            self.cx.span_err(at.span,
                                             &format!("{} wasn't a utf-8 file",
                                                      filename.display()));
                        }
                    }
                } else {
                    items.push(noop_fold_meta_list_item(it, self));
                }
            }

            let meta = attr::mk_list_item("doc".into(), items);
            match at.style {
                ast::AttrStyle::Inner =>
                    Some(attr::mk_spanned_attr_inner(at.span, at.id, meta)),
                ast::AttrStyle::Outer =>
                    Some(attr::mk_spanned_attr_outer(at.span, at.id, meta)),
            }
        } else {
            noop_fold_attribute(at, self)
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
            crate_name,
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
#[derive(Debug)]
pub struct Marker(pub Mark);

impl Folder for Marker {
    fn fold_ident(&mut self, mut ident: Ident) -> Ident {
        ident.ctxt = ident.ctxt.apply_mark(self.0);
        ident
    }

    fn new_span(&mut self, span: Span) -> Span {
        span.with_ctxt(span.ctxt().apply_mark(self.0))
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        noop_fold_mac(mac, self)
    }
}
