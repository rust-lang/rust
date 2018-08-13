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
use source_map::{ExpnInfo, MacroBang, MacroAttribute, dummy_spanned, respan};
use config::{is_test_or_bench, StripUnconfigured};
use errors::{Applicability, FatalError};
use ext::base::*;
use ext::build::AstBuilder;
use ext::derive::{add_derived_markers, collect_derives};
use ext::hygiene::{self, Mark, SyntaxContext};
use ext::placeholders::{placeholder, PlaceholderExpander};
use feature_gate::{self, Features, GateIssue, is_builtin_attr, emit_feature_err};
use fold;
use fold::*;
use parse::{DirectoryOwnership, PResult, ParseSess};
use parse::token::{self, Token};
use parse::parser::Parser;
use ptr::P;
use OneVector;
use symbol::Symbol;
use symbol::keywords;
use syntax_pos::{Span, DUMMY_SP, FileName};
use syntax_pos::hygiene::ExpnFormat;
use tokenstream::{TokenStream, TokenTree};
use visit::{self, Visitor};

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::iter::FromIterator;
use std::{iter, mem};
use std::rc::Rc;
use std::path::PathBuf;

macro_rules! ast_fragments {
    (
        $($Kind:ident($AstTy:ty) {
            $kind_name:expr;
            // FIXME: HACK: this should be `$(one ...)?` and `$(many ...)?` but `?` macro
            // repetition was removed from 2015 edition in #51587 because of ambiguities.
            $(one fn $fold_ast:ident; fn $visit_ast:ident;)*
            $(many fn $fold_ast_elt:ident; fn $visit_ast_elt:ident;)*
            fn $make_ast:ident;
        })*
    ) => {
        /// A fragment of AST that can be produced by a single macro expansion.
        /// Can also serve as an input and intermediate result for macro expansion operations.
        pub enum AstFragment {
            OptExpr(Option<P<ast::Expr>>),
            $($Kind($AstTy),)*
        }

        /// "Discriminant" of an AST fragment.
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub enum AstFragmentKind {
            OptExpr,
            $($Kind,)*
        }

        impl AstFragmentKind {
            pub fn name(self) -> &'static str {
                match self {
                    AstFragmentKind::OptExpr => "expression",
                    $(AstFragmentKind::$Kind => $kind_name,)*
                }
            }

            fn make_from<'a>(self, result: Box<dyn MacResult + 'a>) -> Option<AstFragment> {
                match self {
                    AstFragmentKind::OptExpr =>
                        result.make_expr().map(Some).map(AstFragment::OptExpr),
                    $(AstFragmentKind::$Kind => result.$make_ast().map(AstFragment::$Kind),)*
                }
            }
        }

        impl AstFragment {
            pub fn make_opt_expr(self) -> Option<P<ast::Expr>> {
                match self {
                    AstFragment::OptExpr(expr) => expr,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            }

            $(pub fn $make_ast(self) -> $AstTy {
                match self {
                    AstFragment::$Kind(ast) => ast,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            })*

            pub fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
                match self {
                    AstFragment::OptExpr(expr) =>
                        AstFragment::OptExpr(expr.and_then(|expr| folder.fold_opt_expr(expr))),
                    $($(AstFragment::$Kind(ast) =>
                        AstFragment::$Kind(folder.$fold_ast(ast)),)*)*
                    $($(AstFragment::$Kind(ast) =>
                        AstFragment::$Kind(ast.into_iter()
                                              .flat_map(|ast| folder.$fold_ast_elt(ast))
                                              .collect()),)*)*
                }
            }

            pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) {
                match *self {
                    AstFragment::OptExpr(Some(ref expr)) => visitor.visit_expr(expr),
                    AstFragment::OptExpr(None) => {}
                    $($(AstFragment::$Kind(ref ast) => visitor.$visit_ast(ast),)*)*
                    $($(AstFragment::$Kind(ref ast) => for ast_elt in &ast[..] {
                        visitor.$visit_ast_elt(ast_elt);
                    })*)*
                }
            }
        }

        impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
            fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
                self.expand_fragment(AstFragment::OptExpr(Some(expr))).make_opt_expr()
            }
            $($(fn $fold_ast(&mut self, ast: $AstTy) -> $AstTy {
                self.expand_fragment(AstFragment::$Kind(ast)).$make_ast()
            })*)*
            $($(fn $fold_ast_elt(&mut self, ast_elt: <$AstTy as IntoIterator>::Item) -> $AstTy {
                self.expand_fragment(AstFragment::$Kind(smallvec![ast_elt])).$make_ast()
            })*)*
        }

        impl<'a> MacResult for ::ext::tt::macro_rules::ParserAnyMacro<'a> {
            $(fn $make_ast(self: Box<::ext::tt::macro_rules::ParserAnyMacro<'a>>)
                           -> Option<$AstTy> {
                Some(self.make(AstFragmentKind::$Kind).$make_ast())
            })*
        }
    }
}

ast_fragments! {
    Expr(P<ast::Expr>) { "expression"; one fn fold_expr; fn visit_expr; fn make_expr; }
    Pat(P<ast::Pat>) { "pattern"; one fn fold_pat; fn visit_pat; fn make_pat; }
    Ty(P<ast::Ty>) { "type"; one fn fold_ty; fn visit_ty; fn make_ty; }
    Stmts(OneVector<ast::Stmt>) { "statement"; many fn fold_stmt; fn visit_stmt; fn make_stmts; }
    Items(OneVector<P<ast::Item>>) { "item"; many fn fold_item; fn visit_item; fn make_items; }
    TraitItems(OneVector<ast::TraitItem>) {
        "trait item"; many fn fold_trait_item; fn visit_trait_item; fn make_trait_items;
    }
    ImplItems(OneVector<ast::ImplItem>) {
        "impl item"; many fn fold_impl_item; fn visit_impl_item; fn make_impl_items;
    }
    ForeignItems(OneVector<ast::ForeignItem>) {
        "foreign item"; many fn fold_foreign_item; fn visit_foreign_item; fn make_foreign_items;
    }
}

impl AstFragmentKind {
    fn dummy(self, span: Span) -> Option<AstFragment> {
        self.make_from(DummyResult::any(span))
    }

    fn expect_from_annotatables<I: IntoIterator<Item = Annotatable>>(self, items: I)
                                                                     -> AstFragment {
        let mut items = items.into_iter();
        match self {
            AstFragmentKind::Items =>
                AstFragment::Items(items.map(Annotatable::expect_item).collect()),
            AstFragmentKind::ImplItems =>
                AstFragment::ImplItems(items.map(Annotatable::expect_impl_item).collect()),
            AstFragmentKind::TraitItems =>
                AstFragment::TraitItems(items.map(Annotatable::expect_trait_item).collect()),
            AstFragmentKind::ForeignItems =>
                AstFragment::ForeignItems(items.map(Annotatable::expect_foreign_item).collect()),
            AstFragmentKind::Stmts =>
                AstFragment::Stmts(items.map(Annotatable::expect_stmt).collect()),
            AstFragmentKind::Expr => AstFragment::Expr(
                items.next().expect("expected exactly one expression").expect_expr()
            ),
            AstFragmentKind::OptExpr =>
                AstFragment::OptExpr(items.next().map(Annotatable::expect_expr)),
            AstFragmentKind::Pat | AstFragmentKind::Ty =>
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

        if segment.ident.name != keywords::CrateRoot.name() &&
            segment.ident.name != keywords::DollarCrate.name()
        {
            path_str.push_str(&segment.ident.as_str())
        }
    }

    MacroBang(Symbol::intern(&path_str))
}

pub struct Invocation {
    pub kind: InvocationKind,
    fragment_kind: AstFragmentKind,
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
    pub fn span(&self) -> Span {
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
            directory: match self.cx.source_map().span_to_unmapped_path(krate.span) {
                FileName::Real(path) => path,
                other => PathBuf::from(other.to_string()),
            },
        };
        module.directory.pop();
        self.cx.root_path = module.directory.clone();
        self.cx.current_expansion.module = Rc::new(module);
        self.cx.current_expansion.crate_span = Some(krate.span);

        let orig_mod_span = krate.module.inner;

        let krate_item = AstFragment::Items(smallvec![P(ast::Item {
            attrs: krate.attrs,
            span: krate.span,
            node: ast::ItemKind::Mod(krate.module),
            ident: keywords::Invalid.ident(),
            id: ast::DUMMY_NODE_ID,
            vis: respan(krate.span.shrink_to_lo(), ast::VisibilityKind::Public),
            tokens: None,
        })]);

        match self.expand_fragment(krate_item).make_items().pop().map(P::into_inner) {
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

    // Fully expand all macro invocations in this AST fragment.
    fn expand_fragment(&mut self, input_fragment: AstFragment) -> AstFragment {
        let orig_expansion_data = self.cx.current_expansion.clone();
        self.cx.current_expansion.depth = 0;

        // Collect all macro invocations and replace them with placeholders.
        let (fragment_with_placeholders, mut invocations)
            = self.collect_invocations(input_fragment, &[]);

        // Optimization: if we resolve all imports now,
        // we'll be able to immediately resolve most of imported macros.
        self.resolve_imports();

        // Resolve paths in all invocations and produce output expanded fragments for them, but
        // do not insert them into our input AST fragment yet, only store in `expanded_fragments`.
        // The output fragments also go through expansion recursively until no invocations are left.
        // Unresolved macros produce dummy outputs as a recovery measure.
        invocations.reverse();
        let mut expanded_fragments = Vec::new();
        let mut derives: HashMap<Mark, Vec<_>> = HashMap::new();
        let mut undetermined_invocations = Vec::new();
        let (mut progress, mut force) = (false, !self.monotonic);
        loop {
            let invoc = if let Some(invoc) = invocations.pop() {
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
            let ext = match self.cx.resolver.resolve_macro_invocation(&invoc, scope, force) {
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
            let (expanded_fragment, new_invocations) = if let Some(ext) = ext {
                if let Some(ext) = ext {
                    let dummy = invoc.fragment_kind.dummy(invoc.span()).unwrap();
                    let fragment = self.expand_invoc(invoc, &*ext).unwrap_or(dummy);
                    self.collect_invocations(fragment, &[])
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
                                .map(|t| t.to_string()).collect::<Vec<_>>();
                            let suggestion = format!("#[derive({})]", trait_list.join(", "));
                            err.span_suggestion_with_applicability(
                                span, "try an outer attribute", suggestion,
                                // We don't 𝑘𝑛𝑜𝑤 that the following item is an ADT
                                Applicability::MaybeIncorrect
                            );
                        }
                        err.emit();
                    }

                    let item = self.fully_configure(item)
                        .map_attrs(|mut attrs| { attrs.retain(|a| a.path != "derive"); attrs });
                    let item_with_markers =
                        add_derived_markers(&mut self.cx, item.span(), &traits, item.clone());
                    let derives = derives.entry(invoc.expansion_data.mark).or_default();

                    for path in &traits {
                        let mark = Mark::fresh(self.cx.current_expansion.mark);
                        derives.push(mark);
                        let item = match self.cx.resolver.resolve_macro_path(
                                path, MacroKind::Derive, Mark::root(), &[], false) {
                            Ok(ext) => match *ext {
                                BuiltinDerive(..) => item_with_markers.clone(),
                                _ => item.clone(),
                            },
                            _ => item.clone(),
                        };
                        invocations.push(Invocation {
                            kind: InvocationKind::Derive { path: path.clone(), item: item },
                            fragment_kind: invoc.fragment_kind,
                            expansion_data: ExpansionData {
                                mark,
                                ..invoc.expansion_data.clone()
                            },
                        });
                    }
                    let fragment = invoc.fragment_kind
                        .expect_from_annotatables(::std::iter::once(item_with_markers));
                    self.collect_invocations(fragment, derives)
                } else {
                    unreachable!()
                }
            } else {
                self.collect_invocations(invoc.fragment_kind.dummy(invoc.span()).unwrap(), &[])
            };

            if expanded_fragments.len() < depth {
                expanded_fragments.push(Vec::new());
            }
            expanded_fragments[depth - 1].push((mark, expanded_fragment));
            if !self.cx.ecfg.single_step {
                invocations.extend(new_invocations.into_iter().rev());
            }
        }

        self.cx.current_expansion = orig_expansion_data;

        // Finally incorporate all the expanded macros into the input AST fragment.
        let mut placeholder_expander = PlaceholderExpander::new(self.cx, self.monotonic);
        while let Some(expanded_fragments) = expanded_fragments.pop() {
            for (mark, expanded_fragment) in expanded_fragments.into_iter().rev() {
                let derives = derives.remove(&mark).unwrap_or_else(Vec::new);
                placeholder_expander.add(NodeId::placeholder_from_mark(mark),
                                         expanded_fragment, derives);
            }
        }
        fragment_with_placeholders.fold_with(&mut placeholder_expander)
    }

    fn resolve_imports(&mut self) {
        if self.monotonic {
            let err_count = self.cx.parse_sess.span_diagnostic.err_count();
            self.cx.resolver.resolve_imports();
            self.cx.resolve_err_count += self.cx.parse_sess.span_diagnostic.err_count() - err_count;
        }
    }

    /// Collect all macro invocations reachable at this time in this AST fragment, and replace
    /// them with "placeholders" - dummy macro invocations with specially crafted `NodeId`s.
    /// Then call into resolver that builds a skeleton ("reduced graph") of the fragment and
    /// prepares data for resolving paths of macro invocations.
    fn collect_invocations(&mut self, fragment: AstFragment, derives: &[Mark])
                           -> (AstFragment, Vec<Invocation>) {
        let (fragment_with_placeholders, invocations) = {
            let mut collector = InvocationCollector {
                cfg: StripUnconfigured {
                    should_test: self.cx.ecfg.should_test,
                    sess: self.cx.parse_sess,
                    features: self.cx.ecfg.features,
                },
                cx: self.cx,
                invocations: Vec::new(),
                monotonic: self.monotonic,
                tests_nameable: true,
            };
            (fragment.fold_with(&mut collector), collector.invocations)
        };

        if self.monotonic {
            let err_count = self.cx.parse_sess.span_diagnostic.err_count();
            let mark = self.cx.current_expansion.mark;
            self.cx.resolver.visit_ast_fragment_with_placeholders(mark, &fragment_with_placeholders,
                                                                  derives);
            self.cx.resolve_err_count += self.cx.parse_sess.span_diagnostic.err_count() - err_count;
        }

        (fragment_with_placeholders, invocations)
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
            Annotatable::ForeignItem(item) => {
                Annotatable::ForeignItem(
                    item.map(|item| cfg.fold_foreign_item(item).pop().unwrap())
                )
            }
            Annotatable::Stmt(stmt) => {
                Annotatable::Stmt(stmt.map(|stmt| cfg.fold_stmt(stmt).pop().unwrap()))
            }
            Annotatable::Expr(expr) => {
                Annotatable::Expr(cfg.fold_expr(expr))
            }
        }
    }

    fn expand_invoc(&mut self, invoc: Invocation, ext: &SyntaxExtension) -> Option<AstFragment> {
        if invoc.fragment_kind == AstFragmentKind::ForeignItems &&
           !self.cx.ecfg.macros_in_extern_enabled() {
            if let SyntaxExtension::NonMacroAttr { .. } = *ext {} else {
                emit_feature_err(&self.cx.parse_sess, "macros_in_extern",
                                 invoc.span(), GateIssue::Language,
                                 "macro invocations in `extern {}` blocks are experimental");
            }
        }

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
                         info.format.name()));
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
                         -> Option<AstFragment> {
        let (attr, item) = match invoc.kind {
            InvocationKind::Attr { attr, item, .. } => (attr?, item),
            _ => unreachable!(),
        };

        if let NonMacroAttr { mark_used: false } = *ext {} else {
            // Macro attrs are always used when expanded,
            // non-macro attrs are considered used when the field says so.
            attr::mark_used(&attr);
        }
        invoc.expansion_data.mark.set_expn_info(ExpnInfo {
            call_site: attr.span,
            def_site: None,
            format: MacroAttribute(Symbol::intern(&attr.path.to_string())),
            allow_internal_unstable: false,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            edition: ext.edition(),
        });

        match *ext {
            NonMacroAttr { .. } => {
                attr::mark_known(&attr);
                let item = item.map_attrs(|mut attrs| { attrs.push(attr); attrs });
                Some(invoc.fragment_kind.expect_from_annotatables(iter::once(item)))
            }
            MultiModifier(ref mac) => {
                let meta = attr.parse_meta(self.cx.parse_sess)
                               .map_err(|mut e| { e.emit(); }).ok()?;
                let item = mac.expand(self.cx, attr.span, &meta, item);
                Some(invoc.fragment_kind.expect_from_annotatables(item))
            }
            MultiDecorator(ref mac) => {
                let mut items = Vec::new();
                let meta = attr.parse_meta(self.cx.parse_sess)
                               .expect("derive meta should already have been parsed");
                mac.expand(self.cx, attr.span, &meta, &item, &mut |item| items.push(item));
                items.push(item);
                Some(invoc.fragment_kind.expect_from_annotatables(items))
            }
            AttrProcMacro(ref mac, ..) => {
                self.gate_proc_macro_attr_item(attr.span, &item);
                let item_tok = TokenTree::Token(DUMMY_SP, Token::interpolated(match item {
                    Annotatable::Item(item) => token::NtItem(item),
                    Annotatable::TraitItem(item) => token::NtTraitItem(item.into_inner()),
                    Annotatable::ImplItem(item) => token::NtImplItem(item.into_inner()),
                    Annotatable::ForeignItem(item) => token::NtForeignItem(item.into_inner()),
                    Annotatable::Stmt(stmt) => token::NtStmt(stmt.into_inner()),
                    Annotatable::Expr(expr) => token::NtExpr(expr),
                })).into();
                let input = self.extract_proc_macro_attr_input(attr.tokens, attr.span);
                let tok_result = mac.expand(self.cx, attr.span, input, item_tok);
                let res = self.parse_ast_fragment(tok_result, invoc.fragment_kind,
                                                  &attr.path, attr.span);
                self.gate_proc_macro_expansion(attr.span, &res);
                res
            }
            ProcMacroDerive(..) | BuiltinDerive(..) => {
                self.cx.span_err(attr.span, &format!("`{}` is a derive mode", attr.path));
                self.cx.trace_macros_diag();
                invoc.fragment_kind.dummy(attr.span)
            }
            _ => {
                let msg = &format!("macro `{}` may not be used in attributes", attr.path);
                self.cx.span_err(attr.span, msg);
                self.cx.trace_macros_diag();
                invoc.fragment_kind.dummy(attr.span)
            }
        }
    }

    fn extract_proc_macro_attr_input(&self, tokens: TokenStream, span: Span) -> TokenStream {
        let mut trees = tokens.trees();
        match trees.next() {
            Some(TokenTree::Delimited(_, delim)) => {
                if trees.next().is_none() {
                    return delim.tts.into()
                }
            }
            Some(TokenTree::Token(..)) => {}
            None => return TokenStream::empty(),
        }
        self.cx.span_err(span, "custom attribute invocations must be \
            of the form #[foo] or #[foo(..)], the macro name must only be \
            followed by a delimiter token");
        TokenStream::empty()
    }

    fn gate_proc_macro_attr_item(&self, span: Span, item: &Annotatable) {
        let (kind, gate) = match *item {
            Annotatable::Item(ref item) => {
                match item.node {
                    ItemKind::Mod(_) if self.cx.ecfg.proc_macro_mod() => return,
                    ItemKind::Mod(_) => ("modules", "proc_macro_mod"),
                    _ => return,
                }
            }
            Annotatable::TraitItem(_) => return,
            Annotatable::ImplItem(_) => return,
            Annotatable::ForeignItem(_) => return,
            Annotatable::Stmt(_) |
            Annotatable::Expr(_) if self.cx.ecfg.proc_macro_expr() => return,
            Annotatable::Stmt(_) => ("statements", "proc_macro_expr"),
            Annotatable::Expr(_) => ("expressions", "proc_macro_expr"),
        };
        emit_feature_err(
            self.cx.parse_sess,
            gate,
            span,
            GateIssue::Language,
            &format!("custom attributes cannot be applied to {}", kind),
        );
    }

    fn gate_proc_macro_expansion(&self, span: Span, fragment: &Option<AstFragment>) {
        if self.cx.ecfg.proc_macro_gen() {
            return
        }
        let fragment = match fragment {
            Some(fragment) => fragment,
            None => return,
        };

        fragment.visit_with(&mut DisallowModules {
            span,
            parse_sess: self.cx.parse_sess,
        });

        struct DisallowModules<'a> {
            span: Span,
            parse_sess: &'a ParseSess,
        }

        impl<'ast, 'a> Visitor<'ast> for DisallowModules<'a> {
            fn visit_item(&mut self, i: &'ast ast::Item) {
                let name = match i.node {
                    ast::ItemKind::Mod(_) => Some("modules"),
                    ast::ItemKind::MacroDef(_) => Some("macro definitions"),
                    _ => None,
                };
                if let Some(name) = name {
                    emit_feature_err(
                        self.parse_sess,
                        "proc_macro_gen",
                        self.span,
                        GateIssue::Language,
                        &format!("procedural macros cannot expand to {}", name),
                    );
                }
                visit::walk_item(self, i);
            }

            fn visit_mac(&mut self, _mac: &'ast ast::Mac) {
                // ...
            }
        }
    }

    /// Expand a macro invocation. Returns the resulting expanded AST fragment.
    fn expand_bang_invoc(&mut self,
                         invoc: Invocation,
                         ext: &SyntaxExtension)
                         -> Option<AstFragment> {
        let (mark, kind) = (invoc.expansion_data.mark, invoc.fragment_kind);
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
                                          local_inner_macros,
                                          // can't infer this type
                                          unstable_feature: Option<(Symbol, u32)>,
                                          edition| {

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
                def_site: def_site_span,
                format: macro_bang_format(path),
                allow_internal_unstable,
                allow_internal_unsafe,
                local_inner_macros,
                edition,
            });
            Ok(())
        };

        let opt_expanded = match *ext {
            DeclMacro { ref expander, def_info, edition, .. } => {
                if let Err(dummy_span) = validate_and_set_expn_info(self, def_info.map(|(_, s)| s),
                                                                    false, false, false, None,
                                                                    edition) {
                    dummy_span
                } else {
                    kind.make_from(expander.expand(self.cx, span, mac.node.stream()))
                }
            }

            NormalTT {
                ref expander,
                def_info,
                allow_internal_unstable,
                allow_internal_unsafe,
                local_inner_macros,
                unstable_feature,
                edition,
            } => {
                if let Err(dummy_span) = validate_and_set_expn_info(self, def_info.map(|(_, s)| s),
                                                                    allow_internal_unstable,
                                                                    allow_internal_unsafe,
                                                                    local_inner_macros,
                                                                    unstable_feature,
                                                                    edition) {
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
                        def_site: tt_span,
                        format: macro_bang_format(path),
                        allow_internal_unstable,
                        allow_internal_unsafe: false,
                        local_inner_macros: false,
                        edition: hygiene::default_edition(),
                    });

                    let input: Vec<_> = mac.node.stream().into_trees().collect();
                    kind.make_from(expander.expand(self.cx, span, ident, input))
                }
            }

            MultiDecorator(..) | MultiModifier(..) |
            AttrProcMacro(..) | SyntaxExtension::NonMacroAttr { .. } => {
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

            SyntaxExtension::ProcMacro { ref expander, allow_internal_unstable, edition } => {
                if ident.name != keywords::Invalid.name() {
                    let msg =
                        format!("macro {}! expects no ident argument, given '{}'", path, ident);
                    self.cx.span_err(path.span, &msg);
                    self.cx.trace_macros_diag();
                    kind.dummy(span)
                } else {
                    self.gate_proc_macro_expansion_kind(span, kind);
                    invoc.expansion_data.mark.set_expn_info(ExpnInfo {
                        call_site: span,
                        // FIXME procedural macros do not have proper span info
                        // yet, when they do, we should use it here.
                        def_site: None,
                        format: macro_bang_format(path),
                        // FIXME probably want to follow macro_rules macros here.
                        allow_internal_unstable,
                        allow_internal_unsafe: false,
                        local_inner_macros: false,
                        edition,
                    });

                    let tok_result = expander.expand(self.cx, span, mac.node.stream());
                    let result = self.parse_ast_fragment(tok_result, kind, path, span);
                    self.gate_proc_macro_expansion(span, &result);
                    result
                }
            }
        };

        if opt_expanded.is_some() {
            opt_expanded
        } else {
            let msg = format!("non-{kind} macro in {kind} position: {name}",
                              name = path.segments[0].ident.name, kind = kind.name());
            self.cx.span_err(path.span, &msg);
            self.cx.trace_macros_diag();
            kind.dummy(span)
        }
    }

    fn gate_proc_macro_expansion_kind(&self, span: Span, kind: AstFragmentKind) {
        let kind = match kind {
            AstFragmentKind::Expr => "expressions",
            AstFragmentKind::OptExpr => "expressions",
            AstFragmentKind::Pat => "patterns",
            AstFragmentKind::Ty => "types",
            AstFragmentKind::Stmts => "statements",
            AstFragmentKind::Items => return,
            AstFragmentKind::TraitItems => return,
            AstFragmentKind::ImplItems => return,
            AstFragmentKind::ForeignItems => return,
        };
        if self.cx.ecfg.proc_macro_non_items() {
            return
        }
        emit_feature_err(
            self.cx.parse_sess,
            "proc_macro_non_items",
            span,
            GateIssue::Language,
            &format!("procedural macros cannot be expanded to {}", kind),
        );
    }

    /// Expand a derive invocation. Returns the resulting expanded AST fragment.
    fn expand_derive_invoc(&mut self,
                           invoc: Invocation,
                           ext: &SyntaxExtension)
                           -> Option<AstFragment> {
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
            def_site: None,
            format: MacroAttribute(pretty_name),
            allow_internal_unstable: false,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            edition: ext.edition(),
        };

        match *ext {
            ProcMacroDerive(ref ext, ..) => {
                invoc.expansion_data.mark.set_expn_info(expn_info);
                let span = span.with_ctxt(self.cx.backtrace());
                let dummy = ast::MetaItem { // FIXME(jseyfried) avoid this
                    ident: Path::from_ident(keywords::Invalid.ident()),
                    span: DUMMY_SP,
                    node: ast::MetaItemKind::Word,
                };
                let items = ext.expand(self.cx, span, &dummy, item);
                Some(invoc.fragment_kind.expect_from_annotatables(items))
            }
            BuiltinDerive(func) => {
                expn_info.allow_internal_unstable = true;
                invoc.expansion_data.mark.set_expn_info(expn_info);
                let span = span.with_ctxt(self.cx.backtrace());
                let mut items = Vec::new();
                func(self.cx, span, &attr.meta()?, &item, &mut |a| items.push(a));
                Some(invoc.fragment_kind.expect_from_annotatables(items))
            }
            _ => {
                let msg = &format!("macro `{}` may not be used for derive attributes", attr.path);
                self.cx.span_err(span, msg);
                self.cx.trace_macros_diag();
                invoc.fragment_kind.dummy(span)
            }
        }
    }

    fn parse_ast_fragment(&mut self,
                          toks: TokenStream,
                          kind: AstFragmentKind,
                          path: &Path,
                          span: Span)
                          -> Option<AstFragment> {
        let mut parser = self.cx.new_parser_from_tts(&toks.into_trees().collect::<Vec<_>>());
        match parser.parse_ast_fragment(kind, false) {
            Ok(fragment) => {
                parser.ensure_complete_parse(path, kind.name(), span);
                Some(fragment)
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
    pub fn parse_ast_fragment(&mut self, kind: AstFragmentKind, macro_legacy_warnings: bool)
                              -> PResult<'a, AstFragment> {
        Ok(match kind {
            AstFragmentKind::Items => {
                let mut items = OneVector::new();
                while let Some(item) = self.parse_item()? {
                    items.push(item);
                }
                AstFragment::Items(items)
            }
            AstFragmentKind::TraitItems => {
                let mut items = OneVector::new();
                while self.token != token::Eof {
                    items.push(self.parse_trait_item(&mut false)?);
                }
                AstFragment::TraitItems(items)
            }
            AstFragmentKind::ImplItems => {
                let mut items = OneVector::new();
                while self.token != token::Eof {
                    items.push(self.parse_impl_item(&mut false)?);
                }
                AstFragment::ImplItems(items)
            }
            AstFragmentKind::ForeignItems => {
                let mut items = OneVector::new();
                while self.token != token::Eof {
                    if let Some(item) = self.parse_foreign_item()? {
                        items.push(item);
                    }
                }
                AstFragment::ForeignItems(items)
            }
            AstFragmentKind::Stmts => {
                let mut stmts = OneVector::new();
                while self.token != token::Eof &&
                      // won't make progress on a `}`
                      self.token != token::CloseDelim(token::Brace) {
                    if let Some(stmt) = self.parse_full_stmt(macro_legacy_warnings)? {
                        stmts.push(stmt);
                    }
                }
                AstFragment::Stmts(stmts)
            }
            AstFragmentKind::Expr => AstFragment::Expr(self.parse_expr()?),
            AstFragmentKind::OptExpr => {
                if self.token != token::Eof {
                    AstFragment::OptExpr(Some(self.parse_expr()?))
                } else {
                    AstFragment::OptExpr(None)
                }
            },
            AstFragmentKind::Ty => AstFragment::Ty(self.parse_ty()?),
            AstFragmentKind::Pat => AstFragment::Pat(self.parse_pat()?),
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

    /// Test functions need to be nameable. Tests inside functions or in other
    /// unnameable locations need to be ignored. `tests_nameable` tracks whether
    /// any test functions found in the current context would be nameable.
    tests_nameable: bool,
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn collect(&mut self, fragment_kind: AstFragmentKind, kind: InvocationKind) -> AstFragment {
        let mark = Mark::fresh(self.cx.current_expansion.mark);
        self.invocations.push(Invocation {
            kind,
            fragment_kind,
            expansion_data: ExpansionData {
                mark,
                depth: self.cx.current_expansion.depth + 1,
                ..self.cx.current_expansion.clone()
            },
        });
        placeholder(fragment_kind, NodeId::placeholder_from_mark(mark))
    }

    /// Folds the item allowing tests to be expanded because they are still nameable.
    /// This should probably only be called with module items
    fn fold_nameable(&mut self, item: P<ast::Item>) -> OneVector<P<ast::Item>> {
        fold::noop_fold_item(item, self)
    }

    /// Folds the item but doesn't allow tests to occur within it
    fn fold_unnameable(&mut self, item: P<ast::Item>) -> OneVector<P<ast::Item>> {
        let was_nameable = mem::replace(&mut self.tests_nameable, false);
        let items = fold::noop_fold_item(item, self);
        self.tests_nameable = was_nameable;
        items
    }

    fn collect_bang(&mut self, mac: ast::Mac, span: Span, kind: AstFragmentKind) -> AstFragment {
        self.collect(kind, InvocationKind::Bang { mac: mac, ident: None, span: span })
    }

    fn collect_attr(&mut self,
                    attr: Option<ast::Attribute>,
                    traits: Vec<Path>,
                    item: Annotatable,
                    kind: AstFragmentKind)
                    -> AstFragment {
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

            attr = find_attr_invoc(&mut attrs);
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

            attr = find_attr_invoc(&mut attrs);
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
            self.check_attribute_inner(attr, features);

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
        self.check_attribute_inner(at, features);
    }

    fn check_attribute_inner(&mut self, at: &ast::Attribute, features: &Features) {
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

            // AstFragmentKind::Expr requires the macro to emit an expression
            return self.collect_attr(attr, vec![], Annotatable::Expr(P(expr)),
                                     AstFragmentKind::Expr).make_expr();
        }

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.check_attributes(&expr.attrs);
            self.collect_bang(mac, expr.span, AstFragmentKind::Expr).make_expr()
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
                                     AstFragmentKind::OptExpr)
                .make_opt_expr();
        }

        if let ast::ExprKind::Mac(mac) = expr.node {
            self.check_attributes(&expr.attrs);
            self.collect_bang(mac, expr.span, AstFragmentKind::OptExpr).make_opt_expr()
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
            PatKind::Mac(mac) => self.collect_bang(mac, pat.span, AstFragmentKind::Pat).make_pat(),
            _ => unreachable!(),
        })
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> OneVector<ast::Stmt> {
        let mut stmt = match self.cfg.configure_stmt(stmt) {
            Some(stmt) => stmt,
            None => return OneVector::new(),
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
                                         Annotatable::Stmt(P(stmt_)), AstFragmentKind::Stmts)
                    .make_stmts();
            }

            stmt = stmt_;
        }

        if let StmtKind::Mac(mac) = stmt.node {
            let (mac, style, attrs) = mac.into_inner();
            self.check_attributes(&attrs);
            let mut placeholder = self.collect_bang(mac, stmt.span, AstFragmentKind::Stmts)
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

    fn fold_item(&mut self, item: P<ast::Item>) -> OneVector<P<ast::Item>> {
        let item = configure!(self, item);

        let (attr, traits, mut item) = self.classify_item(item);
        if attr.is_some() || !traits.is_empty() {
            let item = Annotatable::Item(item);
            return self.collect_attr(attr, traits, item, AstFragmentKind::Items).make_items();
        }

        match item.node {
            ast::ItemKind::Mac(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.node {
                    ItemKind::Mac(mac) => {
                        self.collect(AstFragmentKind::Items, InvocationKind::Bang {
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
                    return self.fold_nameable(item);
                }

                let orig_directory_ownership = self.cx.current_expansion.directory_ownership;
                let mut module = (*self.cx.current_expansion.module).clone();
                module.mod_path.push(item.ident);

                // Detect if this is an inline module (`mod m { ... }` as opposed to `mod m;`).
                // In the non-inline case, `inner` is never the dummy span (c.f. `parse_item_mod`).
                // Thus, if `inner` is the dummy span, we know the module is inline.
                let inline_module = item.span.contains(inner) || inner.is_dummy();

                if inline_module {
                    if let Some(path) = attr::first_attr_value_str_by_name(&item.attrs, "path") {
                        self.cx.current_expansion.directory_ownership =
                            DirectoryOwnership::Owned { relative: None };
                        module.directory.push(&*path.as_str());
                    } else {
                        module.directory.push(&*item.ident.as_str());
                    }
                } else {
                    let path = self.cx.parse_sess.source_map().span_to_unmapped_path(inner);
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
                let result = self.fold_nameable(item);
                self.cx.current_expansion.module = orig_module;
                self.cx.current_expansion.directory_ownership = orig_directory_ownership;
                result
            }
            // Ensure that test functions are accessible from the test harness.
            // #[test] fn foo() {}
            // becomes:
            // #[test] pub fn foo_gensym(){}
            // #[allow(unused)]
            // use foo_gensym as foo;
            ast::ItemKind::Fn(..) if self.cx.ecfg.should_test => {
                if self.tests_nameable && item.attrs.iter().any(|attr| is_test_or_bench(attr)) {
                    let orig_ident = item.ident;
                    let orig_vis   = item.vis.clone();

                    // Publicize the item under gensymed name to avoid pollution
                    item = item.map(|mut item| {
                        item.vis = respan(item.vis.span, ast::VisibilityKind::Public);
                        item.ident = item.ident.gensym();
                        item
                    });

                    // Use the gensymed name under the item's original visibility
                    let mut use_item = self.cx.item_use_simple_(
                        item.ident.span,
                        orig_vis,
                        Some(orig_ident),
                        self.cx.path(item.ident.span,
                            vec![keywords::SelfValue.ident(), item.ident]));

                    // #[allow(unused)] because the test function probably isn't being referenced
                    use_item = use_item.map(|mut ui| {
                        ui.attrs.push(
                            self.cx.attribute(DUMMY_SP, attr::mk_list_item(DUMMY_SP,
                                Ident::from_str("allow"), vec![
                                    attr::mk_nested_word_item(Ident::from_str("unused"))
                                ]
                            ))
                        );

                        ui
                    });

                    OneVector::from_iter(
                        self.fold_unnameable(item).into_iter()
                            .chain(self.fold_unnameable(use_item)))
                } else {
                    self.fold_unnameable(item)
                }
            }
            _ => self.fold_unnameable(item),
        }
    }

    fn fold_trait_item(&mut self, item: ast::TraitItem) -> OneVector<ast::TraitItem> {
        let item = configure!(self, item);

        let (attr, traits, item) = self.classify_item(item);
        if attr.is_some() || !traits.is_empty() {
            let item = Annotatable::TraitItem(P(item));
            return self.collect_attr(attr, traits, item, AstFragmentKind::TraitItems)
                .make_trait_items()
        }

        match item.node {
            ast::TraitItemKind::Macro(mac) => {
                let ast::TraitItem { attrs, span, .. } = item;
                self.check_attributes(&attrs);
                self.collect_bang(mac, span, AstFragmentKind::TraitItems).make_trait_items()
            }
            _ => fold::noop_fold_trait_item(item, self),
        }
    }

    fn fold_impl_item(&mut self, item: ast::ImplItem) -> OneVector<ast::ImplItem> {
        let item = configure!(self, item);

        let (attr, traits, item) = self.classify_item(item);
        if attr.is_some() || !traits.is_empty() {
            let item = Annotatable::ImplItem(P(item));
            return self.collect_attr(attr, traits, item, AstFragmentKind::ImplItems)
                .make_impl_items();
        }

        match item.node {
            ast::ImplItemKind::Macro(mac) => {
                let ast::ImplItem { attrs, span, .. } = item;
                self.check_attributes(&attrs);
                self.collect_bang(mac, span, AstFragmentKind::ImplItems).make_impl_items()
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
            ast::TyKind::Mac(mac) => self.collect_bang(mac, ty.span, AstFragmentKind::Ty).make_ty(),
            _ => unreachable!(),
        }
    }

    fn fold_foreign_mod(&mut self, foreign_mod: ast::ForeignMod) -> ast::ForeignMod {
        noop_fold_foreign_mod(self.cfg.configure_foreign_mod(foreign_mod), self)
    }

    fn fold_foreign_item(&mut self,
                         foreign_item: ast::ForeignItem) -> OneVector<ast::ForeignItem> {
        let (attr, traits, foreign_item) = self.classify_item(foreign_item);

        if attr.is_some() || !traits.is_empty() {
            let item = Annotatable::ForeignItem(P(foreign_item));
            return self.collect_attr(attr, traits, item, AstFragmentKind::ForeignItems)
                .make_foreign_items();
        }

        if let ast::ForeignItemKind::Macro(mac) = foreign_item.node {
            self.check_attributes(&foreign_item.attrs);
            return self.collect_bang(mac, foreign_item.span, AstFragmentKind::ForeignItems)
                .make_foreign_items();
        }

        noop_fold_foreign_item(foreign_item, self)
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        match item {
            ast::ItemKind::MacroDef(..) => item,
            _ => noop_fold_item_kind(self.cfg.configure_item_kind(item), self),
        }
    }

    fn fold_generic_param(&mut self, param: ast::GenericParam) -> ast::GenericParam {
        self.cfg.disallow_cfg_on_generic_param(&param);
        noop_fold_generic_param(param, self)
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
                            let src_interned = Symbol::intern(&src);

                            // Add this input file to the code map to make it available as
                            // dependency information
                            self.cx.source_map().new_source_file(filename.into(), src);

                            let include_info = vec![
                                dummy_spanned(ast::NestedMetaItemKind::MetaItem(
                                        attr::mk_name_value_item_str(Ident::from_str("file"),
                                                                     dummy_spanned(file)))),
                                dummy_spanned(ast::NestedMetaItemKind::MetaItem(
                                        attr::mk_name_value_item_str(Ident::from_str("contents"),
                                                            dummy_spanned(src_interned)))),
                            ];

                            let include_ident = Ident::from_str("include");
                            let item = attr::mk_list_item(DUMMY_SP, include_ident, include_info);
                            items.push(dummy_spanned(ast::NestedMetaItemKind::MetaItem(item)));
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

            let meta = attr::mk_list_item(DUMMY_SP, Ident::from_str("doc"), items);
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
        fn enable_format_args_nl = format_args_nl,
        fn macros_in_extern_enabled = macros_in_extern,
        fn proc_macro_mod = proc_macro_mod,
        fn proc_macro_gen = proc_macro_gen,
        fn proc_macro_expr = proc_macro_expr,
        fn proc_macro_non_items = proc_macro_non_items,
    }
}

// A Marker adds the given mark to the syntax context.
#[derive(Debug)]
pub struct Marker(pub Mark);

impl Folder for Marker {
    fn new_span(&mut self, span: Span) -> Span {
        span.apply_mark(self.0)
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        noop_fold_mac(mac, self)
    }
}
