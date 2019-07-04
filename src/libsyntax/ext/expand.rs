use crate::ast::{self, Block, Ident, LitKind, NodeId, PatKind, Path};
use crate::ast::{MacStmtStyle, StmtKind, ItemKind};
use crate::attr::{self, HasAttrs};
use crate::source_map::{dummy_spanned, respan};
use crate::config::StripUnconfigured;
use crate::ext::base::*;
use crate::ext::derive::{add_derived_markers, collect_derives};
use crate::ext::hygiene::{Mark, SyntaxContext};
use crate::ext::placeholders::{placeholder, PlaceholderExpander};
use crate::feature_gate::{self, Features, GateIssue, is_builtin_attr, emit_feature_err};
use crate::mut_visit::*;
use crate::parse::{DirectoryOwnership, PResult, ParseSess};
use crate::parse::token;
use crate::parse::parser::Parser;
use crate::ptr::P;
use crate::symbol::{sym, Symbol};
use crate::tokenstream::{TokenStream, TokenTree};
use crate::visit::{self, Visitor};
use crate::util::map_in_place::MapInPlace;

use errors::{Applicability, FatalError};
use smallvec::{smallvec, SmallVec};
use syntax_pos::{Span, DUMMY_SP, FileName};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use std::fs;
use std::io::ErrorKind;
use std::{iter, mem};
use std::ops::DerefMut;
use std::rc::Rc;
use std::path::PathBuf;

macro_rules! ast_fragments {
    (
        $($Kind:ident($AstTy:ty) {
            $kind_name:expr;
            $(one fn $mut_visit_ast:ident; fn $visit_ast:ident;)?
            $(many fn $flat_map_ast_elt:ident; fn $visit_ast_elt:ident;)?
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

            pub fn mut_visit_with<F: MutVisitor>(&mut self, vis: &mut F) {
                match self {
                    AstFragment::OptExpr(opt_expr) => {
                        visit_clobber(opt_expr, |opt_expr| {
                            if let Some(expr) = opt_expr {
                                vis.filter_map_expr(expr)
                            } else {
                                None
                            }
                        });
                    }
                    $($(AstFragment::$Kind(ast) => vis.$mut_visit_ast(ast),)?)*
                    $($(AstFragment::$Kind(ast) =>
                        ast.flat_map_in_place(|ast| vis.$flat_map_ast_elt(ast)),)?)*
                }
            }

            pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) {
                match *self {
                    AstFragment::OptExpr(Some(ref expr)) => visitor.visit_expr(expr),
                    AstFragment::OptExpr(None) => {}
                    $($(AstFragment::$Kind(ref ast) => visitor.$visit_ast(ast),)?)*
                    $($(AstFragment::$Kind(ref ast) => for ast_elt in &ast[..] {
                        visitor.$visit_ast_elt(ast_elt);
                    })?)*
                }
            }
        }

        impl<'a, 'b> MutVisitor for MacroExpander<'a, 'b> {
            fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
                self.expand_fragment(AstFragment::OptExpr(Some(expr))).make_opt_expr()
            }
            $($(fn $mut_visit_ast(&mut self, ast: &mut $AstTy) {
                visit_clobber(ast, |ast| self.expand_fragment(AstFragment::$Kind(ast)).$make_ast());
            })?)*
            $($(fn $flat_map_ast_elt(&mut self, ast_elt: <$AstTy as IntoIterator>::Item) -> $AstTy {
                self.expand_fragment(AstFragment::$Kind(smallvec![ast_elt])).$make_ast()
            })?)*
        }

        impl<'a> MacResult for crate::ext::tt::macro_rules::ParserAnyMacro<'a> {
            $(fn $make_ast(self: Box<crate::ext::tt::macro_rules::ParserAnyMacro<'a>>)
                           -> Option<$AstTy> {
                Some(self.make(AstFragmentKind::$Kind).$make_ast())
            })*
        }
    }
}

ast_fragments! {
    Expr(P<ast::Expr>) { "expression"; one fn visit_expr; fn visit_expr; fn make_expr; }
    Pat(P<ast::Pat>) { "pattern"; one fn visit_pat; fn visit_pat; fn make_pat; }
    Ty(P<ast::Ty>) { "type"; one fn visit_ty; fn visit_ty; fn make_ty; }
    Stmts(SmallVec<[ast::Stmt; 1]>) {
        "statement"; many fn flat_map_stmt; fn visit_stmt; fn make_stmts;
    }
    Items(SmallVec<[P<ast::Item>; 1]>) {
        "item"; many fn flat_map_item; fn visit_item; fn make_items;
    }
    TraitItems(SmallVec<[ast::TraitItem; 1]>) {
        "trait item"; many fn flat_map_trait_item; fn visit_trait_item; fn make_trait_items;
    }
    ImplItems(SmallVec<[ast::ImplItem; 1]>) {
        "impl item"; many fn flat_map_impl_item; fn visit_impl_item; fn make_impl_items;
    }
    ForeignItems(SmallVec<[ast::ForeignItem; 1]>) {
        "foreign item"; many fn flat_map_foreign_item; fn visit_foreign_item; fn make_foreign_items;
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

pub struct Invocation {
    pub kind: InvocationKind,
    fragment_kind: AstFragmentKind,
    pub expansion_data: ExpansionData,
}

pub enum InvocationKind {
    Bang {
        mac: ast::Mac,
        span: Span,
    },
    Attr {
        attr: Option<ast::Attribute>,
        traits: Vec<Path>,
        item: Annotatable,
        // We temporarily report errors for attribute macros placed after derives
        after_derive: bool,
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

pub struct MacroExpander<'a, 'b> {
    pub cx: &'a mut ExtCtxt<'b>,
    monotonic: bool, // cf. `cx.monotonic_expander()`
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>, monotonic: bool) -> Self {
        MacroExpander { cx, monotonic }
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
            ident: Ident::invalid(),
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
                    inline: true,
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
        let (mut fragment_with_placeholders, mut invocations)
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
        let mut derives: FxHashMap<Mark, Vec<_>> = FxHashMap::default();
        let mut undetermined_invocations = Vec::new();
        let (mut progress, mut force) = (false, !self.monotonic);
        loop {
            let invoc = if let Some(invoc) = invocations.pop() {
                invoc
            } else {
                self.resolve_imports();
                if undetermined_invocations.is_empty() { break }
                invocations = mem::take(&mut undetermined_invocations);
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
                    let (invoc_fragment_kind, invoc_span) = (invoc.fragment_kind, invoc.span());
                    let fragment = self.expand_invoc(invoc, &*ext).unwrap_or_else(|| {
                        invoc_fragment_kind.dummy(invoc_span).unwrap()
                    });
                    self.collect_invocations(fragment, &[])
                } else if let InvocationKind::Attr { attr: None, traits, item, .. } = invoc.kind {
                    if !item.derive_allowed() {
                        let attr = attr::find_by_name(item.attrs(), sym::derive)
                            .expect("`derive` attribute should exist");
                        let span = attr.span;
                        let mut err = self.cx.mut_span_err(span,
                                                           "`derive` may only be applied to \
                                                            structs, enums and unions");
                        if let ast::AttrStyle::Inner = attr.style {
                            let trait_list = traits.iter()
                                .map(|t| t.to_string()).collect::<Vec<_>>();
                            let suggestion = format!("#[derive({})]", trait_list.join(", "));
                            err.span_suggestion(
                                span, "try an outer attribute", suggestion,
                                // We don't 𝑘𝑛𝑜𝑤 that the following item is an ADT
                                Applicability::MaybeIncorrect
                            );
                        }
                        err.emit();
                    }

                    let mut item = self.fully_configure(item);
                    item.visit_attrs(|attrs| attrs.retain(|a| a.path != sym::derive));
                    let mut item_with_markers = item.clone();
                    add_derived_markers(&mut self.cx, item.span(), &traits, &mut item_with_markers);
                    let derives = derives.entry(invoc.expansion_data.mark).or_default();

                    derives.reserve(traits.len());
                    invocations.reserve(traits.len());
                    for path in &traits {
                        let mark = Mark::fresh(self.cx.current_expansion.mark);
                        derives.push(mark);
                        let item = match self.cx.resolver.resolve_macro_path(
                                path, MacroKind::Derive, Mark::root(), Vec::new(), false) {
                            Ok(ext) => match ext.kind {
                                SyntaxExtensionKind::LegacyDerive(..) => item_with_markers.clone(),
                                _ => item.clone(),
                            },
                            _ => item.clone(),
                        };
                        invocations.push(Invocation {
                            kind: InvocationKind::Derive { path: path.clone(), item },
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
        fragment_with_placeholders.mut_visit_with(&mut placeholder_expander);
        fragment_with_placeholders
    }

    fn resolve_imports(&mut self) {
        if self.monotonic {
            self.cx.resolver.resolve_imports();
        }
    }

    /// Collects all macro invocations reachable at this time in this AST fragment, and replace
    /// them with "placeholders" - dummy macro invocations with specially crafted `NodeId`s.
    /// Then call into resolver that builds a skeleton ("reduced graph") of the fragment and
    /// prepares data for resolving paths of macro invocations.
    fn collect_invocations(&mut self, mut fragment: AstFragment, derives: &[Mark])
                           -> (AstFragment, Vec<Invocation>) {
        // Resolve `$crate`s in the fragment for pretty-printing.
        self.cx.resolver.resolve_dollar_crates(&fragment);

        let invocations = {
            let mut collector = InvocationCollector {
                cfg: StripUnconfigured {
                    sess: self.cx.parse_sess,
                    features: self.cx.ecfg.features,
                },
                cx: self.cx,
                invocations: Vec::new(),
                monotonic: self.monotonic,
            };
            fragment.mut_visit_with(&mut collector);
            collector.invocations
        };

        if self.monotonic {
            self.cx.resolver.visit_ast_fragment_with_placeholders(
                self.cx.current_expansion.mark, &fragment, derives);
        }

        (fragment, invocations)
    }

    fn fully_configure(&mut self, item: Annotatable) -> Annotatable {
        let mut cfg = StripUnconfigured {
            sess: self.cx.parse_sess,
            features: self.cx.ecfg.features,
        };
        // Since the item itself has already been configured by the InvocationCollector,
        // we know that fold result vector will contain exactly one element
        match item {
            Annotatable::Item(item) => {
                Annotatable::Item(cfg.flat_map_item(item).pop().unwrap())
            }
            Annotatable::TraitItem(item) => {
                Annotatable::TraitItem(
                    item.map(|item| cfg.flat_map_trait_item(item).pop().unwrap()))
            }
            Annotatable::ImplItem(item) => {
                Annotatable::ImplItem(item.map(|item| cfg.flat_map_impl_item(item).pop().unwrap()))
            }
            Annotatable::ForeignItem(item) => {
                Annotatable::ForeignItem(
                    item.map(|item| cfg.flat_map_foreign_item(item).pop().unwrap())
                )
            }
            Annotatable::Stmt(stmt) => {
                Annotatable::Stmt(stmt.map(|stmt| cfg.flat_map_stmt(stmt).pop().unwrap()))
            }
            Annotatable::Expr(mut expr) => {
                Annotatable::Expr({ cfg.visit_expr(&mut expr); expr })
            }
        }
    }

    fn expand_invoc(&mut self, invoc: Invocation, ext: &SyntaxExtension) -> Option<AstFragment> {
        if invoc.fragment_kind == AstFragmentKind::ForeignItems &&
           !self.cx.ecfg.macros_in_extern_enabled() {
            if let SyntaxExtensionKind::NonMacroAttr { .. } = ext.kind {} else {
                emit_feature_err(&self.cx.parse_sess, sym::macros_in_extern,
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
        let (attr, mut item) = match invoc.kind {
            InvocationKind::Attr { attr, item, .. } => (attr?, item),
            _ => unreachable!(),
        };

        match &ext.kind {
            SyntaxExtensionKind::NonMacroAttr { mark_used } => {
                attr::mark_known(&attr);
                if *mark_used {
                    attr::mark_used(&attr);
                }
                item.visit_attrs(|attrs| attrs.push(attr));
                Some(invoc.fragment_kind.expect_from_annotatables(iter::once(item)))
            }
            SyntaxExtensionKind::LegacyAttr(expander) => {
                let meta = attr.parse_meta(self.cx.parse_sess)
                               .map_err(|mut e| { e.emit(); }).ok()?;
                let item = expander.expand(self.cx, attr.span, &meta, item);
                Some(invoc.fragment_kind.expect_from_annotatables(item))
            }
            SyntaxExtensionKind::Attr(expander) => {
                self.gate_proc_macro_attr_item(attr.span, &item);
                let item_tok = TokenTree::token(token::Interpolated(Lrc::new(match item {
                    Annotatable::Item(item) => token::NtItem(item),
                    Annotatable::TraitItem(item) => token::NtTraitItem(item.into_inner()),
                    Annotatable::ImplItem(item) => token::NtImplItem(item.into_inner()),
                    Annotatable::ForeignItem(item) => token::NtForeignItem(item.into_inner()),
                    Annotatable::Stmt(stmt) => token::NtStmt(stmt.into_inner()),
                    Annotatable::Expr(expr) => token::NtExpr(expr),
                })), DUMMY_SP).into();
                let input = self.extract_proc_macro_attr_input(attr.tokens, attr.span);
                let tok_result = expander.expand(self.cx, attr.span, input, item_tok);
                let res = self.parse_ast_fragment(tok_result, invoc.fragment_kind,
                                                  &attr.path, attr.span);
                self.gate_proc_macro_expansion(attr.span, &res);
                res
            }
            SyntaxExtensionKind::Derive(..) | SyntaxExtensionKind::LegacyDerive(..) => {
                self.cx.span_err(attr.span, &format!("`{}` is a derive macro", attr.path));
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
            Some(TokenTree::Delimited(_, _, tts)) => {
                if trees.next().is_none() {
                    return tts.into()
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
                    ItemKind::Mod(_) if self.cx.ecfg.proc_macro_hygiene() => return,
                    ItemKind::Mod(_) => ("modules", sym::proc_macro_hygiene),
                    _ => return,
                }
            }
            Annotatable::TraitItem(_) => return,
            Annotatable::ImplItem(_) => return,
            Annotatable::ForeignItem(_) => return,
            Annotatable::Stmt(_) |
            Annotatable::Expr(_) if self.cx.ecfg.proc_macro_hygiene() => return,
            Annotatable::Stmt(_) => ("statements", sym::proc_macro_hygiene),
            Annotatable::Expr(_) => ("expressions", sym::proc_macro_hygiene),
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
        if self.cx.ecfg.proc_macro_hygiene() {
            return
        }
        let fragment = match fragment {
            Some(fragment) => fragment,
            None => return,
        };

        fragment.visit_with(&mut DisallowMacros {
            span,
            parse_sess: self.cx.parse_sess,
        });

        struct DisallowMacros<'a> {
            span: Span,
            parse_sess: &'a ParseSess,
        }

        impl<'ast, 'a> Visitor<'ast> for DisallowMacros<'a> {
            fn visit_item(&mut self, i: &'ast ast::Item) {
                if let ast::ItemKind::MacroDef(_) = i.node {
                    emit_feature_err(
                        self.parse_sess,
                        sym::proc_macro_hygiene,
                        self.span,
                        GateIssue::Language,
                        "procedural macros cannot expand to macro definitions",
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
        let kind = invoc.fragment_kind;
        let (mac, span) = match invoc.kind {
            InvocationKind::Bang { mac, span } => (mac, span),
            _ => unreachable!(),
        };
        let path = &mac.node.path;

        let validate = |this: &mut Self| {
            // feature-gate the macro invocation
            if let Some((feature, issue)) = ext.unstable_feature {
                let crate_span = this.cx.current_expansion.crate_span.unwrap();
                // don't stability-check macros in the same crate
                // (the only time this is null is for syntax extensions registered as macros)
                if ext.def_info.map_or(false, |(_, def_span)| !crate_span.contains(def_span))
                    && !span.allows_unstable(feature)
                    && this.cx.ecfg.features.map_or(true, |feats| {
                    // macro features will count as lib features
                    !feats.declared_lib_features.iter().any(|&(feat, _)| feat == feature)
                }) {
                    let explain = format!("macro {}! is unstable", path);
                    emit_feature_err(this.cx.parse_sess, feature, span,
                                     GateIssue::Library(Some(issue)), &explain);
                    this.cx.trace_macros_diag();
                }
            }

            Ok(())
        };

        let opt_expanded = match &ext.kind {
            SyntaxExtensionKind::LegacyBang(expander) => {
                if let Err(dummy_span) = validate(self) {
                    dummy_span
                } else {
                    kind.make_from(expander.expand(
                        self.cx,
                        span,
                        mac.node.stream(),
                        ext.def_info.map(|(_, s)| s),
                    ))
                }
            }

            SyntaxExtensionKind::Attr(..) |
            SyntaxExtensionKind::LegacyAttr(..) |
            SyntaxExtensionKind::NonMacroAttr { .. } => {
                self.cx.span_err(path.span,
                                 &format!("`{}` can only be used in attributes", path));
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }

            SyntaxExtensionKind::Derive(..) | SyntaxExtensionKind::LegacyDerive(..) => {
                self.cx.span_err(path.span, &format!("`{}` is a derive macro", path));
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }

            SyntaxExtensionKind::Bang(expander) => {
                self.gate_proc_macro_expansion_kind(span, kind);
                let tok_result = expander.expand(self.cx, span, mac.node.stream());
                let result = self.parse_ast_fragment(tok_result, kind, path, span);
                self.gate_proc_macro_expansion(span, &result);
                result
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
        if self.cx.ecfg.proc_macro_hygiene() {
            return
        }
        emit_feature_err(
            self.cx.parse_sess,
            sym::proc_macro_hygiene,
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

        match &ext.kind {
            SyntaxExtensionKind::Derive(expander) |
            SyntaxExtensionKind::LegacyDerive(expander) => {
                let meta = ast::MetaItem { node: ast::MetaItemKind::Word, span: path.span, path };
                let span = meta.span.with_ctxt(self.cx.backtrace());
                let items = expander.expand(self.cx, span, &meta, item);
                Some(invoc.fragment_kind.expect_from_annotatables(items))
            }
            _ => {
                let msg = &format!("macro `{}` may not be used for derive attributes", path);
                self.cx.span_err(path.span, msg);
                self.cx.trace_macros_diag();
                invoc.fragment_kind.dummy(path.span)
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
                let mut items = SmallVec::new();
                while let Some(item) = self.parse_item()? {
                    items.push(item);
                }
                AstFragment::Items(items)
            }
            AstFragmentKind::TraitItems => {
                let mut items = SmallVec::new();
                while self.token != token::Eof {
                    items.push(self.parse_trait_item(&mut false)?);
                }
                AstFragment::TraitItems(items)
            }
            AstFragmentKind::ImplItems => {
                let mut items = SmallVec::new();
                while self.token != token::Eof {
                    items.push(self.parse_impl_item(&mut false)?);
                }
                AstFragment::ImplItems(items)
            }
            AstFragmentKind::ForeignItems => {
                let mut items = SmallVec::new();
                while self.token != token::Eof {
                    items.push(self.parse_foreign_item()?);
                }
                AstFragment::ForeignItems(items)
            }
            AstFragmentKind::Stmts => {
                let mut stmts = SmallVec::new();
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
            AstFragmentKind::Pat => AstFragment::Pat(self.parse_pat(None)?),
        })
    }

    pub fn ensure_complete_parse(&mut self, macro_path: &Path, kind_name: &str, span: Span) {
        if self.token != token::Eof {
            let msg = format!("macro expansion ignores token `{}` and any following",
                              self.this_token_to_string());
            // Avoid emitting backtrace info twice.
            let def_site_span = self.token.span.with_ctxt(SyntaxContext::empty());
            let mut err = self.diagnostic().struct_span_err(def_site_span, &msg);
            err.span_label(span, "caused by the macro expansion here");
            let msg = format!(
                "the usage of `{}!` is likely invalid in {} context",
                macro_path,
                kind_name,
            );
            err.note(&msg);
            let semi_span = self.sess.source_map().next_point(span);

            let semi_full_span = semi_span.to(self.sess.source_map().next_point(semi_span));
            match self.sess.source_map().span_to_snippet(semi_full_span) {
                Ok(ref snippet) if &snippet[..] != ";" && kind_name == "expression" => {
                    err.span_suggestion(
                        semi_span,
                        "you might be missing a semicolon here",
                        ";".to_owned(),
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => {}
            }
            err.emit();
        }
    }
}

struct InvocationCollector<'a, 'b> {
    cx: &'a mut ExtCtxt<'b>,
    cfg: StripUnconfigured<'a>,
    invocations: Vec<Invocation>,
    monotonic: bool,
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

    fn collect_bang(&mut self, mac: ast::Mac, span: Span, kind: AstFragmentKind) -> AstFragment {
        self.collect(kind, InvocationKind::Bang { mac, span })
    }

    fn collect_attr(&mut self,
                    attr: Option<ast::Attribute>,
                    traits: Vec<Path>,
                    item: Annotatable,
                    kind: AstFragmentKind,
                    after_derive: bool)
                    -> AstFragment {
        self.collect(kind, InvocationKind::Attr { attr, traits, item, after_derive })
    }

    fn find_attr_invoc(&self, attrs: &mut Vec<ast::Attribute>, after_derive: &mut bool)
                       -> Option<ast::Attribute> {
        let attr = attrs.iter()
                        .position(|a| {
                            if a.path == sym::derive {
                                *after_derive = true;
                            }
                            !attr::is_known(a) && !is_builtin_attr(a)
                        })
                        .map(|i| attrs.remove(i));
        if let Some(attr) = &attr {
            if !self.cx.ecfg.enable_custom_inner_attributes() &&
               attr.style == ast::AttrStyle::Inner && attr.path != sym::test {
                emit_feature_err(&self.cx.parse_sess, sym::custom_inner_attributes,
                                 attr.span, GateIssue::Language,
                                 "non-builtin inner attributes are unstable");
            }
        }
        attr
    }

    /// If `item` is an attr invocation, remove and return the macro attribute and derive traits.
    fn classify_item<T>(&mut self, item: &mut T)
                        -> (Option<ast::Attribute>, Vec<Path>, /* after_derive */ bool)
        where T: HasAttrs,
    {
        let (mut attr, mut traits, mut after_derive) = (None, Vec::new(), false);

        item.visit_attrs(|mut attrs| {
            attr = self.find_attr_invoc(&mut attrs, &mut after_derive);
            traits = collect_derives(&mut self.cx, &mut attrs);
        });

        (attr, traits, after_derive)
    }

    /// Alternative to `classify_item()` that ignores `#[derive]` so invocations fallthrough
    /// to the unused-attributes lint (making it an error on statements and expressions
    /// is a breaking change)
    fn classify_nonitem<T: HasAttrs>(&mut self, nonitem: &mut T)
                                     -> (Option<ast::Attribute>, /* after_derive */ bool) {
        let (mut attr, mut after_derive) = (None, false);

        nonitem.visit_attrs(|mut attrs| {
            attr = self.find_attr_invoc(&mut attrs, &mut after_derive);
        });

        (attr, after_derive)
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
            if attr.path == sym::derive {
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

impl<'a, 'b> MutVisitor for InvocationCollector<'a, 'b> {
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        visit_clobber(expr.deref_mut(), |mut expr| {
            self.cfg.configure_expr_kind(&mut expr.node);

            // ignore derives so they remain unused
            let (attr, after_derive) = self.classify_nonitem(&mut expr);

            if attr.is_some() {
                // Collect the invoc regardless of whether or not attributes are permitted here
                // expansion will eat the attribute so it won't error later.
                attr.as_ref().map(|a| self.cfg.maybe_emit_expr_attr_err(a));

                // AstFragmentKind::Expr requires the macro to emit an expression.
                return self.collect_attr(attr, vec![], Annotatable::Expr(P(expr)),
                                          AstFragmentKind::Expr, after_derive)
                    .make_expr()
                    .into_inner()
            }

            if let ast::ExprKind::Mac(mac) = expr.node {
                self.check_attributes(&expr.attrs);
                self.collect_bang(mac, expr.span, AstFragmentKind::Expr)
                    .make_expr()
                    .into_inner()
            } else {
                noop_visit_expr(&mut expr, self);
                expr
            }
        });
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let expr = configure!(self, expr);
        expr.filter_map(|mut expr| {
            self.cfg.configure_expr_kind(&mut expr.node);

            // Ignore derives so they remain unused.
            let (attr, after_derive) = self.classify_nonitem(&mut expr);

            if attr.is_some() {
                attr.as_ref().map(|a| self.cfg.maybe_emit_expr_attr_err(a));

                return self.collect_attr(attr, vec![], Annotatable::Expr(P(expr)),
                                         AstFragmentKind::OptExpr, after_derive)
                    .make_opt_expr()
                    .map(|expr| expr.into_inner())
            }

            if let ast::ExprKind::Mac(mac) = expr.node {
                self.check_attributes(&expr.attrs);
                self.collect_bang(mac, expr.span, AstFragmentKind::OptExpr)
                    .make_opt_expr()
                    .map(|expr| expr.into_inner())
            } else {
                Some({ noop_visit_expr(&mut expr, self); expr })
            }
        })
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        self.cfg.configure_pat(pat);
        match pat.node {
            PatKind::Mac(_) => {}
            _ => return noop_visit_pat(pat, self),
        }

        visit_clobber(pat, |mut pat| {
            match mem::replace(&mut pat.node, PatKind::Wild) {
                PatKind::Mac(mac) =>
                    self.collect_bang(mac, pat.span, AstFragmentKind::Pat).make_pat(),
                _ => unreachable!(),
            }
        });
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        let mut stmt = configure!(self, stmt);

        // we'll expand attributes on expressions separately
        if !stmt.is_expr() {
            let (attr, derives, after_derive) = if stmt.is_item() {
                self.classify_item(&mut stmt)
            } else {
                // ignore derives on non-item statements so it falls through
                // to the unused-attributes lint
                let (attr, after_derive) = self.classify_nonitem(&mut stmt);
                (attr, vec![], after_derive)
            };

            if attr.is_some() || !derives.is_empty() {
                return self.collect_attr(attr, derives, Annotatable::Stmt(P(stmt)),
                                         AstFragmentKind::Stmts, after_derive).make_stmts();
            }
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
        noop_flat_map_stmt_kind(node, self).into_iter().map(|node| {
            ast::Stmt { id, node, span }
        }).collect()

    }

    fn visit_block(&mut self, block: &mut P<Block>) {
        let old_directory_ownership = self.cx.current_expansion.directory_ownership;
        self.cx.current_expansion.directory_ownership = DirectoryOwnership::UnownedViaBlock;
        noop_visit_block(block, self);
        self.cx.current_expansion.directory_ownership = old_directory_ownership;
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        let mut item = configure!(self, item);

        let (attr, traits, after_derive) = self.classify_item(&mut item);
        if attr.is_some() || !traits.is_empty() {
            return self.collect_attr(attr, traits, Annotatable::Item(item),
                                     AstFragmentKind::Items, after_derive).make_items();
        }

        match item.node {
            ast::ItemKind::Mac(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.node {
                    ItemKind::Mac(mac) => self.collect(
                        AstFragmentKind::Items, InvocationKind::Bang { mac, span: item.span }
                    ).make_items(),
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(ast::Mod { inner, .. }) => {
                if item.ident == Ident::invalid() {
                    return noop_flat_map_item(item, self);
                }

                let orig_directory_ownership = self.cx.current_expansion.directory_ownership;
                let mut module = (*self.cx.current_expansion.module).clone();
                module.mod_path.push(item.ident);

                // Detect if this is an inline module (`mod m { ... }` as opposed to `mod m;`).
                // In the non-inline case, `inner` is never the dummy span (cf. `parse_item_mod`).
                // Thus, if `inner` is the dummy span, we know the module is inline.
                let inline_module = item.span.contains(inner) || inner.is_dummy();

                if inline_module {
                    if let Some(path) = attr::first_attr_value_str_by_name(&item.attrs, sym::path) {
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
                let result = noop_flat_map_item(item, self);
                self.cx.current_expansion.module = orig_module;
                self.cx.current_expansion.directory_ownership = orig_directory_ownership;
                result
            }

            _ => noop_flat_map_item(item, self),
        }
    }

    fn flat_map_trait_item(&mut self, item: ast::TraitItem) -> SmallVec<[ast::TraitItem; 1]> {
        let mut item = configure!(self, item);

        let (attr, traits, after_derive) = self.classify_item(&mut item);
        if attr.is_some() || !traits.is_empty() {
            return self.collect_attr(attr, traits, Annotatable::TraitItem(P(item)),
                                     AstFragmentKind::TraitItems, after_derive).make_trait_items()
        }

        match item.node {
            ast::TraitItemKind::Macro(mac) => {
                let ast::TraitItem { attrs, span, .. } = item;
                self.check_attributes(&attrs);
                self.collect_bang(mac, span, AstFragmentKind::TraitItems).make_trait_items()
            }
            _ => noop_flat_map_trait_item(item, self),
        }
    }

    fn flat_map_impl_item(&mut self, item: ast::ImplItem) -> SmallVec<[ast::ImplItem; 1]> {
        let mut item = configure!(self, item);

        let (attr, traits, after_derive) = self.classify_item(&mut item);
        if attr.is_some() || !traits.is_empty() {
            return self.collect_attr(attr, traits, Annotatable::ImplItem(P(item)),
                                     AstFragmentKind::ImplItems, after_derive).make_impl_items();
        }

        match item.node {
            ast::ImplItemKind::Macro(mac) => {
                let ast::ImplItem { attrs, span, .. } = item;
                self.check_attributes(&attrs);
                self.collect_bang(mac, span, AstFragmentKind::ImplItems).make_impl_items()
            }
            _ => noop_flat_map_impl_item(item, self),
        }
    }

    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        match ty.node {
            ast::TyKind::Mac(_) => {}
            _ => return noop_visit_ty(ty, self),
        };

        visit_clobber(ty, |mut ty| {
            match mem::replace(&mut ty.node, ast::TyKind::Err) {
                ast::TyKind::Mac(mac) =>
                    self.collect_bang(mac, ty.span, AstFragmentKind::Ty).make_ty(),
                _ => unreachable!(),
            }
        });
    }

    fn visit_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        self.cfg.configure_foreign_mod(foreign_mod);
        noop_visit_foreign_mod(foreign_mod, self);
    }

    fn flat_map_foreign_item(&mut self, mut foreign_item: ast::ForeignItem)
        -> SmallVec<[ast::ForeignItem; 1]>
    {
        let (attr, traits, after_derive) = self.classify_item(&mut foreign_item);

        if attr.is_some() || !traits.is_empty() {
            return self.collect_attr(attr, traits, Annotatable::ForeignItem(P(foreign_item)),
                                     AstFragmentKind::ForeignItems, after_derive)
                                     .make_foreign_items();
        }

        if let ast::ForeignItemKind::Macro(mac) = foreign_item.node {
            self.check_attributes(&foreign_item.attrs);
            return self.collect_bang(mac, foreign_item.span, AstFragmentKind::ForeignItems)
                .make_foreign_items();
        }

        noop_flat_map_foreign_item(foreign_item, self)
    }

    fn visit_item_kind(&mut self, item: &mut ast::ItemKind) {
        match item {
            ast::ItemKind::MacroDef(..) => {}
            _ => {
                self.cfg.configure_item_kind(item);
                noop_visit_item_kind(item, self);
            }
        }
    }

    fn visit_generic_params(&mut self, params: &mut Vec<ast::GenericParam>) {
        self.cfg.configure_generic_params(params);
        noop_visit_generic_params(params, self);
    }

    fn visit_attribute(&mut self, at: &mut ast::Attribute) {
        // turn `#[doc(include="filename")]` attributes into `#[doc(include(file="filename",
        // contents="file contents")]` attributes
        if !at.check_name(sym::doc) {
            return noop_visit_attribute(at, self);
        }

        if let Some(list) = at.meta_item_list() {
            if !list.iter().any(|it| it.check_name(sym::include)) {
                return noop_visit_attribute(at, self);
            }

            let mut items = vec![];

            for mut it in list {
                if !it.check_name(sym::include) {
                    items.push({ noop_visit_meta_list_item(&mut it, self); it });
                    continue;
                }

                if let Some(file) = it.value_str() {
                    let err_count = self.cx.parse_sess.span_diagnostic.err_count();
                    self.check_attribute(&at);
                    if self.cx.parse_sess.span_diagnostic.err_count() > err_count {
                        // avoid loading the file if they haven't enabled the feature
                        return noop_visit_attribute(at, self);
                    }

                    let filename = self.cx.root_path.join(file.to_string());
                    match fs::read_to_string(&filename) {
                        Ok(src) => {
                            let src_interned = Symbol::intern(&src);

                            // Add this input file to the code map to make it available as
                            // dependency information
                            self.cx.source_map().new_source_file(filename.into(), src);

                            let include_info = vec![
                                ast::NestedMetaItem::MetaItem(
                                    attr::mk_name_value_item_str(
                                        Ident::with_empty_ctxt(sym::file),
                                        dummy_spanned(file),
                                    ),
                                ),
                                ast::NestedMetaItem::MetaItem(
                                    attr::mk_name_value_item_str(
                                        Ident::with_empty_ctxt(sym::contents),
                                        dummy_spanned(src_interned),
                                    ),
                                ),
                            ];

                            let include_ident = Ident::with_empty_ctxt(sym::include);
                            let item = attr::mk_list_item(DUMMY_SP, include_ident, include_info);
                            items.push(ast::NestedMetaItem::MetaItem(item));
                        }
                        Err(e) => {
                            let lit = it
                                .meta_item()
                                .and_then(|item| item.name_value_literal())
                                .unwrap();

                            if e.kind() == ErrorKind::InvalidData {
                                self.cx
                                    .struct_span_err(
                                        lit.span,
                                        &format!("{} wasn't a utf-8 file", filename.display()),
                                    )
                                    .span_label(lit.span, "contains invalid utf-8")
                                    .emit();
                            } else {
                                let mut err = self.cx.struct_span_err(
                                    lit.span,
                                    &format!("couldn't read {}: {}", filename.display(), e),
                                );
                                err.span_label(lit.span, "couldn't read file");

                                if e.kind() == ErrorKind::NotFound {
                                    err.help("external doc paths are relative to the crate root");
                                }

                                err.emit();
                            }
                        }
                    }
                } else {
                    let mut err = self.cx.struct_span_err(
                        it.span(),
                        &format!("expected path to external documentation"),
                    );

                    // Check if the user erroneously used `doc(include(...))` syntax.
                    let literal = it.meta_item_list().and_then(|list| {
                        if list.len() == 1 {
                            list[0].literal().map(|literal| &literal.node)
                        } else {
                            None
                        }
                    });

                    let (path, applicability) = match &literal {
                        Some(LitKind::Str(path, ..)) => {
                            (path.to_string(), Applicability::MachineApplicable)
                        }
                        _ => (String::from("<path>"), Applicability::HasPlaceholders),
                    };

                    err.span_suggestion(
                        it.span(),
                        "provide a file path with `=`",
                        format!("include = \"{}\"", path),
                        applicability,
                    );

                    err.emit();
                }
            }

            let meta = attr::mk_list_item(DUMMY_SP, Ident::with_empty_ctxt(sym::doc), items);
            match at.style {
                ast::AttrStyle::Inner => *at = attr::mk_spanned_attr_inner(at.span, at.id, meta),
                ast::AttrStyle::Outer => *at = attr::mk_spanned_attr_outer(at.span, at.id, meta),
            }
        } else {
            noop_visit_attribute(at, self)
        }
    }

    fn visit_id(&mut self, id: &mut ast::NodeId) {
        if self.monotonic {
            debug_assert_eq!(*id, ast::DUMMY_NODE_ID);
            *id = self.cx.resolver.next_node_id()
        }
    }

    fn visit_fn_decl(&mut self, mut fn_decl: &mut P<ast::FnDecl>) {
        self.cfg.configure_fn_decl(&mut fn_decl);
        noop_visit_fn_decl(fn_decl, self);
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
        fn enable_asm = asm,
        fn enable_custom_test_frameworks = custom_test_frameworks,
        fn enable_global_asm = global_asm,
        fn enable_log_syntax = log_syntax,
        fn enable_concat_idents = concat_idents,
        fn enable_trace_macros = trace_macros,
        fn enable_allow_internal_unstable = allow_internal_unstable,
        fn enable_format_args_nl = format_args_nl,
        fn macros_in_extern_enabled = macros_in_extern,
        fn proc_macro_hygiene = proc_macro_hygiene,
    }

    fn enable_custom_inner_attributes(&self) -> bool {
        self.features.map_or(false, |features| {
            features.custom_inner_attributes || features.custom_attribute || features.rustc_attrs
        })
    }
}

// A Marker adds the given mark to the syntax context.
#[derive(Debug)]
pub struct Marker(pub Mark);

impl MutVisitor for Marker {
    fn visit_span(&mut self, span: &mut Span) {
        *span = span.apply_mark(self.0)
    }

    fn visit_mac(&mut self, mac: &mut ast::Mac) {
        noop_visit_mac(mac, self)
    }
}
