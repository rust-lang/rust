use crate::base::*;
use crate::config::StripUnconfigured;
use crate::hygiene::{ExpnData, ExpnId, ExpnKind, SyntaxContext};
use crate::mbe::macro_rules::annotate_err_with_kind;
use crate::placeholders::{placeholder, PlaceholderExpander};
use crate::proc_macro::collect_derives;

use rustc_ast_pretty::pprust;
use rustc_attr::{self as attr, is_builtin_attr, HasAttrs};
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, FatalError, PResult};
use rustc_feature::Features;
use rustc_parse::configure;
use rustc_parse::parser::Parser;
use rustc_parse::validate_attr;
use rustc_parse::DirectoryOwnership;
use rustc_session::parse::{feature_err, ParseSess};
use rustc_span::source_map::respan;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{FileName, Span, DUMMY_SP};
use syntax::ast::{self, AttrItem, Block, Ident, LitKind, NodeId, PatKind, Path};
use syntax::ast::{ItemKind, MacArgs, MacStmtStyle, StmtKind};
use syntax::mut_visit::*;
use syntax::ptr::P;
use syntax::token;
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax::util::map_in_place::MapInPlace;
use syntax::visit::{self, AssocCtxt, Visitor};

use smallvec::{smallvec, SmallVec};
use std::io::ErrorKind;
use std::ops::DerefMut;
use std::path::PathBuf;
use std::rc::Rc;
use std::{iter, mem, slice};

macro_rules! ast_fragments {
    (
        $($Kind:ident($AstTy:ty) {
            $kind_name:expr;
            $(one fn $mut_visit_ast:ident; fn $visit_ast:ident;)?
            $(many fn $flat_map_ast_elt:ident; fn $visit_ast_elt:ident($($args:tt)*);)?
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
            pub fn add_placeholders(&mut self, placeholders: &[NodeId]) {
                if placeholders.is_empty() {
                    return;
                }
                match self {
                    $($(AstFragment::$Kind(ast) => ast.extend(placeholders.iter().flat_map(|id| {
                        // We are repeating through arguments with `many`, to do that we have to
                        // mention some macro variable from those arguments even if it's not used.
                        macro _repeating($flat_map_ast_elt) {}
                        placeholder(AstFragmentKind::$Kind, *id, None).$make_ast()
                    })),)?)*
                    _ => panic!("unexpected AST fragment kind")
                }
            }

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
                        visitor.$visit_ast_elt(ast_elt, $($args)*);
                    })?)*
                }
            }
        }

        impl<'a> MacResult for crate::mbe::macro_rules::ParserAnyMacro<'a> {
            $(fn $make_ast(self: Box<crate::mbe::macro_rules::ParserAnyMacro<'a>>)
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
        "statement"; many fn flat_map_stmt; fn visit_stmt(); fn make_stmts;
    }
    Items(SmallVec<[P<ast::Item>; 1]>) {
        "item"; many fn flat_map_item; fn visit_item(); fn make_items;
    }
    TraitItems(SmallVec<[P<ast::AssocItem>; 1]>) {
        "trait item";
        many fn flat_map_trait_item;
        fn visit_assoc_item(AssocCtxt::Trait);
        fn make_trait_items;
    }
    ImplItems(SmallVec<[P<ast::AssocItem>; 1]>) {
        "impl item";
        many fn flat_map_impl_item;
        fn visit_assoc_item(AssocCtxt::Impl);
        fn make_impl_items;
    }
    ForeignItems(SmallVec<[P<ast::ForeignItem>; 1]>) {
        "foreign item";
        many fn flat_map_foreign_item;
        fn visit_foreign_item();
        fn make_foreign_items;
    }
    Arms(SmallVec<[ast::Arm; 1]>) {
        "match arm"; many fn flat_map_arm; fn visit_arm(); fn make_arms;
    }
    Fields(SmallVec<[ast::Field; 1]>) {
        "field expression"; many fn flat_map_field; fn visit_field(); fn make_fields;
    }
    FieldPats(SmallVec<[ast::FieldPat; 1]>) {
        "field pattern";
        many fn flat_map_field_pattern;
        fn visit_field_pattern();
        fn make_field_patterns;
    }
    GenericParams(SmallVec<[ast::GenericParam; 1]>) {
        "generic parameter";
        many fn flat_map_generic_param;
        fn visit_generic_param();
        fn make_generic_params;
    }
    Params(SmallVec<[ast::Param; 1]>) {
        "function parameter"; many fn flat_map_param; fn visit_param(); fn make_params;
    }
    StructFields(SmallVec<[ast::StructField; 1]>) {
        "field";
        many fn flat_map_struct_field;
        fn visit_struct_field();
        fn make_struct_fields;
    }
    Variants(SmallVec<[ast::Variant; 1]>) {
        "variant"; many fn flat_map_variant; fn visit_variant(); fn make_variants;
    }
}

impl AstFragmentKind {
    fn dummy(self, span: Span) -> AstFragment {
        self.make_from(DummyResult::any(span)).expect("couldn't create a dummy AST fragment")
    }

    fn expect_from_annotatables<I: IntoIterator<Item = Annotatable>>(
        self,
        items: I,
    ) -> AstFragment {
        let mut items = items.into_iter();
        match self {
            AstFragmentKind::Arms => {
                AstFragment::Arms(items.map(Annotatable::expect_arm).collect())
            }
            AstFragmentKind::Fields => {
                AstFragment::Fields(items.map(Annotatable::expect_field).collect())
            }
            AstFragmentKind::FieldPats => {
                AstFragment::FieldPats(items.map(Annotatable::expect_field_pattern).collect())
            }
            AstFragmentKind::GenericParams => {
                AstFragment::GenericParams(items.map(Annotatable::expect_generic_param).collect())
            }
            AstFragmentKind::Params => {
                AstFragment::Params(items.map(Annotatable::expect_param).collect())
            }
            AstFragmentKind::StructFields => {
                AstFragment::StructFields(items.map(Annotatable::expect_struct_field).collect())
            }
            AstFragmentKind::Variants => {
                AstFragment::Variants(items.map(Annotatable::expect_variant).collect())
            }
            AstFragmentKind::Items => {
                AstFragment::Items(items.map(Annotatable::expect_item).collect())
            }
            AstFragmentKind::ImplItems => {
                AstFragment::ImplItems(items.map(Annotatable::expect_impl_item).collect())
            }
            AstFragmentKind::TraitItems => {
                AstFragment::TraitItems(items.map(Annotatable::expect_trait_item).collect())
            }
            AstFragmentKind::ForeignItems => {
                AstFragment::ForeignItems(items.map(Annotatable::expect_foreign_item).collect())
            }
            AstFragmentKind::Stmts => {
                AstFragment::Stmts(items.map(Annotatable::expect_stmt).collect())
            }
            AstFragmentKind::Expr => AstFragment::Expr(
                items.next().expect("expected exactly one expression").expect_expr(),
            ),
            AstFragmentKind::OptExpr => {
                AstFragment::OptExpr(items.next().map(Annotatable::expect_expr))
            }
            AstFragmentKind::Pat | AstFragmentKind::Ty => {
                panic!("patterns and types aren't annotatable")
            }
        }
    }
}

pub struct Invocation {
    pub kind: InvocationKind,
    pub fragment_kind: AstFragmentKind,
    pub expansion_data: ExpansionData,
}

pub enum InvocationKind {
    Bang {
        mac: ast::Mac,
        span: Span,
    },
    Attr {
        attr: ast::Attribute,
        item: Annotatable,
        // Required for resolving derive helper attributes.
        derives: Vec<Path>,
        // We temporarily report errors for attribute macros placed after derives
        after_derive: bool,
    },
    Derive {
        path: Path,
        item: Annotatable,
    },
    /// "Invocation" that contains all derives from an item,
    /// broken into multiple `Derive` invocations when expanded.
    /// FIXME: Find a way to remove it.
    DeriveContainer {
        derives: Vec<Path>,
        item: Annotatable,
    },
}

impl InvocationKind {
    fn placeholder_visibility(&self) -> Option<ast::Visibility> {
        // HACK: For unnamed fields placeholders should have the same visibility as the actual
        // fields because for tuple structs/variants resolve determines visibilities of their
        // constructor using these field visibilities before attributes on them are are expanded.
        // The assumption is that the attribute expansion cannot change field visibilities,
        // and it holds because only inert attributes are supported in this position.
        match self {
            InvocationKind::Attr { item: Annotatable::StructField(field), .. }
            | InvocationKind::Derive { item: Annotatable::StructField(field), .. }
            | InvocationKind::DeriveContainer { item: Annotatable::StructField(field), .. }
                if field.ident.is_none() =>
            {
                Some(field.vis.clone())
            }
            _ => None,
        }
    }
}

impl Invocation {
    pub fn span(&self) -> Span {
        match &self.kind {
            InvocationKind::Bang { span, .. } => *span,
            InvocationKind::Attr { attr, .. } => attr.span,
            InvocationKind::Derive { path, .. } => path.span,
            InvocationKind::DeriveContainer { item, .. } => item.span(),
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

        let orig_mod_span = krate.module.inner;

        let krate_item = AstFragment::Items(smallvec![P(ast::Item {
            attrs: krate.attrs,
            span: krate.span,
            kind: ast::ItemKind::Mod(krate.module),
            ident: Ident::invalid(),
            id: ast::DUMMY_NODE_ID,
            vis: respan(krate.span.shrink_to_lo(), ast::VisibilityKind::Public),
            tokens: None,
        })]);

        match self.fully_expand_fragment(krate_item).make_items().pop().map(P::into_inner) {
            Some(ast::Item { attrs, kind: ast::ItemKind::Mod(module), .. }) => {
                krate.attrs = attrs;
                krate.module = module;
            }
            None => {
                // Resolution failed so we return an empty expansion
                krate.attrs = vec![];
                krate.module = ast::Mod { inner: orig_mod_span, items: vec![], inline: true };
            }
            Some(ast::Item { span, kind, .. }) => {
                krate.attrs = vec![];
                krate.module = ast::Mod { inner: orig_mod_span, items: vec![], inline: true };
                self.cx.span_err(
                    span,
                    &format!(
                        "expected crate top-level item to be a module after macro expansion, found a {}",
                        kind.descriptive_variant()
                    ),
                );
            }
        };
        self.cx.trace_macros_diag();
        krate
    }

    // Recursively expand all macro invocations in this AST fragment.
    pub fn fully_expand_fragment(&mut self, input_fragment: AstFragment) -> AstFragment {
        let orig_expansion_data = self.cx.current_expansion.clone();
        self.cx.current_expansion.depth = 0;

        // Collect all macro invocations and replace them with placeholders.
        let (mut fragment_with_placeholders, mut invocations) =
            self.collect_invocations(input_fragment, &[]);

        // Optimization: if we resolve all imports now,
        // we'll be able to immediately resolve most of imported macros.
        self.resolve_imports();

        // Resolve paths in all invocations and produce output expanded fragments for them, but
        // do not insert them into our input AST fragment yet, only store in `expanded_fragments`.
        // The output fragments also go through expansion recursively until no invocations are left.
        // Unresolved macros produce dummy outputs as a recovery measure.
        invocations.reverse();
        let mut expanded_fragments = Vec::new();
        let mut undetermined_invocations = Vec::new();
        let (mut progress, mut force) = (false, !self.monotonic);
        loop {
            let invoc = if let Some(invoc) = invocations.pop() {
                invoc
            } else {
                self.resolve_imports();
                if undetermined_invocations.is_empty() {
                    break;
                }
                invocations = mem::take(&mut undetermined_invocations);
                force = !mem::replace(&mut progress, false);
                continue;
            };

            let eager_expansion_root =
                if self.monotonic { invoc.expansion_data.id } else { orig_expansion_data.id };
            let res = match self.cx.resolver.resolve_macro_invocation(
                &invoc,
                eager_expansion_root,
                force,
            ) {
                Ok(res) => res,
                Err(Indeterminate) => {
                    undetermined_invocations.push(invoc);
                    continue;
                }
            };

            progress = true;
            let ExpansionData { depth, id: expn_id, .. } = invoc.expansion_data;
            self.cx.current_expansion = invoc.expansion_data.clone();

            // FIXME(jseyfried): Refactor out the following logic
            let (expanded_fragment, new_invocations) = match res {
                InvocationRes::Single(ext) => {
                    let fragment = self.expand_invoc(invoc, &ext.kind);
                    self.collect_invocations(fragment, &[])
                }
                InvocationRes::DeriveContainer(_exts) => {
                    // FIXME: Consider using the derive resolutions (`_exts`) immediately,
                    // instead of enqueuing the derives to be resolved again later.
                    let (derives, item) = match invoc.kind {
                        InvocationKind::DeriveContainer { derives, item } => (derives, item),
                        _ => unreachable!(),
                    };
                    if !item.derive_allowed() {
                        self.error_derive_forbidden_on_non_adt(&derives, &item);
                    }

                    let mut item = self.fully_configure(item);
                    item.visit_attrs(|attrs| attrs.retain(|a| !a.has_name(sym::derive)));

                    let mut derive_placeholders = Vec::with_capacity(derives.len());
                    invocations.reserve(derives.len());
                    for path in derives {
                        let expn_id = ExpnId::fresh(None);
                        derive_placeholders.push(NodeId::placeholder_from_expn_id(expn_id));
                        invocations.push(Invocation {
                            kind: InvocationKind::Derive { path, item: item.clone() },
                            fragment_kind: invoc.fragment_kind,
                            expansion_data: ExpansionData {
                                id: expn_id,
                                ..invoc.expansion_data.clone()
                            },
                        });
                    }
                    let fragment =
                        invoc.fragment_kind.expect_from_annotatables(::std::iter::once(item));
                    self.collect_invocations(fragment, &derive_placeholders)
                }
            };

            if expanded_fragments.len() < depth {
                expanded_fragments.push(Vec::new());
            }
            expanded_fragments[depth - 1].push((expn_id, expanded_fragment));
            if !self.cx.ecfg.single_step {
                invocations.extend(new_invocations.into_iter().rev());
            }
        }

        self.cx.current_expansion = orig_expansion_data;

        // Finally incorporate all the expanded macros into the input AST fragment.
        let mut placeholder_expander = PlaceholderExpander::new(self.cx, self.monotonic);
        while let Some(expanded_fragments) = expanded_fragments.pop() {
            for (expn_id, expanded_fragment) in expanded_fragments.into_iter().rev() {
                placeholder_expander
                    .add(NodeId::placeholder_from_expn_id(expn_id), expanded_fragment);
            }
        }
        fragment_with_placeholders.mut_visit_with(&mut placeholder_expander);
        fragment_with_placeholders
    }

    fn error_derive_forbidden_on_non_adt(&self, derives: &[Path], item: &Annotatable) {
        let attr =
            attr::find_by_name(item.attrs(), sym::derive).expect("`derive` attribute should exist");
        let span = attr.span;
        let mut err = self
            .cx
            .struct_span_err(span, "`derive` may only be applied to structs, enums and unions");
        if let ast::AttrStyle::Inner = attr.style {
            let trait_list = derives.iter().map(|t| pprust::path_to_string(t)).collect::<Vec<_>>();
            let suggestion = format!("#[derive({})]", trait_list.join(", "));
            err.span_suggestion(
                span,
                "try an outer attribute",
                suggestion,
                // We don't 𝑘𝑛𝑜𝑤 that the following item is an ADT
                Applicability::MaybeIncorrect,
            );
        }
        err.emit();
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
    fn collect_invocations(
        &mut self,
        mut fragment: AstFragment,
        extra_placeholders: &[NodeId],
    ) -> (AstFragment, Vec<Invocation>) {
        // Resolve `$crate`s in the fragment for pretty-printing.
        self.cx.resolver.resolve_dollar_crates();

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
            fragment.add_placeholders(extra_placeholders);
            collector.invocations
        };

        if self.monotonic {
            self.cx
                .resolver
                .visit_ast_fragment_with_placeholders(self.cx.current_expansion.id, &fragment);
        }

        (fragment, invocations)
    }

    fn fully_configure(&mut self, item: Annotatable) -> Annotatable {
        let mut cfg =
            StripUnconfigured { sess: self.cx.parse_sess, features: self.cx.ecfg.features };
        // Since the item itself has already been configured by the InvocationCollector,
        // we know that fold result vector will contain exactly one element
        match item {
            Annotatable::Item(item) => Annotatable::Item(cfg.flat_map_item(item).pop().unwrap()),
            Annotatable::TraitItem(item) => {
                Annotatable::TraitItem(cfg.flat_map_trait_item(item).pop().unwrap())
            }
            Annotatable::ImplItem(item) => {
                Annotatable::ImplItem(cfg.flat_map_impl_item(item).pop().unwrap())
            }
            Annotatable::ForeignItem(item) => {
                Annotatable::ForeignItem(cfg.flat_map_foreign_item(item).pop().unwrap())
            }
            Annotatable::Stmt(stmt) => {
                Annotatable::Stmt(stmt.map(|stmt| cfg.flat_map_stmt(stmt).pop().unwrap()))
            }
            Annotatable::Expr(mut expr) => Annotatable::Expr({
                cfg.visit_expr(&mut expr);
                expr
            }),
            Annotatable::Arm(arm) => Annotatable::Arm(cfg.flat_map_arm(arm).pop().unwrap()),
            Annotatable::Field(field) => {
                Annotatable::Field(cfg.flat_map_field(field).pop().unwrap())
            }
            Annotatable::FieldPat(fp) => {
                Annotatable::FieldPat(cfg.flat_map_field_pattern(fp).pop().unwrap())
            }
            Annotatable::GenericParam(param) => {
                Annotatable::GenericParam(cfg.flat_map_generic_param(param).pop().unwrap())
            }
            Annotatable::Param(param) => {
                Annotatable::Param(cfg.flat_map_param(param).pop().unwrap())
            }
            Annotatable::StructField(sf) => {
                Annotatable::StructField(cfg.flat_map_struct_field(sf).pop().unwrap())
            }
            Annotatable::Variant(v) => Annotatable::Variant(cfg.flat_map_variant(v).pop().unwrap()),
        }
    }

    fn error_recursion_limit_reached(&mut self) {
        let expn_data = self.cx.current_expansion.id.expn_data();
        let suggested_limit = self.cx.ecfg.recursion_limit * 2;
        self.cx
            .struct_span_err(
                expn_data.call_site,
                &format!("recursion limit reached while expanding `{}`", expn_data.kind.descr()),
            )
            .help(&format!(
                "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate (`{}`)",
                suggested_limit, self.cx.ecfg.crate_name,
            ))
            .emit();
        self.cx.trace_macros_diag();
        FatalError.raise();
    }

    /// A macro's expansion does not fit in this fragment kind.
    /// For example, a non-type macro in a type position.
    fn error_wrong_fragment_kind(&mut self, kind: AstFragmentKind, mac: &ast::Mac, span: Span) {
        let msg = format!(
            "non-{kind} macro in {kind} position: {path}",
            kind = kind.name(),
            path = pprust::path_to_string(&mac.path),
        );
        self.cx.span_err(span, &msg);
        self.cx.trace_macros_diag();
    }

    fn expand_invoc(&mut self, invoc: Invocation, ext: &SyntaxExtensionKind) -> AstFragment {
        if self.cx.current_expansion.depth > self.cx.ecfg.recursion_limit {
            self.error_recursion_limit_reached();
        }

        let (fragment_kind, span) = (invoc.fragment_kind, invoc.span());
        match invoc.kind {
            InvocationKind::Bang { mac, .. } => match ext {
                SyntaxExtensionKind::Bang(expander) => {
                    self.gate_proc_macro_expansion_kind(span, fragment_kind);
                    let tok_result = expander.expand(self.cx, span, mac.args.inner_tokens());
                    self.parse_ast_fragment(tok_result, fragment_kind, &mac.path, span)
                }
                SyntaxExtensionKind::LegacyBang(expander) => {
                    let prev = self.cx.current_expansion.prior_type_ascription;
                    self.cx.current_expansion.prior_type_ascription = mac.prior_type_ascription;
                    let tok_result = expander.expand(self.cx, span, mac.args.inner_tokens());
                    let result = if let Some(result) = fragment_kind.make_from(tok_result) {
                        result
                    } else {
                        self.error_wrong_fragment_kind(fragment_kind, &mac, span);
                        fragment_kind.dummy(span)
                    };
                    self.cx.current_expansion.prior_type_ascription = prev;
                    result
                }
                _ => unreachable!(),
            },
            InvocationKind::Attr { attr, mut item, .. } => match ext {
                SyntaxExtensionKind::Attr(expander) => {
                    self.gate_proc_macro_input(&item);
                    self.gate_proc_macro_attr_item(span, &item);
                    let item_tok = TokenTree::token(
                        token::Interpolated(Lrc::new(match item {
                            Annotatable::Item(item) => token::NtItem(item),
                            Annotatable::TraitItem(item) => token::NtTraitItem(item),
                            Annotatable::ImplItem(item) => token::NtImplItem(item),
                            Annotatable::ForeignItem(item) => token::NtForeignItem(item),
                            Annotatable::Stmt(stmt) => token::NtStmt(stmt.into_inner()),
                            Annotatable::Expr(expr) => token::NtExpr(expr),
                            Annotatable::Arm(..)
                            | Annotatable::Field(..)
                            | Annotatable::FieldPat(..)
                            | Annotatable::GenericParam(..)
                            | Annotatable::Param(..)
                            | Annotatable::StructField(..)
                            | Annotatable::Variant(..) => panic!("unexpected annotatable"),
                        })),
                        DUMMY_SP,
                    )
                    .into();
                    let item = attr.unwrap_normal_item();
                    if let MacArgs::Eq(..) = item.args {
                        self.cx.span_err(span, "key-value macro attributes are not supported");
                    }
                    let tok_result =
                        expander.expand(self.cx, span, item.args.inner_tokens(), item_tok);
                    self.parse_ast_fragment(tok_result, fragment_kind, &item.path, span)
                }
                SyntaxExtensionKind::LegacyAttr(expander) => {
                    match validate_attr::parse_meta(self.cx.parse_sess, &attr) {
                        Ok(meta) => {
                            let item = expander.expand(self.cx, span, &meta, item);
                            fragment_kind.expect_from_annotatables(item)
                        }
                        Err(mut err) => {
                            err.emit();
                            fragment_kind.dummy(span)
                        }
                    }
                }
                SyntaxExtensionKind::NonMacroAttr { mark_used } => {
                    attr::mark_known(&attr);
                    if *mark_used {
                        attr::mark_used(&attr);
                    }
                    item.visit_attrs(|attrs| attrs.push(attr));
                    fragment_kind.expect_from_annotatables(iter::once(item))
                }
                _ => unreachable!(),
            },
            InvocationKind::Derive { path, item } => match ext {
                SyntaxExtensionKind::Derive(expander)
                | SyntaxExtensionKind::LegacyDerive(expander) => {
                    if !item.derive_allowed() {
                        return fragment_kind.dummy(span);
                    }
                    if let SyntaxExtensionKind::Derive(..) = ext {
                        self.gate_proc_macro_input(&item);
                    }
                    let meta = ast::MetaItem { kind: ast::MetaItemKind::Word, span, path };
                    let items = expander.expand(self.cx, span, &meta, item);
                    fragment_kind.expect_from_annotatables(items)
                }
                _ => unreachable!(),
            },
            InvocationKind::DeriveContainer { .. } => unreachable!(),
        }
    }

    fn gate_proc_macro_attr_item(&self, span: Span, item: &Annotatable) {
        let kind = match item {
            Annotatable::Item(_)
            | Annotatable::TraitItem(_)
            | Annotatable::ImplItem(_)
            | Annotatable::ForeignItem(_) => return,
            Annotatable::Stmt(_) => "statements",
            Annotatable::Expr(_) => "expressions",
            Annotatable::Arm(..)
            | Annotatable::Field(..)
            | Annotatable::FieldPat(..)
            | Annotatable::GenericParam(..)
            | Annotatable::Param(..)
            | Annotatable::StructField(..)
            | Annotatable::Variant(..) => panic!("unexpected annotatable"),
        };
        if self.cx.ecfg.proc_macro_hygiene() {
            return;
        }
        feature_err(
            self.cx.parse_sess,
            sym::proc_macro_hygiene,
            span,
            &format!("custom attributes cannot be applied to {}", kind),
        )
        .emit();
    }

    fn gate_proc_macro_input(&self, annotatable: &Annotatable) {
        struct GateProcMacroInput<'a> {
            parse_sess: &'a ParseSess,
        }

        impl<'ast, 'a> Visitor<'ast> for GateProcMacroInput<'a> {
            fn visit_item(&mut self, item: &'ast ast::Item) {
                match &item.kind {
                    ast::ItemKind::Mod(module) if !module.inline => {
                        feature_err(
                            self.parse_sess,
                            sym::proc_macro_hygiene,
                            item.span,
                            "non-inline modules in proc macro input are unstable",
                        )
                        .emit();
                    }
                    _ => {}
                }

                visit::walk_item(self, item);
            }

            fn visit_mac(&mut self, _: &'ast ast::Mac) {}
        }

        if !self.cx.ecfg.proc_macro_hygiene() {
            annotatable.visit_with(&mut GateProcMacroInput { parse_sess: self.cx.parse_sess });
        }
    }

    fn gate_proc_macro_expansion_kind(&self, span: Span, kind: AstFragmentKind) {
        let kind = match kind {
            AstFragmentKind::Expr | AstFragmentKind::OptExpr => "expressions",
            AstFragmentKind::Pat => "patterns",
            AstFragmentKind::Stmts => "statements",
            AstFragmentKind::Ty
            | AstFragmentKind::Items
            | AstFragmentKind::TraitItems
            | AstFragmentKind::ImplItems
            | AstFragmentKind::ForeignItems => return,
            AstFragmentKind::Arms
            | AstFragmentKind::Fields
            | AstFragmentKind::FieldPats
            | AstFragmentKind::GenericParams
            | AstFragmentKind::Params
            | AstFragmentKind::StructFields
            | AstFragmentKind::Variants => panic!("unexpected AST fragment kind"),
        };
        if self.cx.ecfg.proc_macro_hygiene() {
            return;
        }
        feature_err(
            self.cx.parse_sess,
            sym::proc_macro_hygiene,
            span,
            &format!("procedural macros cannot be expanded to {}", kind),
        )
        .emit();
    }

    fn parse_ast_fragment(
        &mut self,
        toks: TokenStream,
        kind: AstFragmentKind,
        path: &Path,
        span: Span,
    ) -> AstFragment {
        let mut parser = self.cx.new_parser_from_tts(toks);
        match parse_ast_fragment(&mut parser, kind) {
            Ok(fragment) => {
                ensure_complete_parse(&mut parser, path, kind.name(), span);
                fragment
            }
            Err(mut err) => {
                err.set_span(span);
                annotate_err_with_kind(&mut err, kind, span);
                err.emit();
                self.cx.trace_macros_diag();
                kind.dummy(span)
            }
        }
    }
}

pub fn parse_ast_fragment<'a>(
    this: &mut Parser<'a>,
    kind: AstFragmentKind,
) -> PResult<'a, AstFragment> {
    Ok(match kind {
        AstFragmentKind::Items => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_item()? {
                items.push(item);
            }
            AstFragment::Items(items)
        }
        AstFragmentKind::TraitItems => {
            let mut items = SmallVec::new();
            while this.token != token::Eof {
                items.push(this.parse_trait_item(&mut false)?);
            }
            AstFragment::TraitItems(items)
        }
        AstFragmentKind::ImplItems => {
            let mut items = SmallVec::new();
            while this.token != token::Eof {
                items.push(this.parse_impl_item(&mut false)?);
            }
            AstFragment::ImplItems(items)
        }
        AstFragmentKind::ForeignItems => {
            let mut items = SmallVec::new();
            while this.token != token::Eof {
                items.push(this.parse_foreign_item(&mut false)?);
            }
            AstFragment::ForeignItems(items)
        }
        AstFragmentKind::Stmts => {
            let mut stmts = SmallVec::new();
            // Won't make progress on a `}`.
            while this.token != token::Eof && this.token != token::CloseDelim(token::Brace) {
                if let Some(stmt) = this.parse_full_stmt()? {
                    stmts.push(stmt);
                }
            }
            AstFragment::Stmts(stmts)
        }
        AstFragmentKind::Expr => AstFragment::Expr(this.parse_expr()?),
        AstFragmentKind::OptExpr => {
            if this.token != token::Eof {
                AstFragment::OptExpr(Some(this.parse_expr()?))
            } else {
                AstFragment::OptExpr(None)
            }
        }
        AstFragmentKind::Ty => AstFragment::Ty(this.parse_ty()?),
        AstFragmentKind::Pat => AstFragment::Pat(this.parse_pat(None)?),
        AstFragmentKind::Arms
        | AstFragmentKind::Fields
        | AstFragmentKind::FieldPats
        | AstFragmentKind::GenericParams
        | AstFragmentKind::Params
        | AstFragmentKind::StructFields
        | AstFragmentKind::Variants => panic!("unexpected AST fragment kind"),
    })
}

pub fn ensure_complete_parse<'a>(
    this: &mut Parser<'a>,
    macro_path: &Path,
    kind_name: &str,
    span: Span,
) {
    if this.token != token::Eof {
        let token = pprust::token_to_string(&this.token);
        let msg = format!("macro expansion ignores token `{}` and any following", token);
        // Avoid emitting backtrace info twice.
        let def_site_span = this.token.span.with_ctxt(SyntaxContext::root());
        let mut err = this.struct_span_err(def_site_span, &msg);
        err.span_label(span, "caused by the macro expansion here");
        let msg = format!(
            "the usage of `{}!` is likely invalid in {} context",
            pprust::path_to_string(macro_path),
            kind_name,
        );
        err.note(&msg);
        let semi_span = this.sess.source_map().next_point(span);

        let semi_full_span = semi_span.to(this.sess.source_map().next_point(semi_span));
        match this.sess.source_map().span_to_snippet(semi_full_span) {
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

struct InvocationCollector<'a, 'b> {
    cx: &'a mut ExtCtxt<'b>,
    cfg: StripUnconfigured<'a>,
    invocations: Vec<Invocation>,
    monotonic: bool,
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn collect(&mut self, fragment_kind: AstFragmentKind, kind: InvocationKind) -> AstFragment {
        // Expansion data for all the collected invocations is set upon their resolution,
        // with exception of the derive container case which is not resolved and can get
        // its expansion data immediately.
        let expn_data = match &kind {
            InvocationKind::DeriveContainer { item, .. } => Some(ExpnData {
                parent: self.cx.current_expansion.id,
                ..ExpnData::default(
                    ExpnKind::Macro(MacroKind::Attr, sym::derive),
                    item.span(),
                    self.cx.parse_sess.edition,
                )
            }),
            _ => None,
        };
        let expn_id = ExpnId::fresh(expn_data);
        let vis = kind.placeholder_visibility();
        self.invocations.push(Invocation {
            kind,
            fragment_kind,
            expansion_data: ExpansionData {
                id: expn_id,
                depth: self.cx.current_expansion.depth + 1,
                ..self.cx.current_expansion.clone()
            },
        });
        placeholder(fragment_kind, NodeId::placeholder_from_expn_id(expn_id), vis)
    }

    fn collect_bang(&mut self, mac: ast::Mac, span: Span, kind: AstFragmentKind) -> AstFragment {
        self.collect(kind, InvocationKind::Bang { mac, span })
    }

    fn collect_attr(
        &mut self,
        attr: Option<ast::Attribute>,
        derives: Vec<Path>,
        item: Annotatable,
        kind: AstFragmentKind,
        after_derive: bool,
    ) -> AstFragment {
        self.collect(
            kind,
            match attr {
                Some(attr) => InvocationKind::Attr { attr, item, derives, after_derive },
                None => InvocationKind::DeriveContainer { derives, item },
            },
        )
    }

    fn find_attr_invoc(
        &self,
        attrs: &mut Vec<ast::Attribute>,
        after_derive: &mut bool,
    ) -> Option<ast::Attribute> {
        let attr = attrs
            .iter()
            .position(|a| {
                if a.has_name(sym::derive) {
                    *after_derive = true;
                }
                !attr::is_known(a) && !is_builtin_attr(a)
            })
            .map(|i| attrs.remove(i));
        if let Some(attr) = &attr {
            if !self.cx.ecfg.custom_inner_attributes()
                && attr.style == ast::AttrStyle::Inner
                && !attr.has_name(sym::test)
            {
                feature_err(
                    &self.cx.parse_sess,
                    sym::custom_inner_attributes,
                    attr.span,
                    "non-builtin inner attributes are unstable",
                )
                .emit();
            }
        }
        attr
    }

    /// If `item` is an attr invocation, remove and return the macro attribute and derive traits.
    fn classify_item(
        &mut self,
        item: &mut impl HasAttrs,
    ) -> (Option<ast::Attribute>, Vec<Path>, /* after_derive */ bool) {
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
    fn classify_nonitem(
        &mut self,
        nonitem: &mut impl HasAttrs,
    ) -> (Option<ast::Attribute>, /* after_derive */ bool) {
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
            rustc_ast_passes::feature_gate::check_attribute(attr, self.cx.parse_sess, features);
            validate_attr::check_meta(self.cx.parse_sess, attr);

            // macros are expanded before any lint passes so this warning has to be hardcoded
            if attr.has_name(sym::derive) {
                self.cx
                    .struct_span_warn(attr.span, "`#[derive]` does nothing on macro invocations")
                    .note("this may become a hard error in a future release")
                    .emit();
            }
        }
    }
}

impl<'a, 'b> MutVisitor for InvocationCollector<'a, 'b> {
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        visit_clobber(expr.deref_mut(), |mut expr| {
            self.cfg.configure_expr_kind(&mut expr.kind);

            // ignore derives so they remain unused
            let (attr, after_derive) = self.classify_nonitem(&mut expr);

            if attr.is_some() {
                // Collect the invoc regardless of whether or not attributes are permitted here
                // expansion will eat the attribute so it won't error later.
                attr.as_ref().map(|a| self.cfg.maybe_emit_expr_attr_err(a));

                // AstFragmentKind::Expr requires the macro to emit an expression.
                return self
                    .collect_attr(
                        attr,
                        vec![],
                        Annotatable::Expr(P(expr)),
                        AstFragmentKind::Expr,
                        after_derive,
                    )
                    .make_expr()
                    .into_inner();
            }

            if let ast::ExprKind::Mac(mac) = expr.kind {
                self.check_attributes(&expr.attrs);
                self.collect_bang(mac, expr.span, AstFragmentKind::Expr).make_expr().into_inner()
            } else {
                noop_visit_expr(&mut expr, self);
                expr
            }
        });
    }

    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        let mut arm = configure!(self, arm);

        let (attr, traits, after_derive) = self.classify_item(&mut arm);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::Arm(arm),
                    AstFragmentKind::Arms,
                    after_derive,
                )
                .make_arms();
        }

        noop_flat_map_arm(arm, self)
    }

    fn flat_map_field(&mut self, field: ast::Field) -> SmallVec<[ast::Field; 1]> {
        let mut field = configure!(self, field);

        let (attr, traits, after_derive) = self.classify_item(&mut field);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::Field(field),
                    AstFragmentKind::Fields,
                    after_derive,
                )
                .make_fields();
        }

        noop_flat_map_field(field, self)
    }

    fn flat_map_field_pattern(&mut self, fp: ast::FieldPat) -> SmallVec<[ast::FieldPat; 1]> {
        let mut fp = configure!(self, fp);

        let (attr, traits, after_derive) = self.classify_item(&mut fp);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::FieldPat(fp),
                    AstFragmentKind::FieldPats,
                    after_derive,
                )
                .make_field_patterns();
        }

        noop_flat_map_field_pattern(fp, self)
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        let mut p = configure!(self, p);

        let (attr, traits, after_derive) = self.classify_item(&mut p);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::Param(p),
                    AstFragmentKind::Params,
                    after_derive,
                )
                .make_params();
        }

        noop_flat_map_param(p, self)
    }

    fn flat_map_struct_field(&mut self, sf: ast::StructField) -> SmallVec<[ast::StructField; 1]> {
        let mut sf = configure!(self, sf);

        let (attr, traits, after_derive) = self.classify_item(&mut sf);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::StructField(sf),
                    AstFragmentKind::StructFields,
                    after_derive,
                )
                .make_struct_fields();
        }

        noop_flat_map_struct_field(sf, self)
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        let mut variant = configure!(self, variant);

        let (attr, traits, after_derive) = self.classify_item(&mut variant);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::Variant(variant),
                    AstFragmentKind::Variants,
                    after_derive,
                )
                .make_variants();
        }

        noop_flat_map_variant(variant, self)
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let expr = configure!(self, expr);
        expr.filter_map(|mut expr| {
            self.cfg.configure_expr_kind(&mut expr.kind);

            // Ignore derives so they remain unused.
            let (attr, after_derive) = self.classify_nonitem(&mut expr);

            if attr.is_some() {
                attr.as_ref().map(|a| self.cfg.maybe_emit_expr_attr_err(a));

                return self
                    .collect_attr(
                        attr,
                        vec![],
                        Annotatable::Expr(P(expr)),
                        AstFragmentKind::OptExpr,
                        after_derive,
                    )
                    .make_opt_expr()
                    .map(|expr| expr.into_inner());
            }

            if let ast::ExprKind::Mac(mac) = expr.kind {
                self.check_attributes(&expr.attrs);
                self.collect_bang(mac, expr.span, AstFragmentKind::OptExpr)
                    .make_opt_expr()
                    .map(|expr| expr.into_inner())
            } else {
                Some({
                    noop_visit_expr(&mut expr, self);
                    expr
                })
            }
        })
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        self.cfg.configure_pat(pat);
        match pat.kind {
            PatKind::Mac(_) => {}
            _ => return noop_visit_pat(pat, self),
        }

        visit_clobber(pat, |mut pat| match mem::replace(&mut pat.kind, PatKind::Wild) {
            PatKind::Mac(mac) => self.collect_bang(mac, pat.span, AstFragmentKind::Pat).make_pat(),
            _ => unreachable!(),
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
                return self
                    .collect_attr(
                        attr,
                        derives,
                        Annotatable::Stmt(P(stmt)),
                        AstFragmentKind::Stmts,
                        after_derive,
                    )
                    .make_stmts();
            }
        }

        if let StmtKind::Mac(mac) = stmt.kind {
            let (mac, style, attrs) = mac.into_inner();
            self.check_attributes(&attrs);
            let mut placeholder =
                self.collect_bang(mac, stmt.span, AstFragmentKind::Stmts).make_stmts();

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
        let ast::Stmt { id, kind, span } = stmt;
        noop_flat_map_stmt_kind(kind, self)
            .into_iter()
            .map(|kind| ast::Stmt { id, kind, span })
            .collect()
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
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::Item(item),
                    AstFragmentKind::Items,
                    after_derive,
                )
                .make_items();
        }

        match item.kind {
            ast::ItemKind::Mac(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.kind {
                    ItemKind::Mac(mac) => self
                        .collect(
                            AstFragmentKind::Items,
                            InvocationKind::Bang { mac, span: item.span },
                        )
                        .make_items(),
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(ast::Mod { inner, inline, .. })
                if item.ident != Ident::invalid() =>
            {
                let orig_directory_ownership = self.cx.current_expansion.directory_ownership;
                let mut module = (*self.cx.current_expansion.module).clone();
                module.mod_path.push(item.ident);

                if inline {
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
                        Some(_) => DirectoryOwnership::Owned { relative: Some(item.ident) },
                        None => DirectoryOwnership::UnownedViaMod,
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

    fn flat_map_trait_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let mut item = configure!(self, item);

        let (attr, traits, after_derive) = self.classify_item(&mut item);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::TraitItem(item),
                    AstFragmentKind::TraitItems,
                    after_derive,
                )
                .make_trait_items();
        }

        match item.kind {
            ast::AssocItemKind::Macro(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.kind {
                    ast::AssocItemKind::Macro(mac) => self
                        .collect_bang(mac, item.span, AstFragmentKind::TraitItems)
                        .make_trait_items(),
                    _ => unreachable!(),
                })
            }
            _ => noop_flat_map_assoc_item(item, self),
        }
    }

    fn flat_map_impl_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let mut item = configure!(self, item);

        let (attr, traits, after_derive) = self.classify_item(&mut item);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::ImplItem(item),
                    AstFragmentKind::ImplItems,
                    after_derive,
                )
                .make_impl_items();
        }

        match item.kind {
            ast::AssocItemKind::Macro(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.kind {
                    ast::AssocItemKind::Macro(mac) => self
                        .collect_bang(mac, item.span, AstFragmentKind::ImplItems)
                        .make_impl_items(),
                    _ => unreachable!(),
                })
            }
            _ => noop_flat_map_assoc_item(item, self),
        }
    }

    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        match ty.kind {
            ast::TyKind::Mac(_) => {}
            _ => return noop_visit_ty(ty, self),
        };

        visit_clobber(ty, |mut ty| match mem::replace(&mut ty.kind, ast::TyKind::Err) {
            ast::TyKind::Mac(mac) => self.collect_bang(mac, ty.span, AstFragmentKind::Ty).make_ty(),
            _ => unreachable!(),
        });
    }

    fn visit_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        self.cfg.configure_foreign_mod(foreign_mod);
        noop_visit_foreign_mod(foreign_mod, self);
    }

    fn flat_map_foreign_item(
        &mut self,
        mut foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        let (attr, traits, after_derive) = self.classify_item(&mut foreign_item);

        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::ForeignItem(foreign_item),
                    AstFragmentKind::ForeignItems,
                    after_derive,
                )
                .make_foreign_items();
        }

        match foreign_item.kind {
            ast::ForeignItemKind::Macro(..) => {
                self.check_attributes(&foreign_item.attrs);
                foreign_item.and_then(|item| match item.kind {
                    ast::ForeignItemKind::Macro(mac) => self
                        .collect_bang(mac, item.span, AstFragmentKind::ForeignItems)
                        .make_foreign_items(),
                    _ => unreachable!(),
                })
            }
            _ => noop_flat_map_foreign_item(foreign_item, self),
        }
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

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        let mut param = configure!(self, param);

        let (attr, traits, after_derive) = self.classify_item(&mut param);
        if attr.is_some() || !traits.is_empty() {
            return self
                .collect_attr(
                    attr,
                    traits,
                    Annotatable::GenericParam(param),
                    AstFragmentKind::GenericParams,
                    after_derive,
                )
                .make_generic_params();
        }

        noop_flat_map_generic_param(param, self)
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
                    items.push({
                        noop_visit_meta_list_item(&mut it, self);
                        it
                    });
                    continue;
                }

                if let Some(file) = it.value_str() {
                    let err_count = self.cx.parse_sess.span_diagnostic.err_count();
                    self.check_attributes(slice::from_ref(at));
                    if self.cx.parse_sess.span_diagnostic.err_count() > err_count {
                        // avoid loading the file if they haven't enabled the feature
                        return noop_visit_attribute(at, self);
                    }

                    let filename = match self.cx.resolve_path(&*file.as_str(), it.span()) {
                        Ok(filename) => filename,
                        Err(mut err) => {
                            err.emit();
                            continue;
                        }
                    };

                    match self.cx.source_map().load_file(&filename) {
                        Ok(source_file) => {
                            let src = source_file
                                .src
                                .as_ref()
                                .expect("freshly loaded file should have a source");
                            let src_interned = Symbol::intern(src.as_str());

                            let include_info = vec![
                                ast::NestedMetaItem::MetaItem(attr::mk_name_value_item_str(
                                    Ident::with_dummy_span(sym::file),
                                    file,
                                    DUMMY_SP,
                                )),
                                ast::NestedMetaItem::MetaItem(attr::mk_name_value_item_str(
                                    Ident::with_dummy_span(sym::contents),
                                    src_interned,
                                    DUMMY_SP,
                                )),
                            ];

                            let include_ident = Ident::with_dummy_span(sym::include);
                            let item = attr::mk_list_item(include_ident, include_info);
                            items.push(ast::NestedMetaItem::MetaItem(item));
                        }
                        Err(e) => {
                            let lit =
                                it.meta_item().and_then(|item| item.name_value_literal()).unwrap();

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
                            list[0].literal().map(|literal| &literal.kind)
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

            let meta = attr::mk_list_item(Ident::with_dummy_span(sym::doc), items);
            *at = ast::Attribute {
                kind: ast::AttrKind::Normal(AttrItem {
                    path: meta.path,
                    args: meta.kind.mac_args(meta.span),
                }),
                span: at.span,
                id: at.id,
                style: at.style,
            };
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

    fn proc_macro_hygiene(&self) -> bool {
        self.features.map_or(false, |features| features.proc_macro_hygiene)
    }
    fn custom_inner_attributes(&self) -> bool {
        self.features.map_or(false, |features| features.custom_inner_attributes)
    }
}
