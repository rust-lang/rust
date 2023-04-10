use crate::base::*;
use crate::config::StripUnconfigured;
use crate::errors::{
    IncompleteParse, RecursionLimitReached, RemoveExprNotSupported, RemoveNodeNotSupported,
    UnsupportedKeyValue, WrongFragmentKind,
};
use crate::hygiene::SyntaxContext;
use crate::mbe::diagnostics::annotate_err_with_kind;
use crate::module::{mod_dir_path, parse_external_mod, DirOwnership, ParsedExternalMod};
use crate::placeholders::{placeholder, PlaceholderExpander};

use rustc_ast as ast;
use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{AssocItemKind, AstNodeWrapper, AttrArgs, AttrStyle, AttrVec, ExprKind};
use rustc_ast::{ForeignItemKind, HasAttrs, HasNodeId};
use rustc_ast::{Inline, ItemKind, MacStmtStyle, MetaItemKind, ModKind};
use rustc_ast::{NestedMetaItem, NodeId, PatKind, StmtKind, TyKind};
use rustc_ast_pretty::pprust;
use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_data_structures::sync::Lrc;
use rustc_errors::PResult;
use rustc_feature::Features;
use rustc_parse::parser::{
    AttemptLocalParseRecovery, CommaRecoveryMode, ForceCollect, Parser, RecoverColon, RecoverComma,
};
use rustc_parse::validate_attr;
use rustc_session::lint::builtin::{UNUSED_ATTRIBUTES, UNUSED_DOC_COMMENTS};
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_session::parse::{feature_err, ParseSess};
use rustc_session::Limit;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{FileName, LocalExpnId, Span};

use smallvec::SmallVec;
use std::ops::Deref;
use std::path::PathBuf;
use std::rc::Rc;
use std::{iter, mem};

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
            MethodReceiverExpr(P<ast::Expr>),
            $($Kind($AstTy),)*
        }

        /// "Discriminant" of an AST fragment.
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub enum AstFragmentKind {
            OptExpr,
            MethodReceiverExpr,
            $($Kind,)*
        }

        impl AstFragmentKind {
            pub fn name(self) -> &'static str {
                match self {
                    AstFragmentKind::OptExpr => "expression",
                    AstFragmentKind::MethodReceiverExpr => "expression",
                    $(AstFragmentKind::$Kind => $kind_name,)*
                }
            }

            fn make_from<'a>(self, result: Box<dyn MacResult + 'a>) -> Option<AstFragment> {
                match self {
                    AstFragmentKind::OptExpr =>
                        result.make_expr().map(Some).map(AstFragment::OptExpr),
                    AstFragmentKind::MethodReceiverExpr =>
                        result.make_expr().map(AstFragment::MethodReceiverExpr),
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
                        ${ignore(flat_map_ast_elt)}
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

            pub fn make_method_receiver_expr(self) -> P<ast::Expr> {
                match self {
                    AstFragment::MethodReceiverExpr(expr) => expr,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            }

            $(pub fn $make_ast(self) -> $AstTy {
                match self {
                    AstFragment::$Kind(ast) => ast,
                    _ => panic!("AstFragment::make_* called on the wrong kind of fragment"),
                }
            })*

            fn make_ast<T: InvocationCollectorNode>(self) -> T::OutputTy {
                T::fragment_to_output(self)
            }

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
                    AstFragment::MethodReceiverExpr(expr) => vis.visit_method_receiver_expr(expr),
                    $($(AstFragment::$Kind(ast) => vis.$mut_visit_ast(ast),)?)*
                    $($(AstFragment::$Kind(ast) =>
                        ast.flat_map_in_place(|ast| vis.$flat_map_ast_elt(ast)),)?)*
                }
            }

            pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) {
                match self {
                    AstFragment::OptExpr(Some(expr)) => visitor.visit_expr(expr),
                    AstFragment::OptExpr(None) => {}
                    AstFragment::MethodReceiverExpr(expr) => visitor.visit_method_receiver_expr(expr),
                    $($(AstFragment::$Kind(ast) => visitor.$visit_ast(ast),)?)*
                    $($(AstFragment::$Kind(ast) => for ast_elt in &ast[..] {
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
    ExprFields(SmallVec<[ast::ExprField; 1]>) {
        "field expression"; many fn flat_map_expr_field; fn visit_expr_field(); fn make_expr_fields;
    }
    PatFields(SmallVec<[ast::PatField; 1]>) {
        "field pattern";
        many fn flat_map_pat_field;
        fn visit_pat_field();
        fn make_pat_fields;
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
    FieldDefs(SmallVec<[ast::FieldDef; 1]>) {
        "field";
        many fn flat_map_field_def;
        fn visit_field_def();
        fn make_field_defs;
    }
    Variants(SmallVec<[ast::Variant; 1]>) {
        "variant"; many fn flat_map_variant; fn visit_variant(); fn make_variants;
    }
    Crate(ast::Crate) { "crate"; one fn visit_crate; fn visit_crate; fn make_crate; }
}

pub enum SupportsMacroExpansion {
    No,
    Yes { supports_inner_attrs: bool },
}

impl AstFragmentKind {
    pub(crate) fn dummy(self, span: Span) -> AstFragment {
        self.make_from(DummyResult::any(span)).expect("couldn't create a dummy AST fragment")
    }

    pub fn supports_macro_expansion(self) -> SupportsMacroExpansion {
        match self {
            AstFragmentKind::OptExpr
            | AstFragmentKind::Expr
            | AstFragmentKind::MethodReceiverExpr
            | AstFragmentKind::Stmts
            | AstFragmentKind::Ty
            | AstFragmentKind::Pat => SupportsMacroExpansion::Yes { supports_inner_attrs: false },
            AstFragmentKind::Items
            | AstFragmentKind::TraitItems
            | AstFragmentKind::ImplItems
            | AstFragmentKind::ForeignItems
            | AstFragmentKind::Crate => SupportsMacroExpansion::Yes { supports_inner_attrs: true },
            AstFragmentKind::Arms
            | AstFragmentKind::ExprFields
            | AstFragmentKind::PatFields
            | AstFragmentKind::GenericParams
            | AstFragmentKind::Params
            | AstFragmentKind::FieldDefs
            | AstFragmentKind::Variants => SupportsMacroExpansion::No,
        }
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
            AstFragmentKind::ExprFields => {
                AstFragment::ExprFields(items.map(Annotatable::expect_expr_field).collect())
            }
            AstFragmentKind::PatFields => {
                AstFragment::PatFields(items.map(Annotatable::expect_pat_field).collect())
            }
            AstFragmentKind::GenericParams => {
                AstFragment::GenericParams(items.map(Annotatable::expect_generic_param).collect())
            }
            AstFragmentKind::Params => {
                AstFragment::Params(items.map(Annotatable::expect_param).collect())
            }
            AstFragmentKind::FieldDefs => {
                AstFragment::FieldDefs(items.map(Annotatable::expect_field_def).collect())
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
            AstFragmentKind::MethodReceiverExpr => AstFragment::MethodReceiverExpr(
                items.next().expect("expected exactly one expression").expect_expr(),
            ),
            AstFragmentKind::OptExpr => {
                AstFragment::OptExpr(items.next().map(Annotatable::expect_expr))
            }
            AstFragmentKind::Crate => {
                AstFragment::Crate(items.next().expect("expected exactly one crate").expect_crate())
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
        mac: P<ast::MacCall>,
        span: Span,
    },
    Attr {
        attr: ast::Attribute,
        // Re-insertion position for inert attributes.
        pos: usize,
        item: Annotatable,
        // Required for resolving derive helper attributes.
        derives: Vec<ast::Path>,
    },
    Derive {
        path: ast::Path,
        is_const: bool,
        item: Annotatable,
    },
}

impl InvocationKind {
    fn placeholder_visibility(&self) -> Option<ast::Visibility> {
        // HACK: For unnamed fields placeholders should have the same visibility as the actual
        // fields because for tuple structs/variants resolve determines visibilities of their
        // constructor using these field visibilities before attributes on them are expanded.
        // The assumption is that the attribute expansion cannot change field visibilities,
        // and it holds because only inert attributes are supported in this position.
        match self {
            InvocationKind::Attr { item: Annotatable::FieldDef(field), .. }
            | InvocationKind::Derive { item: Annotatable::FieldDef(field), .. }
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

    pub fn expand_crate(&mut self, krate: ast::Crate) -> ast::Crate {
        let file_path = match self.cx.source_map().span_to_filename(krate.spans.inner_span) {
            FileName::Real(name) => name
                .into_local_path()
                .expect("attempting to resolve a file path in an external file"),
            other => PathBuf::from(other.prefer_local().to_string()),
        };
        let dir_path = file_path.parent().unwrap_or(&file_path).to_owned();
        self.cx.root_path = dir_path.clone();
        self.cx.current_expansion.module = Rc::new(ModuleData {
            mod_path: vec![Ident::from_str(&self.cx.ecfg.crate_name)],
            file_path_stack: vec![file_path],
            dir_path,
        });
        let krate = self.fully_expand_fragment(AstFragment::Crate(krate)).make_crate();
        assert_eq!(krate.id, ast::CRATE_NODE_ID);
        self.cx.trace_macros_diag();
        krate
    }

    /// Recursively expand all macro invocations in this AST fragment.
    pub fn fully_expand_fragment(&mut self, input_fragment: AstFragment) -> AstFragment {
        let orig_expansion_data = self.cx.current_expansion.clone();
        let orig_force_mode = self.cx.force_mode;

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
            let Some((invoc, ext)) = invocations.pop() else {
                self.resolve_imports();
                if undetermined_invocations.is_empty() {
                    break;
                }
                invocations = mem::take(&mut undetermined_invocations);
                force = !mem::replace(&mut progress, false);
                if force && self.monotonic {
                    self.cx.sess.delay_span_bug(
                        invocations.last().unwrap().0.span(),
                        "expansion entered force mode without producing any errors",
                    );
                }
                continue;
            };

            let ext = match ext {
                Some(ext) => ext,
                None => {
                    let eager_expansion_root = if self.monotonic {
                        invoc.expansion_data.id
                    } else {
                        orig_expansion_data.id
                    };
                    match self.cx.resolver.resolve_macro_invocation(
                        &invoc,
                        eager_expansion_root,
                        force,
                    ) {
                        Ok(ext) => ext,
                        Err(Indeterminate) => {
                            // Cannot resolve, will retry this invocation later.
                            undetermined_invocations.push((invoc, None));
                            continue;
                        }
                    }
                }
            };

            let ExpansionData { depth, id: expn_id, .. } = invoc.expansion_data;
            let depth = depth - orig_expansion_data.depth;
            self.cx.current_expansion = invoc.expansion_data.clone();
            self.cx.force_mode = force;

            let fragment_kind = invoc.fragment_kind;
            let (expanded_fragment, new_invocations) = match self.expand_invoc(invoc, &ext.kind) {
                ExpandResult::Ready(fragment) => {
                    let mut derive_invocations = Vec::new();
                    let derive_placeholders = self
                        .cx
                        .resolver
                        .take_derive_resolutions(expn_id)
                        .map(|derives| {
                            derive_invocations.reserve(derives.len());
                            derives
                                .into_iter()
                                .map(|(path, item, _exts, is_const)| {
                                    // FIXME: Consider using the derive resolutions (`_exts`)
                                    // instead of enqueuing the derives to be resolved again later.
                                    let expn_id = LocalExpnId::fresh_empty();
                                    derive_invocations.push((
                                        Invocation {
                                            kind: InvocationKind::Derive { path, item, is_const },
                                            fragment_kind,
                                            expansion_data: ExpansionData {
                                                id: expn_id,
                                                ..self.cx.current_expansion.clone()
                                            },
                                        },
                                        None,
                                    ));
                                    NodeId::placeholder_from_expn_id(expn_id)
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();

                    let (fragment, collected_invocations) =
                        self.collect_invocations(fragment, &derive_placeholders);
                    // We choose to expand any derive invocations associated with this macro invocation
                    // *before* any macro invocations collected from the output fragment
                    derive_invocations.extend(collected_invocations);
                    (fragment, derive_invocations)
                }
                ExpandResult::Retry(invoc) => {
                    if force {
                        self.cx.span_bug(
                            invoc.span(),
                            "expansion entered force mode but is still stuck",
                        );
                    } else {
                        // Cannot expand, will retry this invocation later.
                        undetermined_invocations.push((invoc, Some(ext)));
                        continue;
                    }
                }
            };

            progress = true;
            if expanded_fragments.len() < depth {
                expanded_fragments.push(Vec::new());
            }
            expanded_fragments[depth - 1].push((expn_id, expanded_fragment));
            invocations.extend(new_invocations.into_iter().rev());
        }

        self.cx.current_expansion = orig_expansion_data;
        self.cx.force_mode = orig_force_mode;

        // Finally incorporate all the expanded macros into the input AST fragment.
        let mut placeholder_expander = PlaceholderExpander::default();
        while let Some(expanded_fragments) = expanded_fragments.pop() {
            for (expn_id, expanded_fragment) in expanded_fragments.into_iter().rev() {
                placeholder_expander
                    .add(NodeId::placeholder_from_expn_id(expn_id), expanded_fragment);
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
    fn collect_invocations(
        &mut self,
        mut fragment: AstFragment,
        extra_placeholders: &[NodeId],
    ) -> (AstFragment, Vec<(Invocation, Option<Lrc<SyntaxExtension>>)>) {
        // Resolve `$crate`s in the fragment for pretty-printing.
        self.cx.resolver.resolve_dollar_crates();

        let mut invocations = {
            let mut collector = InvocationCollector {
                // Non-derive macro invocations cannot see the results of cfg expansion - they
                // will either be removed along with the item, or invoked before the cfg/cfg_attr
                // attribute is expanded. Therefore, we don't need to configure the tokens
                // Derive macros *can* see the results of cfg-expansion - they are handled
                // specially in `fully_expand_fragment`
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

            if self.cx.sess.opts.incremental_relative_spans() {
                for (invoc, _) in invocations.iter_mut() {
                    let expn_id = invoc.expansion_data.id;
                    let parent_def = self.cx.resolver.invocation_parent(expn_id);
                    let span = match &mut invoc.kind {
                        InvocationKind::Bang { span, .. } => span,
                        InvocationKind::Attr { attr, .. } => &mut attr.span,
                        InvocationKind::Derive { path, .. } => &mut path.span,
                    };
                    *span = span.with_parent(Some(parent_def));
                }
            }
        }

        (fragment, invocations)
    }

    fn error_recursion_limit_reached(&mut self) {
        let expn_data = self.cx.current_expansion.id.expn_data();
        let suggested_limit = match self.cx.ecfg.recursion_limit {
            Limit(0) => Limit(2),
            limit => limit * 2,
        };

        self.cx.emit_err(RecursionLimitReached {
            span: expn_data.call_site,
            descr: expn_data.kind.descr(),
            suggested_limit,
            crate_name: &self.cx.ecfg.crate_name,
        });

        self.cx.trace_macros_diag();
    }

    /// A macro's expansion does not fit in this fragment kind.
    /// For example, a non-type macro in a type position.
    fn error_wrong_fragment_kind(&mut self, kind: AstFragmentKind, mac: &ast::MacCall, span: Span) {
        self.cx.emit_err(WrongFragmentKind { span, kind: kind.name(), name: &mac.path });

        self.cx.trace_macros_diag();
    }

    fn expand_invoc(
        &mut self,
        invoc: Invocation,
        ext: &SyntaxExtensionKind,
    ) -> ExpandResult<AstFragment, Invocation> {
        let recursion_limit =
            self.cx.reduced_recursion_limit.unwrap_or(self.cx.ecfg.recursion_limit);
        if !recursion_limit.value_within_limit(self.cx.current_expansion.depth) {
            if self.cx.reduced_recursion_limit.is_none() {
                self.error_recursion_limit_reached();
            }

            // Reduce the recursion limit by half each time it triggers.
            self.cx.reduced_recursion_limit = Some(recursion_limit / 2);

            return ExpandResult::Ready(invoc.fragment_kind.dummy(invoc.span()));
        }

        let (fragment_kind, span) = (invoc.fragment_kind, invoc.span());
        ExpandResult::Ready(match invoc.kind {
            InvocationKind::Bang { mac, .. } => match ext {
                SyntaxExtensionKind::Bang(expander) => {
                    let Ok(tok_result) = expander.expand(self.cx, span, mac.args.tokens.clone()) else {
                        return ExpandResult::Ready(fragment_kind.dummy(span));
                    };
                    self.parse_ast_fragment(tok_result, fragment_kind, &mac.path, span)
                }
                SyntaxExtensionKind::LegacyBang(expander) => {
                    let tok_result = expander.expand(self.cx, span, mac.args.tokens.clone());
                    let result = if let Some(result) = fragment_kind.make_from(tok_result) {
                        result
                    } else {
                        self.error_wrong_fragment_kind(fragment_kind, &mac, span);
                        fragment_kind.dummy(span)
                    };
                    result
                }
                _ => unreachable!(),
            },
            InvocationKind::Attr { attr, pos, mut item, derives } => match ext {
                SyntaxExtensionKind::Attr(expander) => {
                    self.gate_proc_macro_input(&item);
                    self.gate_proc_macro_attr_item(span, &item);
                    let tokens = match &item {
                        // FIXME: Collect tokens and use them instead of generating
                        // fake ones. These are unstable, so it needs to be
                        // fixed prior to stabilization
                        // Fake tokens when we are invoking an inner attribute, and
                        // we are invoking it on an out-of-line module or crate.
                        Annotatable::Crate(krate) => rustc_parse::fake_token_stream_for_crate(
                            &self.cx.sess.parse_sess,
                            krate,
                        ),
                        Annotatable::Item(item_inner)
                            if matches!(attr.style, AttrStyle::Inner)
                                && matches!(
                                    item_inner.kind,
                                    ItemKind::Mod(
                                        _,
                                        ModKind::Unloaded | ModKind::Loaded(_, Inline::No, _),
                                    )
                                ) =>
                        {
                            rustc_parse::fake_token_stream_for_item(
                                &self.cx.sess.parse_sess,
                                item_inner,
                            )
                        }
                        _ => item.to_tokens(),
                    };
                    let attr_item = attr.unwrap_normal_item();
                    if let AttrArgs::Eq(..) = attr_item.args {
                        self.cx.emit_err(UnsupportedKeyValue { span });
                    }
                    let inner_tokens = attr_item.args.inner_tokens();
                    let Ok(tok_result) = expander.expand(self.cx, span, inner_tokens, tokens) else {
                        return ExpandResult::Ready(fragment_kind.dummy(span));
                    };
                    self.parse_ast_fragment(tok_result, fragment_kind, &attr_item.path, span)
                }
                SyntaxExtensionKind::LegacyAttr(expander) => {
                    match validate_attr::parse_meta(&self.cx.sess.parse_sess, &attr) {
                        Ok(meta) => {
                            let items = match expander.expand(self.cx, span, &meta, item, false) {
                                ExpandResult::Ready(items) => items,
                                ExpandResult::Retry(item) => {
                                    // Reassemble the original invocation for retrying.
                                    return ExpandResult::Retry(Invocation {
                                        kind: InvocationKind::Attr { attr, pos, item, derives },
                                        ..invoc
                                    });
                                }
                            };
                            if matches!(
                                fragment_kind,
                                AstFragmentKind::Expr | AstFragmentKind::MethodReceiverExpr
                            ) && items.is_empty()
                            {
                                self.cx.emit_err(RemoveExprNotSupported { span });
                                fragment_kind.dummy(span)
                            } else {
                                fragment_kind.expect_from_annotatables(items)
                            }
                        }
                        Err(mut err) => {
                            err.emit();
                            fragment_kind.dummy(span)
                        }
                    }
                }
                SyntaxExtensionKind::NonMacroAttr => {
                    self.cx.expanded_inert_attrs.mark(&attr);
                    item.visit_attrs(|attrs| attrs.insert(pos, attr));
                    fragment_kind.expect_from_annotatables(iter::once(item))
                }
                _ => unreachable!(),
            },
            InvocationKind::Derive { path, item, is_const } => match ext {
                SyntaxExtensionKind::Derive(expander)
                | SyntaxExtensionKind::LegacyDerive(expander) => {
                    if let SyntaxExtensionKind::Derive(..) = ext {
                        self.gate_proc_macro_input(&item);
                    }
                    let meta = ast::MetaItem { kind: MetaItemKind::Word, span, path };
                    let items = match expander.expand(self.cx, span, &meta, item, is_const) {
                        ExpandResult::Ready(items) => items,
                        ExpandResult::Retry(item) => {
                            // Reassemble the original invocation for retrying.
                            return ExpandResult::Retry(Invocation {
                                kind: InvocationKind::Derive { path: meta.path, item, is_const },
                                ..invoc
                            });
                        }
                    };
                    fragment_kind.expect_from_annotatables(items)
                }
                _ => unreachable!(),
            },
        })
    }

    fn gate_proc_macro_attr_item(&self, span: Span, item: &Annotatable) {
        let kind = match item {
            Annotatable::Item(_)
            | Annotatable::TraitItem(_)
            | Annotatable::ImplItem(_)
            | Annotatable::ForeignItem(_)
            | Annotatable::Crate(..) => return,
            Annotatable::Stmt(stmt) => {
                // Attributes are stable on item statements,
                // but unstable on all other kinds of statements
                if stmt.is_item() {
                    return;
                }
                "statements"
            }
            Annotatable::Expr(_) => "expressions",
            Annotatable::Arm(..)
            | Annotatable::ExprField(..)
            | Annotatable::PatField(..)
            | Annotatable::GenericParam(..)
            | Annotatable::Param(..)
            | Annotatable::FieldDef(..)
            | Annotatable::Variant(..) => panic!("unexpected annotatable"),
        };
        if self.cx.ecfg.proc_macro_hygiene() {
            return;
        }
        feature_err(
            &self.cx.sess.parse_sess,
            sym::proc_macro_hygiene,
            span,
            format!("custom attributes cannot be applied to {}", kind),
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
                    ItemKind::Mod(_, mod_kind)
                        if !matches!(mod_kind, ModKind::Loaded(_, Inline::Yes, _)) =>
                    {
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
        }

        if !self.cx.ecfg.proc_macro_hygiene() {
            annotatable
                .visit_with(&mut GateProcMacroInput { parse_sess: &self.cx.sess.parse_sess });
        }
    }

    fn parse_ast_fragment(
        &mut self,
        toks: TokenStream,
        kind: AstFragmentKind,
        path: &ast::Path,
        span: Span,
    ) -> AstFragment {
        let mut parser = self.cx.new_parser_from_tts(toks);
        match parse_ast_fragment(&mut parser, kind) {
            Ok(fragment) => {
                ensure_complete_parse(&mut parser, path, kind.name(), span);
                fragment
            }
            Err(mut err) => {
                if err.span.is_dummy() {
                    err.set_span(span);
                }
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
            while let Some(item) = this.parse_item(ForceCollect::No)? {
                items.push(item);
            }
            AstFragment::Items(items)
        }
        AstFragmentKind::TraitItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_trait_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::TraitItems(items)
        }
        AstFragmentKind::ImplItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_impl_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::ImplItems(items)
        }
        AstFragmentKind::ForeignItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_foreign_item(ForceCollect::No)? {
                items.extend(item);
            }
            AstFragment::ForeignItems(items)
        }
        AstFragmentKind::Stmts => {
            let mut stmts = SmallVec::new();
            // Won't make progress on a `}`.
            while this.token != token::Eof && this.token != token::CloseDelim(Delimiter::Brace) {
                if let Some(stmt) = this.parse_full_stmt(AttemptLocalParseRecovery::Yes)? {
                    stmts.push(stmt);
                }
            }
            AstFragment::Stmts(stmts)
        }
        AstFragmentKind::Expr => AstFragment::Expr(this.parse_expr()?),
        AstFragmentKind::MethodReceiverExpr => AstFragment::MethodReceiverExpr(this.parse_expr()?),
        AstFragmentKind::OptExpr => {
            if this.token != token::Eof {
                AstFragment::OptExpr(Some(this.parse_expr()?))
            } else {
                AstFragment::OptExpr(None)
            }
        }
        AstFragmentKind::Ty => AstFragment::Ty(this.parse_ty()?),
        AstFragmentKind::Pat => AstFragment::Pat(this.parse_pat_allow_top_alt(
            None,
            RecoverComma::No,
            RecoverColon::Yes,
            CommaRecoveryMode::LikelyTuple,
        )?),
        AstFragmentKind::Crate => AstFragment::Crate(this.parse_crate_mod()?),
        AstFragmentKind::Arms
        | AstFragmentKind::ExprFields
        | AstFragmentKind::PatFields
        | AstFragmentKind::GenericParams
        | AstFragmentKind::Params
        | AstFragmentKind::FieldDefs
        | AstFragmentKind::Variants => panic!("unexpected AST fragment kind"),
    })
}

pub fn ensure_complete_parse<'a>(
    parser: &mut Parser<'a>,
    macro_path: &ast::Path,
    kind_name: &str,
    span: Span,
) {
    if parser.token != token::Eof {
        let token = pprust::token_to_string(&parser.token);
        // Avoid emitting backtrace info twice.
        let def_site_span = parser.token.span.with_ctxt(SyntaxContext::root());

        let semi_span = parser.sess.source_map().next_point(span);
        let add_semicolon = match &parser.sess.source_map().span_to_snippet(semi_span) {
            Ok(snippet) if &snippet[..] != ";" && kind_name == "expression" => {
                Some(span.shrink_to_hi())
            }
            _ => None,
        };

        parser.sess.emit_err(IncompleteParse {
            span: def_site_span,
            token,
            label_span: span,
            macro_path,
            kind_name,
            add_semicolon,
        });
    }
}

/// Wraps a call to `noop_visit_*` / `noop_flat_map_*`
/// for an AST node that supports attributes
/// (see the `Annotatable` enum)
/// This method assigns a `NodeId`, and sets that `NodeId`
/// as our current 'lint node id'. If a macro call is found
/// inside this AST node, we will use this AST node's `NodeId`
/// to emit lints associated with that macro (allowing
/// `#[allow]` / `#[deny]` to be applied close to
/// the macro invocation).
///
/// Do *not* call this for a macro AST node
/// (e.g. `ExprKind::MacCall`) - we cannot emit lints
/// at these AST nodes, since they are removed and
/// replaced with the result of macro expansion.
///
/// All other `NodeId`s are assigned by `visit_id`.
/// * `self` is the 'self' parameter for the current method,
/// * `id` is a mutable reference to the `NodeId` field
///    of the current AST node.
/// * `closure` is a closure that executes the
///   `noop_visit_*` / `noop_flat_map_*` method
///   for the current AST node.
macro_rules! assign_id {
    ($self:ident, $id:expr, $closure:expr) => {{
        let old_id = $self.cx.current_expansion.lint_node_id;
        if $self.monotonic {
            debug_assert_eq!(*$id, ast::DUMMY_NODE_ID);
            let new_id = $self.cx.resolver.next_node_id();
            *$id = new_id;
            $self.cx.current_expansion.lint_node_id = new_id;
        }
        let ret = ($closure)();
        $self.cx.current_expansion.lint_node_id = old_id;
        ret
    }};
}

enum AddSemicolon {
    Yes,
    No,
}

/// A trait implemented for all `AstFragment` nodes and providing all pieces
/// of functionality used by `InvocationCollector`.
trait InvocationCollectorNode: HasAttrs + HasNodeId + Sized {
    type OutputTy = SmallVec<[Self; 1]>;
    type AttrsTy: Deref<Target = [ast::Attribute]> = ast::AttrVec;
    const KIND: AstFragmentKind;
    fn to_annotatable(self) -> Annotatable;
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy;
    fn descr() -> &'static str {
        unreachable!()
    }
    fn noop_flat_map<V: MutVisitor>(self, _visitor: &mut V) -> Self::OutputTy {
        unreachable!()
    }
    fn noop_visit<V: MutVisitor>(&mut self, _visitor: &mut V) {
        unreachable!()
    }
    fn is_mac_call(&self) -> bool {
        false
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        unreachable!()
    }
    fn pre_flat_map_node_collect_attr(_cfg: &StripUnconfigured<'_>, _attr: &ast::Attribute) {}
    fn post_flat_map_node_collect_bang(_output: &mut Self::OutputTy, _add_semicolon: AddSemicolon) {
    }
    fn wrap_flat_map_node_noop_flat_map(
        node: Self,
        collector: &mut InvocationCollector<'_, '_>,
        noop_flat_map: impl FnOnce(Self, &mut InvocationCollector<'_, '_>) -> Self::OutputTy,
    ) -> Result<Self::OutputTy, Self> {
        Ok(noop_flat_map(node, collector))
    }
    fn expand_cfg_false(
        &mut self,
        collector: &mut InvocationCollector<'_, '_>,
        _pos: usize,
        span: Span,
    ) {
        collector.cx.emit_err(RemoveNodeNotSupported { span, descr: Self::descr() });
    }

    /// All of the names (items) declared by this node.
    /// This is an approximation and should only be used for diagnostics.
    fn declared_names(&self) -> Vec<Ident> {
        vec![]
    }
}

impl InvocationCollectorNode for P<ast::Item> {
    const KIND: AstFragmentKind = AstFragmentKind::Items;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Item(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_items()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_item(self, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.into_inner();
        match node.kind {
            ItemKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn wrap_flat_map_node_noop_flat_map(
        mut node: Self,
        collector: &mut InvocationCollector<'_, '_>,
        noop_flat_map: impl FnOnce(Self, &mut InvocationCollector<'_, '_>) -> Self::OutputTy,
    ) -> Result<Self::OutputTy, Self> {
        if !matches!(node.kind, ItemKind::Mod(..)) {
            return Ok(noop_flat_map(node, collector));
        }

        // Work around borrow checker not seeing through `P`'s deref.
        let (ident, span, mut attrs) = (node.ident, node.span, mem::take(&mut node.attrs));
        let ItemKind::Mod(_, mod_kind) = &mut node.kind else {
            unreachable!()
        };

        let ecx = &mut collector.cx;
        let (file_path, dir_path, dir_ownership) = match mod_kind {
            ModKind::Loaded(_, inline, _) => {
                // Inline `mod foo { ... }`, but we still need to push directories.
                let (dir_path, dir_ownership) = mod_dir_path(
                    &ecx.sess,
                    ident,
                    &attrs,
                    &ecx.current_expansion.module,
                    ecx.current_expansion.dir_ownership,
                    *inline,
                );
                node.attrs = attrs;
                (None, dir_path, dir_ownership)
            }
            ModKind::Unloaded => {
                // We have an outline `mod foo;` so we need to parse the file.
                let old_attrs_len = attrs.len();
                let ParsedExternalMod { items, spans, file_path, dir_path, dir_ownership } =
                    parse_external_mod(
                        &ecx.sess,
                        ident,
                        span,
                        &ecx.current_expansion.module,
                        ecx.current_expansion.dir_ownership,
                        &mut attrs,
                    );

                if let Some(lint_store) = ecx.lint_store {
                    lint_store.pre_expansion_lint(
                        ecx.sess,
                        ecx.resolver.registered_tools(),
                        ecx.current_expansion.lint_node_id,
                        &attrs,
                        &items,
                        ident.name,
                    );
                }

                *mod_kind = ModKind::Loaded(items, Inline::No, spans);
                node.attrs = attrs;
                if node.attrs.len() > old_attrs_len {
                    // If we loaded an out-of-line module and added some inner attributes,
                    // then we need to re-configure it and re-collect attributes for
                    // resolution and expansion.
                    return Err(node);
                }
                (Some(file_path), dir_path, dir_ownership)
            }
        };

        // Set the module info before we flat map.
        let mut module = ecx.current_expansion.module.with_dir_path(dir_path);
        module.mod_path.push(ident);
        if let Some(file_path) = file_path {
            module.file_path_stack.push(file_path);
        }

        let orig_module = mem::replace(&mut ecx.current_expansion.module, Rc::new(module));
        let orig_dir_ownership =
            mem::replace(&mut ecx.current_expansion.dir_ownership, dir_ownership);

        let res = Ok(noop_flat_map(node, collector));

        collector.cx.current_expansion.dir_ownership = orig_dir_ownership;
        collector.cx.current_expansion.module = orig_module;
        res
    }
    fn declared_names(&self) -> Vec<Ident> {
        if let ItemKind::Use(ut) = &self.kind {
            fn collect_use_tree_leaves(ut: &ast::UseTree, idents: &mut Vec<Ident>) {
                match &ut.kind {
                    ast::UseTreeKind::Glob => {}
                    ast::UseTreeKind::Simple(_) => idents.push(ut.ident()),
                    ast::UseTreeKind::Nested(nested) => {
                        for (ut, _) in nested {
                            collect_use_tree_leaves(&ut, idents);
                        }
                    }
                }
            }

            let mut idents = Vec::new();
            collect_use_tree_leaves(&ut, &mut idents);
            return idents;
        }

        vec![self.ident]
    }
}

struct TraitItemTag;
impl InvocationCollectorNode for AstNodeWrapper<P<ast::AssocItem>, TraitItemTag> {
    type OutputTy = SmallVec<[P<ast::AssocItem>; 1]>;
    const KIND: AstFragmentKind = AstFragmentKind::TraitItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::TraitItem(self.wrapped)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_trait_items()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_assoc_item(self.wrapped, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, AssocItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let item = self.wrapped.into_inner();
        match item.kind {
            AssocItemKind::MacCall(mac) => (mac, item.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

struct ImplItemTag;
impl InvocationCollectorNode for AstNodeWrapper<P<ast::AssocItem>, ImplItemTag> {
    type OutputTy = SmallVec<[P<ast::AssocItem>; 1]>;
    const KIND: AstFragmentKind = AstFragmentKind::ImplItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::ImplItem(self.wrapped)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_impl_items()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_assoc_item(self.wrapped, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, AssocItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let item = self.wrapped.into_inner();
        match item.kind {
            AssocItemKind::MacCall(mac) => (mac, item.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for P<ast::ForeignItem> {
    const KIND: AstFragmentKind = AstFragmentKind::ForeignItems;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::ForeignItem(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_foreign_items()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_foreign_item(self, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ForeignItemKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.into_inner();
        match node.kind {
            ForeignItemKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for ast::Variant {
    const KIND: AstFragmentKind = AstFragmentKind::Variants;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Variant(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_variants()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_variant(self, visitor)
    }
}

impl InvocationCollectorNode for ast::FieldDef {
    const KIND: AstFragmentKind = AstFragmentKind::FieldDefs;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::FieldDef(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_field_defs()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_field_def(self, visitor)
    }
}

impl InvocationCollectorNode for ast::PatField {
    const KIND: AstFragmentKind = AstFragmentKind::PatFields;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::PatField(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_pat_fields()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_pat_field(self, visitor)
    }
}

impl InvocationCollectorNode for ast::ExprField {
    const KIND: AstFragmentKind = AstFragmentKind::ExprFields;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::ExprField(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_expr_fields()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_expr_field(self, visitor)
    }
}

impl InvocationCollectorNode for ast::Param {
    const KIND: AstFragmentKind = AstFragmentKind::Params;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Param(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_params()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_param(self, visitor)
    }
}

impl InvocationCollectorNode for ast::GenericParam {
    const KIND: AstFragmentKind = AstFragmentKind::GenericParams;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::GenericParam(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_generic_params()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_generic_param(self, visitor)
    }
}

impl InvocationCollectorNode for ast::Arm {
    const KIND: AstFragmentKind = AstFragmentKind::Arms;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Arm(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_arms()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_arm(self, visitor)
    }
}

impl InvocationCollectorNode for ast::Stmt {
    type AttrsTy = ast::AttrVec;
    const KIND: AstFragmentKind = AstFragmentKind::Stmts;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Stmt(P(self))
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_stmts()
    }
    fn noop_flat_map<V: MutVisitor>(self, visitor: &mut V) -> Self::OutputTy {
        noop_flat_map_stmt(self, visitor)
    }
    fn is_mac_call(&self) -> bool {
        match &self.kind {
            StmtKind::MacCall(..) => true,
            StmtKind::Item(item) => matches!(item.kind, ItemKind::MacCall(..)),
            StmtKind::Semi(expr) => matches!(expr.kind, ExprKind::MacCall(..)),
            StmtKind::Expr(..) => unreachable!(),
            StmtKind::Local(..) | StmtKind::Empty => false,
        }
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        // We pull macro invocations (both attributes and fn-like macro calls) out of their
        // `StmtKind`s and treat them as statement macro invocations, not as items or expressions.
        let (add_semicolon, mac, attrs) = match self.kind {
            StmtKind::MacCall(mac) => {
                let ast::MacCallStmt { mac, style, attrs, .. } = mac.into_inner();
                (style == MacStmtStyle::Semicolon, mac, attrs)
            }
            StmtKind::Item(item) => match item.into_inner() {
                ast::Item { kind: ItemKind::MacCall(mac), attrs, .. } => {
                    (mac.args.need_semicolon(), mac, attrs)
                }
                _ => unreachable!(),
            },
            StmtKind::Semi(expr) => match expr.into_inner() {
                ast::Expr { kind: ExprKind::MacCall(mac), attrs, .. } => {
                    (mac.args.need_semicolon(), mac, attrs)
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        (mac, attrs, if add_semicolon { AddSemicolon::Yes } else { AddSemicolon::No })
    }
    fn post_flat_map_node_collect_bang(stmts: &mut Self::OutputTy, add_semicolon: AddSemicolon) {
        // If this is a macro invocation with a semicolon, then apply that
        // semicolon to the final statement produced by expansion.
        if matches!(add_semicolon, AddSemicolon::Yes) {
            if let Some(stmt) = stmts.pop() {
                stmts.push(stmt.add_trailing_semicolon());
            }
        }
    }
}

impl InvocationCollectorNode for ast::Crate {
    type OutputTy = ast::Crate;
    const KIND: AstFragmentKind = AstFragmentKind::Crate;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Crate(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_crate()
    }
    fn noop_visit<V: MutVisitor>(&mut self, visitor: &mut V) {
        noop_visit_crate(self, visitor)
    }
    fn expand_cfg_false(
        &mut self,
        collector: &mut InvocationCollector<'_, '_>,
        pos: usize,
        _span: Span,
    ) {
        // Attributes above `cfg(FALSE)` are left in place, because we may want to configure
        // some global crate properties even on fully unconfigured crates.
        self.attrs.truncate(pos);
        // Standard prelude imports are left in the crate for backward compatibility.
        self.items.truncate(collector.cx.num_standard_library_imports);
    }
}

impl InvocationCollectorNode for P<ast::Ty> {
    type OutputTy = P<ast::Ty>;
    const KIND: AstFragmentKind = AstFragmentKind::Ty;
    fn to_annotatable(self) -> Annotatable {
        unreachable!()
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_ty()
    }
    fn noop_visit<V: MutVisitor>(&mut self, visitor: &mut V) {
        noop_visit_ty(self, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ast::TyKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.into_inner();
        match node.kind {
            TyKind::MacCall(mac) => (mac, AttrVec::new(), AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for P<ast::Pat> {
    type OutputTy = P<ast::Pat>;
    const KIND: AstFragmentKind = AstFragmentKind::Pat;
    fn to_annotatable(self) -> Annotatable {
        unreachable!()
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_pat()
    }
    fn noop_visit<V: MutVisitor>(&mut self, visitor: &mut V) {
        noop_visit_pat(self, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, PatKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.into_inner();
        match node.kind {
            PatKind::MacCall(mac) => (mac, AttrVec::new(), AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

impl InvocationCollectorNode for P<ast::Expr> {
    type OutputTy = P<ast::Expr>;
    type AttrsTy = ast::AttrVec;
    const KIND: AstFragmentKind = AstFragmentKind::Expr;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Expr(self)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_expr()
    }
    fn descr() -> &'static str {
        "an expression"
    }
    fn noop_visit<V: MutVisitor>(&mut self, visitor: &mut V) {
        noop_visit_expr(self, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.kind, ExprKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.into_inner();
        match node.kind {
            ExprKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

struct OptExprTag;
impl InvocationCollectorNode for AstNodeWrapper<P<ast::Expr>, OptExprTag> {
    type OutputTy = Option<P<ast::Expr>>;
    type AttrsTy = ast::AttrVec;
    const KIND: AstFragmentKind = AstFragmentKind::OptExpr;
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Expr(self.wrapped)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        fragment.make_opt_expr()
    }
    fn noop_flat_map<V: MutVisitor>(mut self, visitor: &mut V) -> Self::OutputTy {
        noop_visit_expr(&mut self.wrapped, visitor);
        Some(self.wrapped)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, ast::ExprKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.wrapped.into_inner();
        match node.kind {
            ExprKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
    fn pre_flat_map_node_collect_attr(cfg: &StripUnconfigured<'_>, attr: &ast::Attribute) {
        cfg.maybe_emit_expr_attr_err(&attr);
    }
}

/// This struct is a hack to workaround unstable of `stmt_expr_attributes`.
/// It can be removed once that feature is stabilized.
struct MethodReceiverTag;
impl DummyAstNode for MethodReceiverTag {
    fn dummy() -> MethodReceiverTag {
        MethodReceiverTag
    }
}
impl InvocationCollectorNode for AstNodeWrapper<P<ast::Expr>, MethodReceiverTag> {
    type OutputTy = Self;
    type AttrsTy = ast::AttrVec;
    const KIND: AstFragmentKind = AstFragmentKind::MethodReceiverExpr;
    fn descr() -> &'static str {
        "an expression"
    }
    fn to_annotatable(self) -> Annotatable {
        Annotatable::Expr(self.wrapped)
    }
    fn fragment_to_output(fragment: AstFragment) -> Self::OutputTy {
        AstNodeWrapper::new(fragment.make_method_receiver_expr(), MethodReceiverTag)
    }
    fn noop_visit<V: MutVisitor>(&mut self, visitor: &mut V) {
        noop_visit_expr(&mut self.wrapped, visitor)
    }
    fn is_mac_call(&self) -> bool {
        matches!(self.wrapped.kind, ast::ExprKind::MacCall(..))
    }
    fn take_mac_call(self) -> (P<ast::MacCall>, Self::AttrsTy, AddSemicolon) {
        let node = self.wrapped.into_inner();
        match node.kind {
            ExprKind::MacCall(mac) => (mac, node.attrs, AddSemicolon::No),
            _ => unreachable!(),
        }
    }
}

struct InvocationCollector<'a, 'b> {
    cx: &'a mut ExtCtxt<'b>,
    invocations: Vec<(Invocation, Option<Lrc<SyntaxExtension>>)>,
    monotonic: bool,
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
    fn cfg(&self) -> StripUnconfigured<'_> {
        StripUnconfigured {
            sess: &self.cx.sess,
            features: self.cx.ecfg.features,
            config_tokens: false,
            lint_node_id: self.cx.current_expansion.lint_node_id,
        }
    }

    fn collect(&mut self, fragment_kind: AstFragmentKind, kind: InvocationKind) -> AstFragment {
        let expn_id = LocalExpnId::fresh_empty();
        let vis = kind.placeholder_visibility();
        self.invocations.push((
            Invocation {
                kind,
                fragment_kind,
                expansion_data: ExpansionData {
                    id: expn_id,
                    depth: self.cx.current_expansion.depth + 1,
                    ..self.cx.current_expansion.clone()
                },
            },
            None,
        ));
        placeholder(fragment_kind, NodeId::placeholder_from_expn_id(expn_id), vis)
    }

    fn collect_bang(&mut self, mac: P<ast::MacCall>, kind: AstFragmentKind) -> AstFragment {
        // cache the macro call span so that it can be
        // easily adjusted for incremental compilation
        let span = mac.span();
        self.collect(kind, InvocationKind::Bang { mac, span })
    }

    fn collect_attr(
        &mut self,
        (attr, pos, derives): (ast::Attribute, usize, Vec<ast::Path>),
        item: Annotatable,
        kind: AstFragmentKind,
    ) -> AstFragment {
        self.collect(kind, InvocationKind::Attr { attr, pos, item, derives })
    }

    /// If `item` is an attribute invocation, remove the attribute and return it together with
    /// its position and derives following it. We have to collect the derives in order to resolve
    /// legacy derive helpers (helpers written before derives that introduce them).
    fn take_first_attr(
        &self,
        item: &mut impl HasAttrs,
    ) -> Option<(ast::Attribute, usize, Vec<ast::Path>)> {
        let mut attr = None;

        let mut cfg_pos = None;
        let mut attr_pos = None;
        for (pos, attr) in item.attrs().iter().enumerate() {
            if !attr.is_doc_comment() && !self.cx.expanded_inert_attrs.is_marked(attr) {
                let name = attr.ident().map(|ident| ident.name);
                if name == Some(sym::cfg) || name == Some(sym::cfg_attr) {
                    cfg_pos = Some(pos); // a cfg attr found, no need to search anymore
                    break;
                } else if attr_pos.is_none()
                    && !name.is_some_and(rustc_feature::is_builtin_attr_name)
                {
                    attr_pos = Some(pos); // a non-cfg attr found, still may find a cfg attr
                }
            }
        }

        item.visit_attrs(|attrs| {
            attr = Some(match (cfg_pos, attr_pos) {
                (Some(pos), _) => (attrs.remove(pos), pos, Vec::new()),
                (_, Some(pos)) => {
                    let attr = attrs.remove(pos);
                    let following_derives = attrs[pos..]
                        .iter()
                        .filter(|a| a.has_name(sym::derive))
                        .flat_map(|a| a.meta_item_list().unwrap_or_default())
                        .filter_map(|nested_meta| match nested_meta {
                            NestedMetaItem::MetaItem(ast::MetaItem {
                                kind: MetaItemKind::Word,
                                path,
                                ..
                            }) => Some(path),
                            _ => None,
                        })
                        .collect();

                    (attr, pos, following_derives)
                }
                _ => return,
            });
        });

        attr
    }

    // Detect use of feature-gated or invalid attributes on macro invocations
    // since they will not be detected after macro expansion.
    fn check_attributes(&self, attrs: &[ast::Attribute], call: &ast::MacCall) {
        let features = self.cx.ecfg.features.unwrap();
        let mut attrs = attrs.iter().peekable();
        let mut span: Option<Span> = None;
        while let Some(attr) = attrs.next() {
            rustc_ast_passes::feature_gate::check_attribute(attr, self.cx.sess, features);
            validate_attr::check_attr(&self.cx.sess.parse_sess, attr);

            let current_span = if let Some(sp) = span { sp.to(attr.span) } else { attr.span };
            span = Some(current_span);

            if attrs.peek().is_some_and(|next_attr| next_attr.doc_str().is_some()) {
                continue;
            }

            if attr.is_doc_comment() {
                self.cx.sess.parse_sess.buffer_lint_with_diagnostic(
                    &UNUSED_DOC_COMMENTS,
                    current_span,
                    self.cx.current_expansion.lint_node_id,
                    "unused doc comment",
                    BuiltinLintDiagnostics::UnusedDocComment(attr.span),
                );
            } else if rustc_attr::is_builtin_attr(attr) {
                let attr_name = attr.ident().unwrap().name;
                // `#[cfg]` and `#[cfg_attr]` are special - they are
                // eagerly evaluated.
                if attr_name != sym::cfg && attr_name != sym::cfg_attr {
                    self.cx.sess.parse_sess.buffer_lint_with_diagnostic(
                        &UNUSED_ATTRIBUTES,
                        attr.span,
                        self.cx.current_expansion.lint_node_id,
                        format!("unused attribute `{}`", attr_name),
                        BuiltinLintDiagnostics::UnusedBuiltinAttribute {
                            attr_name,
                            macro_name: pprust::path_to_string(&call.path),
                            invoc_span: call.path.span,
                        },
                    );
                }
            }
        }
    }

    fn expand_cfg_true(
        &mut self,
        node: &mut impl HasAttrs,
        attr: ast::Attribute,
        pos: usize,
    ) -> (bool, Option<ast::MetaItem>) {
        let (res, meta_item) = self.cfg().cfg_true(&attr);
        if res {
            // FIXME: `cfg(TRUE)` attributes do not currently remove themselves during expansion,
            // and some tools like rustdoc and clippy rely on that. Find a way to remove them
            // while keeping the tools working.
            self.cx.expanded_inert_attrs.mark(&attr);
            node.visit_attrs(|attrs| attrs.insert(pos, attr));
        }

        (res, meta_item)
    }

    fn expand_cfg_attr(&self, node: &mut impl HasAttrs, attr: &ast::Attribute, pos: usize) {
        node.visit_attrs(|attrs| {
            // Repeated `insert` calls is inefficient, but the number of
            // insertions is almost always 0 or 1 in practice.
            for cfg in self.cfg().expand_cfg_attr(attr, false).into_iter().rev() {
                attrs.insert(pos, cfg)
            }
        });
    }

    fn flat_map_node<Node: InvocationCollectorNode<OutputTy: Default>>(
        &mut self,
        mut node: Node,
    ) -> Node::OutputTy {
        loop {
            return match self.take_first_attr(&mut node) {
                Some((attr, pos, derives)) => match attr.name_or_empty() {
                    sym::cfg => {
                        let (res, meta_item) = self.expand_cfg_true(&mut node, attr, pos);
                        if res {
                            continue;
                        }

                        if let Some(meta_item) = meta_item {
                            for name in node.declared_names() {
                                self.cx.resolver.append_stripped_cfg_item(
                                    self.cx.current_expansion.lint_node_id,
                                    name,
                                    meta_item.clone(),
                                )
                            }
                        }
                        Default::default()
                    }
                    sym::cfg_attr => {
                        self.expand_cfg_attr(&mut node, &attr, pos);
                        continue;
                    }
                    _ => {
                        Node::pre_flat_map_node_collect_attr(&self.cfg(), &attr);
                        self.collect_attr((attr, pos, derives), node.to_annotatable(), Node::KIND)
                            .make_ast::<Node>()
                    }
                },
                None if node.is_mac_call() => {
                    let (mac, attrs, add_semicolon) = node.take_mac_call();
                    self.check_attributes(&attrs, &mac);
                    let mut res = self.collect_bang(mac, Node::KIND).make_ast::<Node>();
                    Node::post_flat_map_node_collect_bang(&mut res, add_semicolon);
                    res
                }
                None => {
                    match Node::wrap_flat_map_node_noop_flat_map(node, self, |mut node, this| {
                        assign_id!(this, node.node_id_mut(), || node.noop_flat_map(this))
                    }) {
                        Ok(output) => output,
                        Err(returned_node) => {
                            node = returned_node;
                            continue;
                        }
                    }
                }
            };
        }
    }

    fn visit_node<Node: InvocationCollectorNode<OutputTy = Node> + DummyAstNode>(
        &mut self,
        node: &mut Node,
    ) {
        loop {
            return match self.take_first_attr(node) {
                Some((attr, pos, derives)) => match attr.name_or_empty() {
                    sym::cfg => {
                        let span = attr.span;
                        if self.expand_cfg_true(node, attr, pos).0 {
                            continue;
                        }

                        node.expand_cfg_false(self, pos, span);
                        continue;
                    }
                    sym::cfg_attr => {
                        self.expand_cfg_attr(node, &attr, pos);
                        continue;
                    }
                    _ => visit_clobber(node, |node| {
                        self.collect_attr((attr, pos, derives), node.to_annotatable(), Node::KIND)
                            .make_ast::<Node>()
                    }),
                },
                None if node.is_mac_call() => {
                    visit_clobber(node, |node| {
                        // Do not clobber unless it's actually a macro (uncommon case).
                        let (mac, attrs, _) = node.take_mac_call();
                        self.check_attributes(&attrs, &mac);
                        self.collect_bang(mac, Node::KIND).make_ast::<Node>()
                    })
                }
                None => {
                    assign_id!(self, node.node_id_mut(), || node.noop_visit(self))
                }
            };
        }
    }
}

impl<'a, 'b> MutVisitor for InvocationCollector<'a, 'b> {
    fn flat_map_item(&mut self, node: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_trait_item(&mut self, node: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        self.flat_map_node(AstNodeWrapper::new(node, TraitItemTag))
    }

    fn flat_map_impl_item(&mut self, node: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        self.flat_map_node(AstNodeWrapper::new(node, ImplItemTag))
    }

    fn flat_map_foreign_item(
        &mut self,
        node: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_variant(&mut self, node: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_field_def(&mut self, node: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_pat_field(&mut self, node: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_expr_field(&mut self, node: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_param(&mut self, node: ast::Param) -> SmallVec<[ast::Param; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_generic_param(
        &mut self,
        node: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_arm(&mut self, node: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        self.flat_map_node(node)
    }

    fn flat_map_stmt(&mut self, node: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        // FIXME: invocations in semicolon-less expressions positions are expanded as expressions,
        // changing that requires some compatibility measures.
        if node.is_expr() {
            // The only way that we can end up with a `MacCall` expression statement,
            // (as opposed to a `StmtKind::MacCall`) is if we have a macro as the
            // trailing expression in a block (e.g. `fn foo() { my_macro!() }`).
            // Record this information, so that we can report a more specific
            // `SEMICOLON_IN_EXPRESSIONS_FROM_MACROS` lint if needed.
            // See #78991 for an investigation of treating macros in this position
            // as statements, rather than expressions, during parsing.
            return match &node.kind {
                StmtKind::Expr(expr)
                    if matches!(**expr, ast::Expr { kind: ExprKind::MacCall(..), .. }) =>
                {
                    self.cx.current_expansion.is_trailing_mac = true;
                    // Don't use `assign_id` for this statement - it may get removed
                    // entirely due to a `#[cfg]` on the contained expression
                    let res = noop_flat_map_stmt(node, self);
                    self.cx.current_expansion.is_trailing_mac = false;
                    res
                }
                _ => noop_flat_map_stmt(node, self),
            };
        }

        self.flat_map_node(node)
    }

    fn visit_crate(&mut self, node: &mut ast::Crate) {
        self.visit_node(node)
    }

    fn visit_ty(&mut self, node: &mut P<ast::Ty>) {
        self.visit_node(node)
    }

    fn visit_pat(&mut self, node: &mut P<ast::Pat>) {
        self.visit_node(node)
    }

    fn visit_expr(&mut self, node: &mut P<ast::Expr>) {
        // FIXME: Feature gating is performed inconsistently between `Expr` and `OptExpr`.
        if let Some(attr) = node.attrs.first() {
            self.cfg().maybe_emit_expr_attr_err(attr);
        }
        self.visit_node(node)
    }

    fn visit_method_receiver_expr(&mut self, node: &mut P<ast::Expr>) {
        visit_clobber(node, |node| {
            let mut wrapper = AstNodeWrapper::new(node, MethodReceiverTag);
            self.visit_node(&mut wrapper);
            wrapper.wrapped
        })
    }

    fn filter_map_expr(&mut self, node: P<ast::Expr>) -> Option<P<ast::Expr>> {
        self.flat_map_node(AstNodeWrapper::new(node, OptExprTag))
    }

    fn visit_block(&mut self, node: &mut P<ast::Block>) {
        let orig_dir_ownership = mem::replace(
            &mut self.cx.current_expansion.dir_ownership,
            DirOwnership::UnownedViaBlock,
        );
        noop_visit_block(node, self);
        self.cx.current_expansion.dir_ownership = orig_dir_ownership;
    }

    fn visit_id(&mut self, id: &mut NodeId) {
        // We may have already assigned a `NodeId`
        // by calling `assign_id`
        if self.monotonic && *id == ast::DUMMY_NODE_ID {
            *id = self.cx.resolver.next_node_id();
        }
    }
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: String,
    pub features: Option<&'feat Features>,
    pub recursion_limit: Limit,
    pub trace_mac: bool,
    /// If false, strip `#[test]` nodes
    pub should_test: bool,
    /// If true, use verbose debugging for `proc_macro::Span`
    pub span_debug: bool,
    /// If true, show backtraces for proc-macro panics
    pub proc_macro_backtrace: bool,
}

impl<'feat> ExpansionConfig<'feat> {
    pub fn default(crate_name: String) -> ExpansionConfig<'static> {
        ExpansionConfig {
            crate_name,
            features: None,
            recursion_limit: Limit::new(1024),
            trace_mac: false,
            should_test: false,
            span_debug: false,
            proc_macro_backtrace: false,
        }
    }

    fn proc_macro_hygiene(&self) -> bool {
        self.features.is_some_and(|features| features.proc_macro_hygiene)
    }
}
