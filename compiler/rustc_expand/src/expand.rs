use crate::base::*;
use crate::config::StripUnconfigured;
use crate::configure;
use crate::hygiene::SyntaxContext;
use crate::mbe::macro_rules::annotate_err_with_kind;
use crate::module::{mod_dir_path, parse_external_mod, DirOwnership, ParsedExternalMod};
use crate::placeholders::{placeholder, PlaceholderExpander};

use rustc_ast as ast;
use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{AstLike, Block, Inline, ItemKind, MacArgs, MacCall};
use rustc_ast::{MacCallStmt, MacStmtStyle, MetaItemKind, ModKind, NestedMetaItem};
use rustc_ast::{NodeId, PatKind, Path, StmtKind, Unsafe};
use rustc_ast_pretty::pprust;
use rustc_attr::is_builtin_attr;
use rustc_data_structures::map_in_place::MapInPlace;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, FatalError, PResult};
use rustc_feature::Features;
use rustc_parse::parser::{
    AttemptLocalParseRecovery, ForceCollect, Parser, RecoverColon, RecoverComma,
};
use rustc_parse::validate_attr;
use rustc_session::lint::builtin::{UNUSED_ATTRIBUTES, UNUSED_DOC_COMMENTS};
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_session::parse::{feature_err, ParseSess};
use rustc_session::Limit;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{FileName, LocalExpnId, Span};

use smallvec::{smallvec, SmallVec};
use std::ops::DerefMut;
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
    Fields(SmallVec<[ast::ExprField; 1]>) {
        "field expression"; many fn flat_map_expr_field; fn visit_expr_field(); fn make_expr_fields;
    }
    FieldPats(SmallVec<[ast::PatField; 1]>) {
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
    StructFields(SmallVec<[ast::FieldDef; 1]>) {
        "field";
        many fn flat_map_field_def;
        fn visit_field_def();
        fn make_field_defs;
    }
    Variants(SmallVec<[ast::Variant; 1]>) {
        "variant"; many fn flat_map_variant; fn visit_variant(); fn make_variants;
    }
}

pub enum SupportsMacroExpansion {
    No,
    Yes { supports_inner_attrs: bool },
}

impl AstFragmentKind {
    crate fn dummy(self, span: Span) -> AstFragment {
        self.make_from(DummyResult::any(span)).expect("couldn't create a dummy AST fragment")
    }

    pub fn supports_macro_expansion(self) -> SupportsMacroExpansion {
        match self {
            AstFragmentKind::OptExpr
            | AstFragmentKind::Expr
            | AstFragmentKind::Stmts
            | AstFragmentKind::Ty
            | AstFragmentKind::Pat => SupportsMacroExpansion::Yes { supports_inner_attrs: false },
            AstFragmentKind::Items
            | AstFragmentKind::TraitItems
            | AstFragmentKind::ImplItems
            | AstFragmentKind::ForeignItems => {
                SupportsMacroExpansion::Yes { supports_inner_attrs: true }
            }
            AstFragmentKind::Arms
            | AstFragmentKind::Fields
            | AstFragmentKind::FieldPats
            | AstFragmentKind::GenericParams
            | AstFragmentKind::Params
            | AstFragmentKind::StructFields
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
            AstFragmentKind::Fields => {
                AstFragment::Fields(items.map(Annotatable::expect_expr_field).collect())
            }
            AstFragmentKind::FieldPats => {
                AstFragment::FieldPats(items.map(Annotatable::expect_pat_field).collect())
            }
            AstFragmentKind::GenericParams => {
                AstFragment::GenericParams(items.map(Annotatable::expect_generic_param).collect())
            }
            AstFragmentKind::Params => {
                AstFragment::Params(items.map(Annotatable::expect_param).collect())
            }
            AstFragmentKind::StructFields => {
                AstFragment::StructFields(items.map(Annotatable::expect_field_def).collect())
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
        mac: ast::MacCall,
        span: Span,
    },
    Attr {
        attr: ast::Attribute,
        // Re-insertion position for inert attributes.
        pos: usize,
        item: Annotatable,
        // Required for resolving derive helper attributes.
        derives: Vec<Path>,
    },
    Derive {
        path: Path,
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

    // FIXME: Avoid visiting the crate as a `Mod` item,
    // make crate a first class expansion target instead.
    pub fn expand_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        let file_path = match self.cx.source_map().span_to_filename(krate.span) {
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

        let krate_item = AstFragment::Items(smallvec![P(ast::Item {
            attrs: krate.attrs,
            span: krate.span,
            kind: ast::ItemKind::Mod(
                Unsafe::No,
                ModKind::Loaded(krate.items, Inline::Yes, krate.span)
            ),
            ident: Ident::invalid(),
            id: ast::DUMMY_NODE_ID,
            vis: ast::Visibility {
                span: krate.span.shrink_to_lo(),
                kind: ast::VisibilityKind::Public,
                tokens: None,
            },
            tokens: None,
        })]);

        match self.fully_expand_fragment(krate_item).make_items().pop().map(P::into_inner) {
            Some(ast::Item {
                attrs,
                kind: ast::ItemKind::Mod(_, ModKind::Loaded(items, ..)),
                ..
            }) => {
                krate.attrs = attrs;
                krate.items = items;
            }
            None => {
                // Resolution failed so we return an empty expansion
                krate.attrs = vec![];
                krate.items = vec![];
            }
            Some(ast::Item { span, kind, .. }) => {
                krate.attrs = vec![];
                krate.items = vec![];
                self.cx.span_err(
                    span,
                    &format!(
                        "expected crate top-level item to be a module after macro expansion, found {} {}",
                        kind.article(), kind.descr()
                    ),
                );
                // FIXME: this workaround issue #84569
                FatalError.raise();
            }
        };
        self.cx.trace_macros_diag();
        krate
    }

    // Recursively expand all macro invocations in this AST fragment.
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
            let (invoc, ext) = if let Some(invoc) = invocations.pop() {
                invoc
            } else {
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
                                .map(|(path, item, _exts)| {
                                    // FIXME: Consider using the derive resolutions (`_exts`)
                                    // instead of enqueuing the derives to be resolved again later.
                                    let expn_id = LocalExpnId::fresh_empty();
                                    derive_invocations.push((
                                        Invocation {
                                            kind: InvocationKind::Derive { path, item },
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
                cfg: StripUnconfigured {
                    sess: &self.cx.sess,
                    features: self.cx.ecfg.features,
                    config_tokens: false,
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

            if self.cx.sess.opts.debugging_opts.incremental_relative_spans {
                for (invoc, _) in invocations.iter_mut() {
                    let expn_id = invoc.expansion_data.id;
                    let parent_def = self.cx.resolver.invocation_parent(expn_id);
                    let span = match &mut invoc.kind {
                        InvocationKind::Bang { ref mut span, .. } => span,
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
    }

    /// A macro's expansion does not fit in this fragment kind.
    /// For example, a non-type macro in a type position.
    fn error_wrong_fragment_kind(&mut self, kind: AstFragmentKind, mac: &ast::MacCall, span: Span) {
        let msg = format!(
            "non-{kind} macro in {kind} position: {path}",
            kind = kind.name(),
            path = pprust::path_to_string(&mac.path),
        );
        self.cx.span_err(span, &msg);
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
                    let tok_result = match expander.expand(self.cx, span, mac.args.inner_tokens()) {
                        Err(_) => return ExpandResult::Ready(fragment_kind.dummy(span)),
                        Ok(ts) => ts,
                    };
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
            InvocationKind::Attr { attr, pos, mut item, derives } => match ext {
                SyntaxExtensionKind::Attr(expander) => {
                    self.gate_proc_macro_input(&item);
                    self.gate_proc_macro_attr_item(span, &item);
                    let mut fake_tokens = false;
                    if let Annotatable::Item(item_inner) = &item {
                        if let ItemKind::Mod(_, mod_kind) = &item_inner.kind {
                            // FIXME: Collect tokens and use them instead of generating
                            // fake ones. These are unstable, so it needs to be
                            // fixed prior to stabilization
                            // Fake tokens when we are invoking an inner attribute, and:
                            fake_tokens = matches!(attr.style, ast::AttrStyle::Inner) &&
                                // We are invoking an attribute on the crate root, or an outline
                                // module
                                (item_inner.ident.name.is_empty() || !matches!(mod_kind, ast::ModKind::Loaded(_, Inline::Yes, _)));
                        }
                    }
                    let tokens = if fake_tokens {
                        rustc_parse::fake_token_stream(
                            &self.cx.sess.parse_sess,
                            &item.into_nonterminal(),
                        )
                    } else {
                        item.into_tokens(&self.cx.sess.parse_sess)
                    };
                    let attr_item = attr.unwrap_normal_item();
                    if let MacArgs::Eq(..) = attr_item.args {
                        self.cx.span_err(span, "key-value macro attributes are not supported");
                    }
                    let inner_tokens = attr_item.args.inner_tokens();
                    let tok_result = match expander.expand(self.cx, span, inner_tokens, tokens) {
                        Err(_) => return ExpandResult::Ready(fragment_kind.dummy(span)),
                        Ok(ts) => ts,
                    };
                    self.parse_ast_fragment(tok_result, fragment_kind, &attr_item.path, span)
                }
                SyntaxExtensionKind::LegacyAttr(expander) => {
                    match validate_attr::parse_meta(&self.cx.sess.parse_sess, &attr) {
                        Ok(meta) => {
                            let items = match expander.expand(self.cx, span, &meta, item) {
                                ExpandResult::Ready(items) => items,
                                ExpandResult::Retry(item) => {
                                    // Reassemble the original invocation for retrying.
                                    return ExpandResult::Retry(Invocation {
                                        kind: InvocationKind::Attr { attr, pos, item, derives },
                                        ..invoc
                                    });
                                }
                            };
                            if fragment_kind == AstFragmentKind::Expr && items.is_empty() {
                                let msg =
                                    "removing an expression is not supported in this position";
                                self.cx.span_err(span, msg);
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
            InvocationKind::Derive { path, item } => match ext {
                SyntaxExtensionKind::Derive(expander)
                | SyntaxExtensionKind::LegacyDerive(expander) => {
                    if let SyntaxExtensionKind::Derive(..) = ext {
                        self.gate_proc_macro_input(&item);
                    }
                    let meta = ast::MetaItem { kind: ast::MetaItemKind::Word, span, path };
                    let items = match expander.expand(self.cx, span, &meta, item) {
                        ExpandResult::Ready(items) => items,
                        ExpandResult::Retry(item) => {
                            // Reassemble the original invocation for retrying.
                            return ExpandResult::Retry(Invocation {
                                kind: InvocationKind::Derive { path: meta.path, item },
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
            | Annotatable::ForeignItem(_) => return,
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
                    ast::ItemKind::Mod(_, mod_kind)
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
            while this.token != token::Eof && this.token != token::CloseDelim(token::Brace) {
                if let Some(stmt) = this.parse_full_stmt(AttemptLocalParseRecovery::Yes)? {
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
        AstFragmentKind::Pat => AstFragment::Pat(this.parse_pat_allow_top_alt(
            None,
            RecoverComma::No,
            RecoverColon::Yes,
        )?),
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
    invocations: Vec<(Invocation, Option<Lrc<SyntaxExtension>>)>,
    monotonic: bool,
}

impl<'a, 'b> InvocationCollector<'a, 'b> {
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

    fn collect_bang(
        &mut self,
        mac: ast::MacCall,
        span: Span,
        kind: AstFragmentKind,
    ) -> AstFragment {
        self.collect(kind, InvocationKind::Bang { mac, span })
    }

    fn collect_attr(
        &mut self,
        (attr, pos, derives): (ast::Attribute, usize, Vec<Path>),
        item: Annotatable,
        kind: AstFragmentKind,
    ) -> AstFragment {
        self.collect(kind, InvocationKind::Attr { attr, pos, item, derives })
    }

    /// If `item` is an attribute invocation, remove the attribute and return it together with
    /// its position and derives following it. We have to collect the derives in order to resolve
    /// legacy derive helpers (helpers written before derives that introduce them).
    fn take_first_attr(
        &mut self,
        item: &mut impl AstLike,
    ) -> Option<(ast::Attribute, usize, Vec<Path>)> {
        let mut attr = None;

        item.visit_attrs(|attrs| {
            attr = attrs
                .iter()
                .position(|a| !self.cx.expanded_inert_attrs.is_marked(a) && !is_builtin_attr(a))
                .map(|attr_pos| {
                    let attr = attrs.remove(attr_pos);
                    let following_derives = attrs[attr_pos..]
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

                    (attr, attr_pos, following_derives)
                })
        });

        attr
    }

    fn take_stmt_bang(
        &mut self,
        stmt: ast::Stmt,
    ) -> Result<(bool, MacCall, Vec<ast::Attribute>), ast::Stmt> {
        match stmt.kind {
            StmtKind::MacCall(mac) => {
                let MacCallStmt { mac, style, attrs, .. } = mac.into_inner();
                Ok((style == MacStmtStyle::Semicolon, mac, attrs.into()))
            }
            StmtKind::Item(ref item) if matches!(item.kind, ItemKind::MacCall(..)) => {
                match stmt.kind {
                    StmtKind::Item(item) => match item.into_inner() {
                        ast::Item { kind: ItemKind::MacCall(mac), attrs, .. } => {
                            Ok((mac.args.need_semicolon(), mac, attrs))
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
            StmtKind::Semi(ref expr) if matches!(expr.kind, ast::ExprKind::MacCall(..)) => {
                match stmt.kind {
                    StmtKind::Semi(expr) => match expr.into_inner() {
                        ast::Expr { kind: ast::ExprKind::MacCall(mac), attrs, .. } => {
                            Ok((mac.args.need_semicolon(), mac, attrs.into()))
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
            StmtKind::Local(..) | StmtKind::Empty | StmtKind::Item(..) | StmtKind::Semi(..) => {
                Err(stmt)
            }
            StmtKind::Expr(..) => unreachable!(),
        }
    }

    fn configure<T: AstLike>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }

    // Detect use of feature-gated or invalid attributes on macro invocations
    // since they will not be detected after macro expansion.
    fn check_attributes(&self, attrs: &[ast::Attribute], call: &MacCall) {
        let features = self.cx.ecfg.features.unwrap();
        let mut attrs = attrs.iter().peekable();
        let mut span: Option<Span> = None;
        while let Some(attr) = attrs.next() {
            rustc_ast_passes::feature_gate::check_attribute(attr, self.cx.sess, features);
            validate_attr::check_meta(&self.cx.sess.parse_sess, attr);

            let current_span = if let Some(sp) = span { sp.to(attr.span) } else { attr.span };
            span = Some(current_span);

            if attrs.peek().map_or(false, |next_attr| next_attr.doc_str().is_some()) {
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
                        &format!("unused attribute `{}`", attr_name),
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

impl<'a, 'b> MutVisitor for InvocationCollector<'a, 'b> {
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        visit_clobber(expr.deref_mut(), |mut expr| {
            if let Some(attr) = self.take_first_attr(&mut expr) {
                // Collect the invoc regardless of whether or not attributes are permitted here
                // expansion will eat the attribute so it won't error later.
                self.cfg.maybe_emit_expr_attr_err(&attr.0);

                // AstFragmentKind::Expr requires the macro to emit an expression.
                return self
                    .collect_attr(attr, Annotatable::Expr(P(expr)), AstFragmentKind::Expr)
                    .make_expr()
                    .into_inner();
            }

            if let ast::ExprKind::MacCall(mac) = expr.kind {
                self.check_attributes(&expr.attrs, &mac);
                self.collect_bang(mac, expr.span, AstFragmentKind::Expr).make_expr().into_inner()
            } else {
                assign_id!(self, &mut expr.id, || {
                    ensure_sufficient_stack(|| noop_visit_expr(&mut expr, self));
                });
                expr
            }
        });
    }

    fn flat_map_arm(&mut self, arm: ast::Arm) -> SmallVec<[ast::Arm; 1]> {
        let mut arm = configure!(self, arm);

        if let Some(attr) = self.take_first_attr(&mut arm) {
            return self
                .collect_attr(attr, Annotatable::Arm(arm), AstFragmentKind::Arms)
                .make_arms();
        }

        assign_id!(self, &mut arm.id, || noop_flat_map_arm(arm, self))
    }

    fn flat_map_expr_field(&mut self, field: ast::ExprField) -> SmallVec<[ast::ExprField; 1]> {
        let mut field = configure!(self, field);

        if let Some(attr) = self.take_first_attr(&mut field) {
            return self
                .collect_attr(attr, Annotatable::ExprField(field), AstFragmentKind::Fields)
                .make_expr_fields();
        }

        assign_id!(self, &mut field.id, || noop_flat_map_expr_field(field, self))
    }

    fn flat_map_pat_field(&mut self, fp: ast::PatField) -> SmallVec<[ast::PatField; 1]> {
        let mut fp = configure!(self, fp);

        if let Some(attr) = self.take_first_attr(&mut fp) {
            return self
                .collect_attr(attr, Annotatable::PatField(fp), AstFragmentKind::FieldPats)
                .make_pat_fields();
        }

        assign_id!(self, &mut fp.id, || noop_flat_map_pat_field(fp, self))
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        let mut p = configure!(self, p);

        if let Some(attr) = self.take_first_attr(&mut p) {
            return self
                .collect_attr(attr, Annotatable::Param(p), AstFragmentKind::Params)
                .make_params();
        }

        assign_id!(self, &mut p.id, || noop_flat_map_param(p, self))
    }

    fn flat_map_field_def(&mut self, sf: ast::FieldDef) -> SmallVec<[ast::FieldDef; 1]> {
        let mut sf = configure!(self, sf);

        if let Some(attr) = self.take_first_attr(&mut sf) {
            return self
                .collect_attr(attr, Annotatable::FieldDef(sf), AstFragmentKind::StructFields)
                .make_field_defs();
        }

        assign_id!(self, &mut sf.id, || noop_flat_map_field_def(sf, self))
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        let mut variant = configure!(self, variant);

        if let Some(attr) = self.take_first_attr(&mut variant) {
            return self
                .collect_attr(attr, Annotatable::Variant(variant), AstFragmentKind::Variants)
                .make_variants();
        }

        assign_id!(self, &mut variant.id, || noop_flat_map_variant(variant, self))
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let expr = configure!(self, expr);
        expr.filter_map(|mut expr| {
            if let Some(attr) = self.take_first_attr(&mut expr) {
                self.cfg.maybe_emit_expr_attr_err(&attr.0);

                return self
                    .collect_attr(attr, Annotatable::Expr(P(expr)), AstFragmentKind::OptExpr)
                    .make_opt_expr()
                    .map(|expr| expr.into_inner());
            }

            if let ast::ExprKind::MacCall(mac) = expr.kind {
                self.check_attributes(&expr.attrs, &mac);
                self.collect_bang(mac, expr.span, AstFragmentKind::OptExpr)
                    .make_opt_expr()
                    .map(|expr| expr.into_inner())
            } else {
                assign_id!(self, &mut expr.id, || {
                    Some({
                        noop_visit_expr(&mut expr, self);
                        expr
                    })
                })
            }
        })
    }

    fn visit_pat(&mut self, pat: &mut P<ast::Pat>) {
        match pat.kind {
            PatKind::MacCall(_) => {}
            _ => return noop_visit_pat(pat, self),
        }

        visit_clobber(pat, |mut pat| match mem::replace(&mut pat.kind, PatKind::Wild) {
            PatKind::MacCall(mac) => {
                self.collect_bang(mac, pat.span, AstFragmentKind::Pat).make_pat()
            }
            _ => unreachable!(),
        });
    }

    fn flat_map_stmt(&mut self, stmt: ast::Stmt) -> SmallVec<[ast::Stmt; 1]> {
        let mut stmt = configure!(self, stmt);

        // We pull macro invocations (both attributes and fn-like macro calls) out of their
        // `StmtKind`s and treat them as statement macro invocations, not as items or expressions.
        // FIXME: invocations in semicolon-less expressions positions are expanded as expressions,
        // changing that requires some compatibility measures.
        let mut stmt = if !stmt.is_expr() {
            if let Some(attr) = self.take_first_attr(&mut stmt) {
                return self
                    .collect_attr(attr, Annotatable::Stmt(P(stmt)), AstFragmentKind::Stmts)
                    .make_stmts();
            }

            let span = stmt.span;
            match self.take_stmt_bang(stmt) {
                Ok((add_semicolon, mac, attrs)) => {
                    self.check_attributes(&attrs, &mac);
                    let mut stmts =
                        self.collect_bang(mac, span, AstFragmentKind::Stmts).make_stmts();

                    // If this is a macro invocation with a semicolon, then apply that
                    // semicolon to the final statement produced by expansion.
                    if add_semicolon {
                        if let Some(stmt) = stmts.pop() {
                            stmts.push(stmt.add_trailing_semicolon());
                        }
                    }

                    return stmts;
                }
                Err(stmt) => stmt,
            }
        } else {
            stmt
        };

        // The only way that we can end up with a `MacCall` expression statement,
        // (as opposed to a `StmtKind::MacCall`) is if we have a macro as the
        // traiing expression in a block (e.g. `fn foo() { my_macro!() }`).
        // Record this information, so that we can report a more specific
        // `SEMICOLON_IN_EXPRESSIONS_FROM_MACROS` lint if needed.
        // See #78991 for an investigation of treating macros in this position
        // as statements, rather than expressions, during parsing.
        let res = match &stmt.kind {
            StmtKind::Expr(expr)
                if matches!(**expr, ast::Expr { kind: ast::ExprKind::MacCall(..), .. }) =>
            {
                self.cx.current_expansion.is_trailing_mac = true;
                // Don't use `assign_id` for this statement - it may get removed
                // entirely due to a `#[cfg]` on the contained expression
                noop_flat_map_stmt(stmt, self)
            }
            _ => assign_id!(self, &mut stmt.id, || noop_flat_map_stmt(stmt, self)),
        };
        self.cx.current_expansion.is_trailing_mac = false;
        res
    }

    fn visit_block(&mut self, block: &mut P<Block>) {
        let orig_dir_ownership = mem::replace(
            &mut self.cx.current_expansion.dir_ownership,
            DirOwnership::UnownedViaBlock,
        );
        noop_visit_block(block, self);
        self.cx.current_expansion.dir_ownership = orig_dir_ownership;
    }

    fn flat_map_item(&mut self, item: P<ast::Item>) -> SmallVec<[P<ast::Item>; 1]> {
        let mut item = configure!(self, item);

        if let Some(attr) = self.take_first_attr(&mut item) {
            return self
                .collect_attr(attr, Annotatable::Item(item), AstFragmentKind::Items)
                .make_items();
        }

        let mut attrs = mem::take(&mut item.attrs); // We do this to please borrowck.
        let ident = item.ident;
        let span = item.span;

        match item.kind {
            ast::ItemKind::MacCall(ref mac) => {
                self.check_attributes(&attrs, &mac);
                item.attrs = attrs;
                item.and_then(|item| match item.kind {
                    ItemKind::MacCall(mac) => {
                        self.collect_bang(mac, span, AstFragmentKind::Items).make_items()
                    }
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(_, ref mut mod_kind) if ident != Ident::invalid() => {
                let (file_path, dir_path, dir_ownership) = match mod_kind {
                    ModKind::Loaded(_, inline, _) => {
                        // Inline `mod foo { ... }`, but we still need to push directories.
                        let (dir_path, dir_ownership) = mod_dir_path(
                            &self.cx.sess,
                            ident,
                            &attrs,
                            &self.cx.current_expansion.module,
                            self.cx.current_expansion.dir_ownership,
                            *inline,
                        );
                        item.attrs = attrs;
                        (None, dir_path, dir_ownership)
                    }
                    ModKind::Unloaded => {
                        // We have an outline `mod foo;` so we need to parse the file.
                        let old_attrs_len = attrs.len();
                        let ParsedExternalMod {
                            mut items,
                            inner_span,
                            file_path,
                            dir_path,
                            dir_ownership,
                        } = parse_external_mod(
                            &self.cx.sess,
                            ident,
                            span,
                            &self.cx.current_expansion.module,
                            self.cx.current_expansion.dir_ownership,
                            &mut attrs,
                        );

                        if let Some(extern_mod_loaded) = self.cx.extern_mod_loaded {
                            (attrs, items) = extern_mod_loaded(ident, attrs, items, inner_span);
                        }

                        *mod_kind = ModKind::Loaded(items, Inline::No, inner_span);
                        item.attrs = attrs;
                        if item.attrs.len() > old_attrs_len {
                            // If we loaded an out-of-line module and added some inner attributes,
                            // then we need to re-configure it and re-collect attributes for
                            // resolution and expansion.
                            item = configure!(self, item);

                            if let Some(attr) = self.take_first_attr(&mut item) {
                                return self
                                    .collect_attr(
                                        attr,
                                        Annotatable::Item(item),
                                        AstFragmentKind::Items,
                                    )
                                    .make_items();
                            }
                        }
                        (Some(file_path), dir_path, dir_ownership)
                    }
                };

                // Set the module info before we flat map.
                let mut module = self.cx.current_expansion.module.with_dir_path(dir_path);
                module.mod_path.push(ident);
                if let Some(file_path) = file_path {
                    module.file_path_stack.push(file_path);
                }

                let orig_module =
                    mem::replace(&mut self.cx.current_expansion.module, Rc::new(module));
                let orig_dir_ownership =
                    mem::replace(&mut self.cx.current_expansion.dir_ownership, dir_ownership);

                let result = assign_id!(self, &mut item.id, || noop_flat_map_item(item, self));

                // Restore the module info.
                self.cx.current_expansion.dir_ownership = orig_dir_ownership;
                self.cx.current_expansion.module = orig_module;

                result
            }
            _ => {
                item.attrs = attrs;
                // The crate root is special - don't assign an ID to it.
                if !(matches!(item.kind, ast::ItemKind::Mod(..)) && ident == Ident::invalid()) {
                    assign_id!(self, &mut item.id, || noop_flat_map_item(item, self))
                } else {
                    noop_flat_map_item(item, self)
                }
            }
        }
    }

    fn flat_map_trait_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let mut item = configure!(self, item);

        if let Some(attr) = self.take_first_attr(&mut item) {
            return self
                .collect_attr(attr, Annotatable::TraitItem(item), AstFragmentKind::TraitItems)
                .make_trait_items();
        }

        match item.kind {
            ast::AssocItemKind::MacCall(ref mac) => {
                self.check_attributes(&item.attrs, &mac);
                item.and_then(|item| match item.kind {
                    ast::AssocItemKind::MacCall(mac) => self
                        .collect_bang(mac, item.span, AstFragmentKind::TraitItems)
                        .make_trait_items(),
                    _ => unreachable!(),
                })
            }
            _ => {
                assign_id!(self, &mut item.id, || noop_flat_map_assoc_item(item, self))
            }
        }
    }

    fn flat_map_impl_item(&mut self, item: P<ast::AssocItem>) -> SmallVec<[P<ast::AssocItem>; 1]> {
        let mut item = configure!(self, item);

        if let Some(attr) = self.take_first_attr(&mut item) {
            return self
                .collect_attr(attr, Annotatable::ImplItem(item), AstFragmentKind::ImplItems)
                .make_impl_items();
        }

        match item.kind {
            ast::AssocItemKind::MacCall(ref mac) => {
                self.check_attributes(&item.attrs, &mac);
                item.and_then(|item| match item.kind {
                    ast::AssocItemKind::MacCall(mac) => self
                        .collect_bang(mac, item.span, AstFragmentKind::ImplItems)
                        .make_impl_items(),
                    _ => unreachable!(),
                })
            }
            _ => {
                assign_id!(self, &mut item.id, || noop_flat_map_assoc_item(item, self))
            }
        }
    }

    fn visit_ty(&mut self, ty: &mut P<ast::Ty>) {
        match ty.kind {
            ast::TyKind::MacCall(_) => {}
            _ => return noop_visit_ty(ty, self),
        };

        visit_clobber(ty, |mut ty| match mem::replace(&mut ty.kind, ast::TyKind::Err) {
            ast::TyKind::MacCall(mac) => {
                self.collect_bang(mac, ty.span, AstFragmentKind::Ty).make_ty()
            }
            _ => unreachable!(),
        });
    }

    fn flat_map_foreign_item(
        &mut self,
        foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
        let mut foreign_item = configure!(self, foreign_item);

        if let Some(attr) = self.take_first_attr(&mut foreign_item) {
            return self
                .collect_attr(
                    attr,
                    Annotatable::ForeignItem(foreign_item),
                    AstFragmentKind::ForeignItems,
                )
                .make_foreign_items();
        }

        match foreign_item.kind {
            ast::ForeignItemKind::MacCall(ref mac) => {
                self.check_attributes(&foreign_item.attrs, &mac);
                foreign_item.and_then(|item| match item.kind {
                    ast::ForeignItemKind::MacCall(mac) => self
                        .collect_bang(mac, item.span, AstFragmentKind::ForeignItems)
                        .make_foreign_items(),
                    _ => unreachable!(),
                })
            }
            _ => {
                assign_id!(self, &mut foreign_item.id, || noop_flat_map_foreign_item(
                    foreign_item,
                    self
                ))
            }
        }
    }

    fn flat_map_generic_param(
        &mut self,
        param: ast::GenericParam,
    ) -> SmallVec<[ast::GenericParam; 1]> {
        let mut param = configure!(self, param);

        if let Some(attr) = self.take_first_attr(&mut param) {
            return self
                .collect_attr(
                    attr,
                    Annotatable::GenericParam(param),
                    AstFragmentKind::GenericParams,
                )
                .make_generic_params();
        }

        assign_id!(self, &mut param.id, || noop_flat_map_generic_param(param, self))
    }

    fn visit_id(&mut self, id: &mut ast::NodeId) {
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
    pub should_test: bool,          // If false, strip `#[test]` nodes
    pub span_debug: bool,           // If true, use verbose debugging for `proc_macro::Span`
    pub proc_macro_backtrace: bool, // If true, show backtraces for proc-macro panics
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
        self.features.map_or(false, |features| features.proc_macro_hygiene)
    }
}
