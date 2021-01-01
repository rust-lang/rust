use crate::base::*;
use crate::config::StripUnconfigured;
use crate::configure;
use crate::hygiene::{ExpnData, ExpnKind, SyntaxContext};
use crate::mbe::macro_rules::annotate_err_with_kind;
use crate::module::{parse_external_mod, push_directory, Directory, DirectoryOwnership};
use crate::placeholders::{placeholder, PlaceholderExpander};
use crate::proc_macro::collect_derives;

use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{self as ast, AttrItem, AttrStyle, Block, LitKind, NodeId, PatKind, Path};
use rustc_ast::{ItemKind, MacArgs, MacCallStmt, MacStmtStyle, StmtKind, Unsafe};
use rustc_ast_pretty::pprust;
use rustc_attr::{self as attr, is_builtin_attr, HasAttrs};
use rustc_data_structures::map_in_place::MapInPlace;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{struct_span_err, Applicability, PResult};
use rustc_feature::Features;
use rustc_parse::parser::{AttemptLocalParseRecovery, Parser};
use rustc_parse::validate_attr;
use rustc_session::lint::builtin::UNUSED_DOC_COMMENTS;
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_session::parse::{feature_err, ParseSess};
use rustc_session::Limit;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{ExpnId, FileName, Span, DUMMY_SP};

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
    crate fn dummy(self, span: Span) -> AstFragment {
        self.make_from(DummyResult::any(span)).expect("couldn't create a dummy AST fragment")
    }

    /// Fragment supports macro expansion and not just inert attributes, `cfg` and `cfg_attr`.
    pub fn supports_macro_expansion(self) -> bool {
        match self {
            AstFragmentKind::OptExpr
            | AstFragmentKind::Expr
            | AstFragmentKind::Pat
            | AstFragmentKind::Ty
            | AstFragmentKind::Stmts
            | AstFragmentKind::Items
            | AstFragmentKind::TraitItems
            | AstFragmentKind::ImplItems
            | AstFragmentKind::ForeignItems => true,
            AstFragmentKind::Arms
            | AstFragmentKind::Fields
            | AstFragmentKind::FieldPats
            | AstFragmentKind::GenericParams
            | AstFragmentKind::Params
            | AstFragmentKind::StructFields
            | AstFragmentKind::Variants => false,
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
        mac: ast::MacCall,
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
                FileName::Real(name) => name.into_local_path(),
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
            vis: ast::Visibility {
                span: krate.span.shrink_to_lo(),
                kind: ast::VisibilityKind::Public,
                tokens: None,
            },
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
                krate.module = ast::Mod {
                    inner: orig_mod_span,
                    unsafety: Unsafe::No,
                    items: vec![],
                    inline: true,
                };
            }
            Some(ast::Item { span, kind, .. }) => {
                krate.attrs = vec![];
                krate.module = ast::Mod {
                    inner: orig_mod_span,
                    unsafety: Unsafe::No,
                    items: vec![],
                    inline: true,
                };
                self.cx.span_err(
                    span,
                    &format!(
                        "expected crate top-level item to be a module after macro expansion, found {} {}",
                        kind.article(), kind.descr()
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
        let orig_force_mode = self.cx.force_mode;
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
            let (invoc, res) = if let Some(invoc) = invocations.pop() {
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

            let res = match res {
                Some(res) => res,
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
                        Ok(res) => res,
                        Err(Indeterminate) => {
                            // Cannot resolve, will retry this invocation later.
                            undetermined_invocations.push((invoc, None));
                            continue;
                        }
                    }
                }
            };

            let ExpansionData { depth, id: expn_id, .. } = invoc.expansion_data;
            self.cx.current_expansion = invoc.expansion_data.clone();
            self.cx.force_mode = force;

            // FIXME(jseyfried): Refactor out the following logic
            let fragment_kind = invoc.fragment_kind;
            let (expanded_fragment, new_invocations) = match res {
                InvocationRes::Single(ext) => match self.expand_invoc(invoc, &ext.kind) {
                    ExpandResult::Ready(fragment) => self.collect_invocations(fragment, &[]),
                    ExpandResult::Retry(invoc) => {
                        if force {
                            self.cx.span_bug(
                                invoc.span(),
                                "expansion entered force mode but is still stuck",
                            );
                        } else {
                            // Cannot expand, will retry this invocation later.
                            undetermined_invocations
                                .push((invoc, Some(InvocationRes::Single(ext))));
                            continue;
                        }
                    }
                },
                InvocationRes::DeriveContainer(_exts) => {
                    // FIXME: Consider using the derive resolutions (`_exts`) immediately,
                    // instead of enqueuing the derives to be resolved again later.
                    let (derives, mut item) = match invoc.kind {
                        InvocationKind::DeriveContainer { derives, item } => (derives, item),
                        _ => unreachable!(),
                    };
                    let (item, derive_placeholders) = if !item.derive_allowed() {
                        self.error_derive_forbidden_on_non_adt(&derives, &item);
                        item.visit_attrs(|attrs| attrs.retain(|a| !a.has_name(sym::derive)));
                        (item, Vec::new())
                    } else {
                        let mut visitor = StripUnconfigured {
                            sess: self.cx.sess,
                            features: self.cx.ecfg.features,
                            modified: false,
                        };
                        let mut item = visitor.fully_configure(item);
                        item.visit_attrs(|attrs| attrs.retain(|a| !a.has_name(sym::derive)));
                        if visitor.modified && !derives.is_empty() {
                            // Erase the tokens if cfg-stripping modified the item
                            // This will cause us to synthesize fake tokens
                            // when `nt_to_tokenstream` is called on this item.
                            match &mut item {
                                Annotatable::Item(item) => item.tokens = None,
                                Annotatable::Stmt(stmt) => {
                                    if let StmtKind::Item(item) = &mut stmt.kind {
                                        item.tokens = None
                                    } else {
                                        panic!("Unexpected stmt {:?}", stmt);
                                    }
                                }
                                _ => panic!("Unexpected annotatable {:?}", item),
                            }
                        }

                        invocations.reserve(derives.len());
                        let derive_placeholders = derives
                            .into_iter()
                            .map(|path| {
                                let expn_id = ExpnId::fresh(None);
                                invocations.push((
                                    Invocation {
                                        kind: InvocationKind::Derive { path, item: item.clone() },
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
                            .collect::<Vec<_>>();
                        (item, derive_placeholders)
                    };

                    let fragment = fragment_kind.expect_from_annotatables(::std::iter::once(item));
                    self.collect_invocations(fragment, &derive_placeholders)
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
        let attr = self.cx.sess.find_by_name(item.attrs(), sym::derive);
        let span = attr.map_or(item.span(), |attr| attr.span);
        let mut err = struct_span_err!(
            self.cx.sess,
            span,
            E0774,
            "`derive` may only be applied to structs, enums and unions",
        );
        if let Some(ast::Attribute { style: ast::AttrStyle::Inner, .. }) = attr {
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
    ) -> (AstFragment, Vec<(Invocation, Option<InvocationRes>)>) {
        // Resolve `$crate`s in the fragment for pretty-printing.
        self.cx.resolver.resolve_dollar_crates();

        let invocations = {
            let mut collector = InvocationCollector {
                cfg: StripUnconfigured {
                    sess: &self.cx.sess,
                    features: self.cx.ecfg.features,
                    modified: false,
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
            InvocationKind::Attr { attr, mut item, derives, after_derive } => match ext {
                SyntaxExtensionKind::Attr(expander) => {
                    self.gate_proc_macro_input(&item);
                    self.gate_proc_macro_attr_item(span, &item);
                    let tokens = match attr.style {
                        AttrStyle::Outer => item.into_tokens(&self.cx.sess.parse_sess),
                        // FIXME: Properly collect tokens for inner attributes
                        AttrStyle::Inner => rustc_parse::fake_token_stream(
                            &self.cx.sess.parse_sess,
                            &item.into_nonterminal(),
                            span,
                        ),
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
                                        kind: InvocationKind::Attr {
                                            attr,
                                            item,
                                            derives,
                                            after_derive,
                                        },
                                        ..invoc
                                    });
                                }
                            };
                            fragment_kind.expect_from_annotatables(items)
                        }
                        Err(mut err) => {
                            err.emit();
                            fragment_kind.dummy(span)
                        }
                    }
                }
                SyntaxExtensionKind::NonMacroAttr { mark_used } => {
                    self.cx.sess.mark_attr_known(&attr);
                    if *mark_used {
                        self.cx.sess.mark_attr_used(&attr);
                    }
                    item.visit_attrs(|attrs| attrs.push(attr));
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
            InvocationKind::DeriveContainer { .. } => unreachable!(),
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
            while let Some(item) = this.parse_trait_item()? {
                items.extend(item);
            }
            AstFragment::TraitItems(items)
        }
        AstFragmentKind::ImplItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_impl_item()? {
                items.extend(item);
            }
            AstFragment::ImplItems(items)
        }
        AstFragmentKind::ForeignItems => {
            let mut items = SmallVec::new();
            while let Some(item) = this.parse_foreign_item()? {
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
    invocations: Vec<(Invocation, Option<InvocationRes>)>,
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
                    self.cx.sess.parse_sess.edition,
                    None,
                )
            }),
            _ => None,
        };
        let expn_id = ExpnId::fresh(expn_data);
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
        (attr, derives, after_derive): (Option<ast::Attribute>, Vec<Path>, bool),
        item: Annotatable,
        kind: AstFragmentKind,
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
        attrs
            .iter()
            .position(|a| {
                if a.has_name(sym::derive) {
                    *after_derive = true;
                }
                !self.cx.sess.is_attr_known(a) && !is_builtin_attr(a)
            })
            .map(|i| attrs.remove(i))
    }

    /// If `item` is an attr invocation, remove and return the macro attribute and derive traits.
    fn take_first_attr(
        &mut self,
        item: &mut impl HasAttrs,
    ) -> Option<(Option<ast::Attribute>, Vec<Path>, /* after_derive */ bool)> {
        let (mut attr, mut traits, mut after_derive) = (None, Vec::new(), false);

        item.visit_attrs(|mut attrs| {
            attr = self.find_attr_invoc(&mut attrs, &mut after_derive);
            traits = collect_derives(&mut self.cx, &mut attrs);
        });

        if attr.is_some() || !traits.is_empty() { Some((attr, traits, after_derive)) } else { None }
    }

    /// Alternative to `take_first_attr()` that ignores `#[derive]` so invocations fallthrough
    /// to the unused-attributes lint (making it an error on statements and expressions
    /// is a breaking change)
    fn take_first_attr_no_derive(
        &mut self,
        nonitem: &mut impl HasAttrs,
    ) -> Option<(Option<ast::Attribute>, Vec<Path>, /* after_derive */ bool)> {
        let (mut attr, mut after_derive) = (None, false);

        nonitem.visit_attrs(|mut attrs| {
            attr = self.find_attr_invoc(&mut attrs, &mut after_derive);
        });

        attr.map(|attr| (Some(attr), Vec::new(), after_derive))
    }

    fn configure<T: HasAttrs>(&mut self, node: T) -> Option<T> {
        self.cfg.configure(node)
    }

    // Detect use of feature-gated or invalid attributes on macro invocations
    // since they will not be detected after macro expansion.
    fn check_attributes(&mut self, attrs: &[ast::Attribute]) {
        let features = self.cx.ecfg.features.unwrap();
        for attr in attrs.iter() {
            rustc_ast_passes::feature_gate::check_attribute(attr, self.cx.sess, features);
            validate_attr::check_meta(&self.cx.sess.parse_sess, attr);

            // macros are expanded before any lint passes so this warning has to be hardcoded
            if attr.has_name(sym::derive) {
                self.cx
                    .parse_sess()
                    .span_diagnostic
                    .struct_span_warn(attr.span, "`#[derive]` does nothing on macro invocations")
                    .note("this may become a hard error in a future release")
                    .emit();
            }

            if attr.doc_str().is_some() {
                self.cx.sess.parse_sess.buffer_lint_with_diagnostic(
                    &UNUSED_DOC_COMMENTS,
                    attr.span,
                    ast::CRATE_NODE_ID,
                    "unused doc comment",
                    BuiltinLintDiagnostics::UnusedDocComment(attr.span),
                );
            }
        }
    }
}

impl<'a, 'b> MutVisitor for InvocationCollector<'a, 'b> {
    fn visit_expr(&mut self, expr: &mut P<ast::Expr>) {
        self.cfg.configure_expr(expr);
        visit_clobber(expr.deref_mut(), |mut expr| {
            self.cfg.configure_expr_kind(&mut expr.kind);

            if let Some(attr) = self.take_first_attr_no_derive(&mut expr) {
                // Collect the invoc regardless of whether or not attributes are permitted here
                // expansion will eat the attribute so it won't error later.
                if let Some(attr) = attr.0.as_ref() {
                    self.cfg.maybe_emit_expr_attr_err(attr)
                }

                // AstFragmentKind::Expr requires the macro to emit an expression.
                return self
                    .collect_attr(attr, Annotatable::Expr(P(expr)), AstFragmentKind::Expr)
                    .make_expr()
                    .into_inner();
            }

            if let ast::ExprKind::MacCall(mac) = expr.kind {
                self.check_attributes(&expr.attrs);
                self.collect_bang(mac, expr.span, AstFragmentKind::Expr).make_expr().into_inner()
            } else {
                ensure_sufficient_stack(|| noop_visit_expr(&mut expr, self));
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

        noop_flat_map_arm(arm, self)
    }

    fn flat_map_field(&mut self, field: ast::Field) -> SmallVec<[ast::Field; 1]> {
        let mut field = configure!(self, field);

        if let Some(attr) = self.take_first_attr(&mut field) {
            return self
                .collect_attr(attr, Annotatable::Field(field), AstFragmentKind::Fields)
                .make_fields();
        }

        noop_flat_map_field(field, self)
    }

    fn flat_map_field_pattern(&mut self, fp: ast::FieldPat) -> SmallVec<[ast::FieldPat; 1]> {
        let mut fp = configure!(self, fp);

        if let Some(attr) = self.take_first_attr(&mut fp) {
            return self
                .collect_attr(attr, Annotatable::FieldPat(fp), AstFragmentKind::FieldPats)
                .make_field_patterns();
        }

        noop_flat_map_field_pattern(fp, self)
    }

    fn flat_map_param(&mut self, p: ast::Param) -> SmallVec<[ast::Param; 1]> {
        let mut p = configure!(self, p);

        if let Some(attr) = self.take_first_attr(&mut p) {
            return self
                .collect_attr(attr, Annotatable::Param(p), AstFragmentKind::Params)
                .make_params();
        }

        noop_flat_map_param(p, self)
    }

    fn flat_map_struct_field(&mut self, sf: ast::StructField) -> SmallVec<[ast::StructField; 1]> {
        let mut sf = configure!(self, sf);

        if let Some(attr) = self.take_first_attr(&mut sf) {
            return self
                .collect_attr(attr, Annotatable::StructField(sf), AstFragmentKind::StructFields)
                .make_struct_fields();
        }

        noop_flat_map_struct_field(sf, self)
    }

    fn flat_map_variant(&mut self, variant: ast::Variant) -> SmallVec<[ast::Variant; 1]> {
        let mut variant = configure!(self, variant);

        if let Some(attr) = self.take_first_attr(&mut variant) {
            return self
                .collect_attr(attr, Annotatable::Variant(variant), AstFragmentKind::Variants)
                .make_variants();
        }

        noop_flat_map_variant(variant, self)
    }

    fn filter_map_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        let expr = configure!(self, expr);
        expr.filter_map(|mut expr| {
            self.cfg.configure_expr_kind(&mut expr.kind);

            if let Some(attr) = self.take_first_attr_no_derive(&mut expr) {
                if let Some(attr) = attr.0.as_ref() {
                    self.cfg.maybe_emit_expr_attr_err(attr)
                }

                return self
                    .collect_attr(attr, Annotatable::Expr(P(expr)), AstFragmentKind::OptExpr)
                    .make_opt_expr()
                    .map(|expr| expr.into_inner());
            }

            if let ast::ExprKind::MacCall(mac) = expr.kind {
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

        // we'll expand attributes on expressions separately
        if !stmt.is_expr() {
            let attr = if stmt.is_item() {
                self.take_first_attr(&mut stmt)
            } else {
                // Ignore derives on non-item statements for backwards compatibility.
                // This will result in a unused attribute warning
                self.take_first_attr_no_derive(&mut stmt)
            };

            if let Some(attr) = attr {
                return self
                    .collect_attr(attr, Annotatable::Stmt(P(stmt)), AstFragmentKind::Stmts)
                    .make_stmts();
            }
        }

        if let StmtKind::MacCall(mac) = stmt.kind {
            let MacCallStmt { mac, style, attrs, tokens: _ } = mac.into_inner();
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

        if let Some(attr) = self.take_first_attr(&mut item) {
            return self
                .collect_attr(attr, Annotatable::Item(item), AstFragmentKind::Items)
                .make_items();
        }

        let mut attrs = mem::take(&mut item.attrs); // We do this to please borrowck.
        let ident = item.ident;
        let span = item.span;

        match item.kind {
            ast::ItemKind::MacCall(..) => {
                item.attrs = attrs;
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.kind {
                    ItemKind::MacCall(mac) => {
                        self.collect_bang(mac, span, AstFragmentKind::Items).make_items()
                    }
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(ref mut old_mod @ ast::Mod { .. }) if ident != Ident::invalid() => {
                let sess = &self.cx.sess.parse_sess;
                let orig_ownership = self.cx.current_expansion.directory_ownership;
                let mut module = (*self.cx.current_expansion.module).clone();

                let pushed = &mut false; // Record `parse_external_mod` pushing so we can pop.
                let dir = Directory { ownership: orig_ownership, path: module.directory };
                let Directory { ownership, path } = if old_mod.inline {
                    // Inline `mod foo { ... }`, but we still need to push directories.
                    item.attrs = attrs;
                    push_directory(&self.cx.sess, ident, &item.attrs, dir)
                } else {
                    // We have an outline `mod foo;` so we need to parse the file.
                    let (new_mod, dir) = parse_external_mod(
                        &self.cx.sess,
                        ident,
                        span,
                        old_mod.unsafety,
                        dir,
                        &mut attrs,
                        pushed,
                    );

                    let krate = ast::Crate {
                        span: new_mod.inner,
                        module: new_mod,
                        attrs,
                        proc_macros: vec![],
                    };
                    if let Some(extern_mod_loaded) = self.cx.extern_mod_loaded {
                        extern_mod_loaded(&krate);
                    }

                    *old_mod = krate.module;
                    item.attrs = krate.attrs;
                    // File can have inline attributes, e.g., `#![cfg(...)]` & co. => Reconfigure.
                    item = match self.configure(item) {
                        Some(node) => node,
                        None => {
                            if *pushed {
                                sess.included_mod_stack.borrow_mut().pop();
                            }
                            return Default::default();
                        }
                    };
                    dir
                };

                // Set the module info before we flat map.
                self.cx.current_expansion.directory_ownership = ownership;
                module.directory = path;
                module.mod_path.push(ident);
                let orig_module =
                    mem::replace(&mut self.cx.current_expansion.module, Rc::new(module));

                let result = noop_flat_map_item(item, self);

                // Restore the module info.
                self.cx.current_expansion.module = orig_module;
                self.cx.current_expansion.directory_ownership = orig_ownership;
                if *pushed {
                    sess.included_mod_stack.borrow_mut().pop();
                }
                result
            }
            _ => {
                item.attrs = attrs;
                noop_flat_map_item(item, self)
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
            ast::AssocItemKind::MacCall(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.kind {
                    ast::AssocItemKind::MacCall(mac) => self
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

        if let Some(attr) = self.take_first_attr(&mut item) {
            return self
                .collect_attr(attr, Annotatable::ImplItem(item), AstFragmentKind::ImplItems)
                .make_impl_items();
        }

        match item.kind {
            ast::AssocItemKind::MacCall(..) => {
                self.check_attributes(&item.attrs);
                item.and_then(|item| match item.kind {
                    ast::AssocItemKind::MacCall(mac) => self
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

    fn visit_foreign_mod(&mut self, foreign_mod: &mut ast::ForeignMod) {
        self.cfg.configure_foreign_mod(foreign_mod);
        noop_visit_foreign_mod(foreign_mod, self);
    }

    fn flat_map_foreign_item(
        &mut self,
        mut foreign_item: P<ast::ForeignItem>,
    ) -> SmallVec<[P<ast::ForeignItem>; 1]> {
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
            ast::ForeignItemKind::MacCall(..) => {
                self.check_attributes(&foreign_item.attrs);
                foreign_item.and_then(|item| match item.kind {
                    ast::ForeignItemKind::MacCall(mac) => self
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

        if let Some(attr) = self.take_first_attr(&mut param) {
            return self
                .collect_attr(
                    attr,
                    Annotatable::GenericParam(param),
                    AstFragmentKind::GenericParams,
                )
                .make_generic_params();
        }

        noop_flat_map_generic_param(param, self)
    }

    fn visit_attribute(&mut self, at: &mut ast::Attribute) {
        // turn `#[doc(include="filename")]` attributes into `#[doc(include(file="filename",
        // contents="file contents")]` attributes
        if !self.cx.sess.check_name(at, sym::doc) {
            return noop_visit_attribute(at, self);
        }

        if let Some(list) = at.meta_item_list() {
            if !list.iter().any(|it| it.has_name(sym::include)) {
                return noop_visit_attribute(at, self);
            }

            let mut items = vec![];

            for mut it in list {
                if !it.has_name(sym::include) {
                    items.push({
                        noop_visit_meta_list_item(&mut it, self);
                        it
                    });
                    continue;
                }

                if let Some(file) = it.value_str() {
                    let err_count = self.cx.sess.parse_sess.span_diagnostic.err_count();
                    self.check_attributes(slice::from_ref(at));
                    if self.cx.sess.parse_sess.span_diagnostic.err_count() > err_count {
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
                            let lit_span = it.name_value_literal_span().unwrap();

                            if e.kind() == ErrorKind::InvalidData {
                                self.cx
                                    .struct_span_err(
                                        lit_span,
                                        &format!("{} wasn't a utf-8 file", filename.display()),
                                    )
                                    .span_label(lit_span, "contains invalid utf-8")
                                    .emit();
                            } else {
                                let mut err = self.cx.struct_span_err(
                                    lit_span,
                                    &format!("couldn't read {}: {}", filename.display(), e),
                                );
                                err.span_label(lit_span, "couldn't read file");

                                err.emit();
                            }
                        }
                    }
                } else {
                    let mut err = self
                        .cx
                        .struct_span_err(it.span(), "expected path to external documentation");

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
                kind: ast::AttrKind::Normal(
                    AttrItem { path: meta.path, args: meta.kind.mac_args(meta.span), tokens: None },
                    None,
                ),
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
    pub recursion_limit: Limit,
    pub trace_mac: bool,
    pub should_test: bool, // If false, strip `#[test]` nodes
    pub keep_macs: bool,
    pub span_debug: bool, // If true, use verbose debugging for `proc_macro::Span`
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
            keep_macs: false,
            span_debug: false,
            proc_macro_backtrace: false,
        }
    }

    fn proc_macro_hygiene(&self) -> bool {
        self.features.map_or(false, |features| features.proc_macro_hygiene)
    }
}
