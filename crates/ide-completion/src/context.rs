//! See [`CompletionContext`] structure.

mod analysis;
#[cfg(test)]
mod tests;

use std::iter;

use hir::{
    HasAttrs, Local, Name, PathResolution, ScopeDef, Semantics, SemanticsScope, Type, TypeInfo,
};
use ide_db::{
    base_db::{FilePosition, SourceDatabase},
    famous_defs::FamousDefs,
    helpers::is_editable_crate,
    FxHashMap, FxHashSet, RootDatabase,
};
use syntax::{
    ast::{self, AttrKind, NameOrNameRef},
    AstNode, SmolStr,
    SyntaxKind::{self, *},
    SyntaxToken, TextRange, TextSize, T,
};
use text_edit::Indel;

use crate::{
    context::analysis::{expand_and_analyze, AnalysisResult},
    CompletionConfig,
};

const COMPLETION_MARKER: &str = "intellijRulezz";

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PatternRefutability {
    Refutable,
    Irrefutable,
}

#[derive(Debug)]
pub(crate) enum Visible {
    Yes,
    Editable,
    No,
}

/// Existing qualifiers for the thing we are currently completing.
#[derive(Debug, Default)]
pub(crate) struct QualifierCtx {
    pub(crate) unsafe_tok: Option<SyntaxToken>,
    pub(crate) vis_node: Option<ast::Visibility>,
}

impl QualifierCtx {
    pub(crate) fn none(&self) -> bool {
        self.unsafe_tok.is_none() && self.vis_node.is_none()
    }
}

/// The state of the path we are currently completing.
#[derive(Debug)]
pub(crate) struct PathCompletionCtx {
    /// If this is a call with () already there (or {} in case of record patterns)
    pub(crate) has_call_parens: bool,
    /// If this has a macro call bang !
    pub(crate) has_macro_bang: bool,
    /// The qualifier of the current path.
    pub(crate) qualified: Qualified,
    /// The parent of the path we are completing.
    pub(crate) parent: Option<ast::Path>,
    #[allow(dead_code)]
    /// The path of which we are completing the segment
    pub(crate) path: ast::Path,
    /// The path of which we are completing the segment in the original file
    pub(crate) original_path: Option<ast::Path>,
    pub(crate) kind: PathKind,
    /// Whether the path segment has type args or not.
    pub(crate) has_type_args: bool,
    /// Whether the qualifier comes from a use tree parent or not
    pub(crate) use_tree_parent: bool,
}

impl PathCompletionCtx {
    pub(crate) fn is_trivial_path(&self) -> bool {
        matches!(
            self,
            PathCompletionCtx {
                has_call_parens: false,
                has_macro_bang: false,
                qualified: Qualified::No,
                parent: None,
                has_type_args: false,
                ..
            }
        )
    }
}

/// The kind of path we are completing right now.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PathKind {
    Expr {
        expr_ctx: PathExprCtx,
    },
    Type {
        location: TypeLocation,
    },
    Attr {
        attr_ctx: AttrCtx,
    },
    Derive {
        existing_derives: ExistingDerives,
    },
    /// Path in item position, that is inside an (Assoc)ItemList
    Item {
        kind: ItemListKind,
    },
    Pat {
        pat_ctx: PatternContext,
    },
    Vis {
        has_in_token: bool,
    },
    Use,
}

pub(crate) type ExistingDerives = FxHashSet<hir::Macro>;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AttrCtx {
    pub(crate) kind: AttrKind,
    pub(crate) annotated_item_kind: Option<SyntaxKind>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PathExprCtx {
    pub(crate) in_block_expr: bool,
    pub(crate) in_breakable: BreakableKind,
    pub(crate) after_if_expr: bool,
    /// Whether this expression is the direct condition of an if or while expression
    pub(crate) in_condition: bool,
    pub(crate) incomplete_let: bool,
    pub(crate) ref_expr_parent: Option<ast::RefExpr>,
    /// The surrounding RecordExpression we are completing a functional update
    pub(crate) is_func_update: Option<ast::RecordExpr>,
    pub(crate) self_param: Option<hir::SelfParam>,
    pub(crate) innermost_ret_ty: Option<hir::Type>,
    pub(crate) impl_: Option<ast::Impl>,
    /// Whether this expression occurs in match arm guard position: before the
    /// fat arrow token
    pub(crate) in_match_guard: bool,
}

/// Original file ast nodes
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypeLocation {
    TupleField,
    TypeAscription(TypeAscriptionTarget),
    /// Generic argument position e.g. `Foo<$0>`
    GenericArg {
        /// The generic argument list containing the generic arg
        args: Option<ast::GenericArgList>,
        /// `Some(trait_)` if `trait_` is being instantiated with `args`
        of_trait: Option<hir::Trait>,
        /// The generic parameter being filled in by the generic arg
        corresponding_param: Option<ast::GenericParam>,
    },
    /// Associated type equality constraint e.g. `Foo<Bar = $0>`
    AssocTypeEq,
    /// Associated constant equality constraint e.g. `Foo<X = $0>`
    AssocConstEq,
    TypeBound,
    ImplTarget,
    ImplTrait,
    Other,
}

impl TypeLocation {
    pub(crate) fn complete_lifetimes(&self) -> bool {
        matches!(
            self,
            TypeLocation::GenericArg {
                corresponding_param: Some(ast::GenericParam::LifetimeParam(_)),
                ..
            }
        )
    }

    pub(crate) fn complete_consts(&self) -> bool {
        matches!(
            self,
            TypeLocation::GenericArg {
                corresponding_param: Some(ast::GenericParam::ConstParam(_)),
                ..
            } | TypeLocation::AssocConstEq
        )
    }

    pub(crate) fn complete_types(&self) -> bool {
        match self {
            TypeLocation::GenericArg { corresponding_param: Some(param), .. } => {
                matches!(param, ast::GenericParam::TypeParam(_))
            }
            TypeLocation::AssocConstEq => false,
            TypeLocation::AssocTypeEq => true,
            _ => true,
        }
    }

    pub(crate) fn complete_self_type(&self) -> bool {
        self.complete_types() && !matches!(self, TypeLocation::ImplTarget | TypeLocation::ImplTrait)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypeAscriptionTarget {
    Let(Option<ast::Pat>),
    FnParam(Option<ast::Pat>),
    RetType(Option<ast::Expr>),
    Const(Option<ast::Expr>),
}

/// The kind of item list a [`PathKind::Item`] belongs to.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ItemListKind {
    SourceFile,
    Module,
    Impl,
    TraitImpl(Option<ast::Impl>),
    Trait,
    ExternBlock,
}

#[derive(Debug)]
pub(crate) enum Qualified {
    No,
    With {
        path: ast::Path,
        resolution: Option<PathResolution>,
        /// How many `super` segments are present in the path
        ///
        /// This would be None, if path is not solely made of
        /// `super` segments, e.g.
        ///
        /// ```rust
        ///   use super::foo;
        /// ```
        ///
        /// Otherwise it should be Some(count of `super`)
        super_chain_len: Option<usize>,
    },
    /// <_>::
    TypeAnchor {
        ty: Option<hir::Type>,
        trait_: Option<hir::Trait>,
    },
    /// Whether the path is an absolute path
    Absolute,
}

/// The state of the pattern we are completing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PatternContext {
    pub(crate) refutability: PatternRefutability,
    pub(crate) param_ctx: Option<ParamContext>,
    pub(crate) has_type_ascription: bool,
    pub(crate) parent_pat: Option<ast::Pat>,
    pub(crate) ref_token: Option<SyntaxToken>,
    pub(crate) mut_token: Option<SyntaxToken>,
    /// The record pattern this name or ref is a field of
    pub(crate) record_pat: Option<ast::RecordPat>,
    pub(crate) impl_: Option<ast::Impl>,
    /// List of missing variants in a match expr
    pub(crate) missing_variants: Vec<hir::Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParamContext {
    pub(crate) param_list: ast::ParamList,
    pub(crate) param: ast::Param,
    pub(crate) kind: ParamKind,
}

/// The state of the lifetime we are completing.
#[derive(Debug)]
pub(crate) struct LifetimeContext {
    pub(crate) lifetime: Option<ast::Lifetime>,
    pub(crate) kind: LifetimeKind,
}

/// The kind of lifetime we are completing.
#[derive(Debug)]
pub(crate) enum LifetimeKind {
    LifetimeParam { is_decl: bool, param: ast::LifetimeParam },
    Lifetime,
    LabelRef,
    LabelDef,
}

/// The state of the name we are completing.
#[derive(Debug)]
pub(crate) struct NameContext {
    #[allow(dead_code)]
    pub(crate) name: Option<ast::Name>,
    pub(crate) kind: NameKind,
}

/// The kind of the name we are completing.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum NameKind {
    Const,
    ConstParam,
    Enum,
    Function,
    IdentPat(PatternContext),
    MacroDef,
    MacroRules,
    /// Fake node
    Module(ast::Module),
    RecordField,
    Rename,
    SelfParam,
    Static,
    Struct,
    Trait,
    TypeAlias,
    TypeParam,
    Union,
    Variant,
}

/// The state of the NameRef we are completing.
#[derive(Debug)]
pub(crate) struct NameRefContext {
    /// NameRef syntax in the original file
    pub(crate) nameref: Option<ast::NameRef>,
    pub(crate) kind: NameRefKind,
}

/// The kind of the NameRef we are completing.
#[derive(Debug)]
pub(crate) enum NameRefKind {
    Path(PathCompletionCtx),
    DotAccess(DotAccess),
    /// Position where we are only interested in keyword completions
    Keyword(ast::Item),
    /// The record expression this nameref is a field of and whether a dot precedes the completion identifier.
    RecordExpr {
        dot_prefix: bool,
        expr: ast::RecordExpr,
    },
    Pattern(PatternContext),
    ExternCrate,
}

/// The identifier we are currently completing.
#[derive(Debug)]
pub(crate) enum CompletionAnalysis {
    Name(NameContext),
    NameRef(NameRefContext),
    Lifetime(LifetimeContext),
    /// The string the cursor is currently inside
    String {
        /// original token
        original: ast::String,
        /// fake token
        expanded: Option<ast::String>,
    },
    /// Set if we are currently completing in an unexpanded attribute, this usually implies a builtin attribute like `allow($0)`
    UnexpandedAttrTT {
        colon_prefix: bool,
        fake_attribute_under_caret: Option<ast::Attr>,
        extern_crate: Option<ast::ExternCrate>,
    },
}

/// Information about the field or method access we are completing.
#[derive(Debug)]
pub(crate) struct DotAccess {
    pub(crate) receiver: Option<ast::Expr>,
    pub(crate) receiver_ty: Option<TypeInfo>,
    pub(crate) kind: DotAccessKind,
    pub(crate) ctx: DotAccessExprCtx,
}

#[derive(Debug)]
pub(crate) enum DotAccessKind {
    Field {
        /// True if the receiver is an integer and there is no ident in the original file after it yet
        /// like `0.$0`
        receiver_is_ambiguous_float_literal: bool,
    },
    Method {
        has_parens: bool,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DotAccessExprCtx {
    pub(crate) in_block_expr: bool,
    pub(crate) in_breakable: BreakableKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum BreakableKind {
    None,
    Loop,
    For,
    While,
    Block,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParamKind {
    Function(ast::Fn),
    Closure(ast::ClosureExpr),
}

/// `CompletionContext` is created early during completion to figure out, where
/// exactly is the cursor, syntax-wise.
#[derive(Debug)]
pub(crate) struct CompletionContext<'a> {
    pub(crate) sema: Semantics<'a, RootDatabase>,
    pub(crate) scope: SemanticsScope<'a>,
    pub(crate) db: &'a RootDatabase,
    pub(crate) config: &'a CompletionConfig,
    pub(crate) position: FilePosition,

    /// The token before the cursor, in the original file.
    pub(crate) original_token: SyntaxToken,
    /// The token before the cursor, in the macro-expanded file.
    pub(crate) token: SyntaxToken,
    /// The crate of the current file.
    pub(crate) krate: hir::Crate,
    /// The module of the `scope`.
    pub(crate) module: hir::Module,
    /// Whether nightly toolchain is used. Cached since this is looked up a lot.
    is_nightly: bool,

    /// The expected name of what we are completing.
    /// This is usually the parameter name of the function argument we are completing.
    pub(crate) expected_name: Option<NameOrNameRef>,
    /// The expected type of what we are completing.
    pub(crate) expected_type: Option<Type>,

    pub(crate) qualifier_ctx: QualifierCtx,

    pub(crate) locals: FxHashMap<Name, Local>,

    /// The module depth of the current module of the cursor position.
    /// - crate-root
    ///  - mod foo
    ///   - mod bar
    /// Here depth will be 2
    pub(crate) depth_from_crate_root: usize,
}

impl CompletionContext<'_> {
    /// The range of the identifier that is being completed.
    pub(crate) fn source_range(&self) -> TextRange {
        let kind = self.original_token.kind();
        match kind {
            CHAR => {
                // assume we are completing a lifetime but the user has only typed the '
                cov_mark::hit!(completes_if_lifetime_without_idents);
                TextRange::at(self.original_token.text_range().start(), TextSize::from(1))
            }
            IDENT | LIFETIME_IDENT | UNDERSCORE => self.original_token.text_range(),
            _ if kind.is_keyword() => self.original_token.text_range(),
            _ => TextRange::empty(self.position.offset),
        }
    }

    pub(crate) fn famous_defs(&self) -> FamousDefs<'_, '_> {
        FamousDefs(&self.sema, self.krate)
    }

    /// Checks if an item is visible and not `doc(hidden)` at the completion site.
    pub(crate) fn def_is_visible(&self, item: &ScopeDef) -> Visible {
        match item {
            ScopeDef::ModuleDef(def) => match def {
                hir::ModuleDef::Module(it) => self.is_visible(it),
                hir::ModuleDef::Function(it) => self.is_visible(it),
                hir::ModuleDef::Adt(it) => self.is_visible(it),
                hir::ModuleDef::Variant(it) => self.is_visible(it),
                hir::ModuleDef::Const(it) => self.is_visible(it),
                hir::ModuleDef::Static(it) => self.is_visible(it),
                hir::ModuleDef::Trait(it) => self.is_visible(it),
                hir::ModuleDef::TraitAlias(it) => self.is_visible(it),
                hir::ModuleDef::TypeAlias(it) => self.is_visible(it),
                hir::ModuleDef::Macro(it) => self.is_visible(it),
                hir::ModuleDef::BuiltinType(_) => Visible::Yes,
            },
            ScopeDef::GenericParam(_)
            | ScopeDef::ImplSelfType(_)
            | ScopeDef::AdtSelfType(_)
            | ScopeDef::Local(_)
            | ScopeDef::Label(_)
            | ScopeDef::Unknown => Visible::Yes,
        }
    }

    /// Checks if an item is visible and not `doc(hidden)` at the completion site.
    pub(crate) fn is_visible<I>(&self, item: &I) -> Visible
    where
        I: hir::HasVisibility + hir::HasAttrs + hir::HasCrate + Copy,
    {
        let vis = item.visibility(self.db);
        let attrs = item.attrs(self.db);
        self.is_visible_impl(&vis, &attrs, item.krate(self.db))
    }

    pub(crate) fn doc_aliases<I>(&self, item: &I) -> Vec<SmolStr>
    where
        I: hir::HasAttrs + Copy,
    {
        let attrs = item.attrs(self.db);
        attrs.doc_aliases().collect()
    }

    /// Check if an item is `#[doc(hidden)]`.
    pub(crate) fn is_item_hidden(&self, item: &hir::ItemInNs) -> bool {
        let attrs = item.attrs(self.db);
        let krate = item.krate(self.db);
        match (attrs, krate) {
            (Some(attrs), Some(krate)) => self.is_doc_hidden(&attrs, krate),
            _ => false,
        }
    }

    /// Checks whether this item should be listed in regards to stability. Returns `true` if we should.
    pub(crate) fn check_stability(&self, attrs: Option<&hir::Attrs>) -> bool {
        let Some(attrs) = attrs else {
            return true;
        };
        !attrs.is_unstable() || self.is_nightly
    }

    /// Whether the given trait is an operator trait or not.
    pub(crate) fn is_ops_trait(&self, trait_: hir::Trait) -> bool {
        match trait_.attrs(self.db).lang() {
            Some(lang) => OP_TRAIT_LANG_NAMES.contains(&lang.as_str()),
            None => false,
        }
    }

    /// Whether the given trait has `#[doc(notable_trait)]`
    pub(crate) fn is_doc_notable_trait(&self, trait_: hir::Trait) -> bool {
        trait_.attrs(self.db).has_doc_notable_trait()
    }

    /// Returns the traits in scope, with the [`Drop`] trait removed.
    pub(crate) fn traits_in_scope(&self) -> hir::VisibleTraits {
        let mut traits_in_scope = self.scope.visible_traits();
        if let Some(drop) = self.famous_defs().core_ops_Drop() {
            traits_in_scope.0.remove(&drop.into());
        }
        traits_in_scope
    }

    pub(crate) fn iterate_path_candidates(
        &self,
        ty: &hir::Type,
        mut cb: impl FnMut(hir::AssocItem),
    ) {
        let mut seen = FxHashSet::default();
        ty.iterate_path_candidates(
            self.db,
            &self.scope,
            &self.traits_in_scope(),
            Some(self.module),
            None,
            |item| {
                // We might iterate candidates of a trait multiple times here, so deduplicate
                // them.
                if seen.insert(item) {
                    cb(item)
                }
                None::<()>
            },
        );
    }

    /// A version of [`SemanticsScope::process_all_names`] that filters out `#[doc(hidden)]` items and
    /// passes all doc-aliases along, to funnel it into [`Completions::add_path_resolution`].
    pub(crate) fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef, Vec<SmolStr>)) {
        let _p =
            tracing::span!(tracing::Level::INFO, "CompletionContext::process_all_names").entered();
        self.scope.process_all_names(&mut |name, def| {
            if self.is_scope_def_hidden(def) {
                return;
            }
            let doc_aliases = self.doc_aliases_in_scope(def);
            f(name, def, doc_aliases);
        });
    }

    pub(crate) fn process_all_names_raw(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let _p = tracing::span!(tracing::Level::INFO, "CompletionContext::process_all_names_raw")
            .entered();
        self.scope.process_all_names(f);
    }

    fn is_scope_def_hidden(&self, scope_def: ScopeDef) -> bool {
        if let (Some(attrs), Some(krate)) = (scope_def.attrs(self.db), scope_def.krate(self.db)) {
            return self.is_doc_hidden(&attrs, krate);
        }

        false
    }

    fn is_visible_impl(
        &self,
        vis: &hir::Visibility,
        attrs: &hir::Attrs,
        defining_crate: hir::Crate,
    ) -> Visible {
        if !vis.is_visible_from(self.db, self.module.into()) {
            if !self.config.enable_private_editable {
                return Visible::No;
            }
            // If the definition location is editable, also show private items
            return if is_editable_crate(defining_crate, self.db) {
                Visible::Editable
            } else {
                Visible::No
            };
        }

        if self.is_doc_hidden(attrs, defining_crate) {
            Visible::No
        } else {
            Visible::Yes
        }
    }

    fn is_doc_hidden(&self, attrs: &hir::Attrs, defining_crate: hir::Crate) -> bool {
        // `doc(hidden)` items are only completed within the defining crate.
        self.krate != defining_crate && attrs.has_doc_hidden()
    }

    pub(crate) fn doc_aliases_in_scope(&self, scope_def: ScopeDef) -> Vec<SmolStr> {
        if let Some(attrs) = scope_def.attrs(self.db) {
            attrs.doc_aliases().collect()
        } else {
            vec![]
        }
    }
}

// CompletionContext construction
impl<'a> CompletionContext<'a> {
    pub(crate) fn new(
        db: &'a RootDatabase,
        position @ FilePosition { file_id, offset }: FilePosition,
        config: &'a CompletionConfig,
    ) -> Option<(CompletionContext<'a>, CompletionAnalysis)> {
        let _p = tracing::span!(tracing::Level::INFO, "CompletionContext::new").entered();
        let sema = Semantics::new(db);

        let original_file = sema.parse(file_id);

        // Insert a fake ident to get a valid parse tree. We will use this file
        // to determine context, though the original_file will be used for
        // actual completion.
        let file_with_fake_ident = {
            let parse = db.parse(file_id);
            let edit = Indel::insert(offset, COMPLETION_MARKER.to_owned());
            parse.reparse(&edit).tree()
        };

        // always pick the token to the immediate left of the cursor, as that is what we are actually
        // completing on
        let original_token = original_file.syntax().token_at_offset(offset).left_biased()?;

        // try to skip completions on path with invalid colons
        // this approach works in normal path and inside token tree
        if original_token.kind() == T![:] {
            // return if no prev token before colon
            let prev_token = original_token.prev_token()?;

            // only has a single colon
            if prev_token.kind() != T![:] {
                return None;
            }

            // has 3 colon or 2 coloncolon in a row
            // special casing this as per discussion in https://github.com/rust-lang/rust-analyzer/pull/13611#discussion_r1031845205
            // and https://github.com/rust-lang/rust-analyzer/pull/13611#discussion_r1032812751
            if prev_token
                .prev_token()
                .map(|t| t.kind() == T![:] || t.kind() == T![::])
                .unwrap_or(false)
            {
                return None;
            }
        }

        let AnalysisResult {
            analysis,
            expected: (expected_type, expected_name),
            qualifier_ctx,
            token,
            offset,
        } = expand_and_analyze(
            &sema,
            original_file.syntax().clone(),
            file_with_fake_ident.syntax().clone(),
            offset,
            &original_token,
        )?;

        // adjust for macro input, this still fails if there is no token written yet
        let scope = sema.scope_at_offset(&token.parent()?, offset)?;

        let krate = scope.krate();
        let module = scope.module();

        let toolchain = db.crate_graph()[krate.into()].channel();
        // `toolchain == None` means we're in some detached files. Since we have no information on
        // the toolchain being used, let's just allow unstable items to be listed.
        let is_nightly = matches!(toolchain, Some(base_db::ReleaseChannel::Nightly) | None);

        let mut locals = FxHashMap::default();
        scope.process_all_names(&mut |name, scope| {
            if let ScopeDef::Local(local) = scope {
                locals.insert(name, local);
            }
        });

        let depth_from_crate_root = iter::successors(module.parent(db), |m| m.parent(db)).count();

        let ctx = CompletionContext {
            sema,
            scope,
            db,
            config,
            position,
            original_token,
            token,
            krate,
            module,
            is_nightly,
            expected_name,
            expected_type,
            qualifier_ctx,
            locals,
            depth_from_crate_root,
        };
        Some((ctx, analysis))
    }
}

const OP_TRAIT_LANG_NAMES: &[&str] = &[
    "add_assign",
    "add",
    "bitand_assign",
    "bitand",
    "bitor_assign",
    "bitor",
    "bitxor_assign",
    "bitxor",
    "deref_mut",
    "deref",
    "div_assign",
    "div",
    "eq",
    "fn_mut",
    "fn_once",
    "fn",
    "index_mut",
    "index",
    "mul_assign",
    "mul",
    "neg",
    "not",
    "partial_ord",
    "rem_assign",
    "rem",
    "shl_assign",
    "shl",
    "shr_assign",
    "shr",
    "sub",
];
