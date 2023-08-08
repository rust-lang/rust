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
pub(super) struct QualifierCtx {
    pub(super) unsafe_tok: Option<SyntaxToken>,
    pub(super) vis_node: Option<ast::Visibility>,
}

impl QualifierCtx {
    pub(super) fn none(&self) -> bool {
        self.unsafe_tok.is_none() && self.vis_node.is_none()
    }
}

/// The state of the path we are currently completing.
#[derive(Debug)]
pub(crate) struct PathCompletionCtx {
    /// If this is a call with () already there (or {} in case of record patterns)
    pub(super) has_call_parens: bool,
    /// If this has a macro call bang !
    pub(super) has_macro_bang: bool,
    /// The qualifier of the current path.
    pub(super) qualified: Qualified,
    /// The parent of the path we are completing.
    pub(super) parent: Option<ast::Path>,
    #[allow(dead_code)]
    /// The path of which we are completing the segment
    pub(super) path: ast::Path,
    /// The path of which we are completing the segment in the original file
    pub(crate) original_path: Option<ast::Path>,
    pub(super) kind: PathKind,
    /// Whether the path segment has type args or not.
    pub(super) has_type_args: bool,
    /// Whether the qualifier comes from a use tree parent or not
    pub(crate) use_tree_parent: bool,
}

impl PathCompletionCtx {
    pub(super) fn is_trivial_path(&self) -> bool {
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
pub(super) enum PathKind {
    Expr {
        expr_ctx: ExprCtx,
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
pub(crate) struct ExprCtx {
    pub(crate) in_block_expr: bool,
    pub(crate) in_loop_body: bool,
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
    GenericArgList(Option<ast::GenericArgList>),
    TypeBound,
    ImplTarget,
    ImplTrait,
    Other,
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
pub(super) enum ItemListKind {
    SourceFile,
    Module,
    Impl,
    TraitImpl(Option<ast::Impl>),
    Trait,
    ExternBlock,
}

#[derive(Debug)]
pub(super) enum Qualified {
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
pub(super) struct PatternContext {
    pub(super) refutability: PatternRefutability,
    pub(super) param_ctx: Option<ParamContext>,
    pub(super) has_type_ascription: bool,
    pub(super) parent_pat: Option<ast::Pat>,
    pub(super) ref_token: Option<SyntaxToken>,
    pub(super) mut_token: Option<SyntaxToken>,
    /// The record pattern this name or ref is a field of
    pub(super) record_pat: Option<ast::RecordPat>,
    pub(super) impl_: Option<ast::Impl>,
    /// List of missing variants in a match expr
    pub(super) missing_variants: Vec<hir::Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ParamContext {
    pub(super) param_list: ast::ParamList,
    pub(super) param: ast::Param,
    pub(super) kind: ParamKind,
}

/// The state of the lifetime we are completing.
#[derive(Debug)]
pub(super) struct LifetimeContext {
    pub(super) lifetime: Option<ast::Lifetime>,
    pub(super) kind: LifetimeKind,
}

/// The kind of lifetime we are completing.
#[derive(Debug)]
pub(super) enum LifetimeKind {
    LifetimeParam { is_decl: bool, param: ast::LifetimeParam },
    Lifetime,
    LabelRef,
    LabelDef,
}

/// The state of the name we are completing.
#[derive(Debug)]
pub(super) struct NameContext {
    #[allow(dead_code)]
    pub(super) name: Option<ast::Name>,
    pub(super) kind: NameKind,
}

/// The kind of the name we are completing.
#[derive(Debug)]
#[allow(dead_code)]
pub(super) enum NameKind {
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
pub(super) struct NameRefContext {
    /// NameRef syntax in the original file
    pub(super) nameref: Option<ast::NameRef>,
    pub(super) kind: NameRefKind,
}

/// The kind of the NameRef we are completing.
#[derive(Debug)]
pub(super) enum NameRefKind {
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
}

/// The identifier we are currently completing.
#[derive(Debug)]
pub(super) enum CompletionAnalysis {
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
    },
}

/// Information about the field or method access we are completing.
#[derive(Debug)]
pub(super) struct DotAccess {
    pub(super) receiver: Option<ast::Expr>,
    pub(super) receiver_ty: Option<TypeInfo>,
    pub(super) kind: DotAccessKind,
}

#[derive(Debug)]
pub(super) enum DotAccessKind {
    Field {
        /// True if the receiver is an integer and there is no ident in the original file after it yet
        /// like `0.$0`
        receiver_is_ambiguous_float_literal: bool,
    },
    Method {
        has_parens: bool,
    },
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
    pub(super) sema: Semantics<'a, RootDatabase>,
    pub(super) scope: SemanticsScope<'a>,
    pub(super) db: &'a RootDatabase,
    pub(super) config: &'a CompletionConfig,
    pub(super) position: FilePosition,

    /// The token before the cursor, in the original file.
    pub(super) original_token: SyntaxToken,
    /// The token before the cursor, in the macro-expanded file.
    pub(super) token: SyntaxToken,
    /// The crate of the current file.
    pub(super) krate: hir::Crate,
    /// The module of the `scope`.
    pub(super) module: hir::Module,
    /// Whether nightly toolchain is used. Cached since this is looked up a lot.
    is_nightly: bool,

    /// The expected name of what we are completing.
    /// This is usually the parameter name of the function argument we are completing.
    pub(super) expected_name: Option<NameOrNameRef>,
    /// The expected type of what we are completing.
    pub(super) expected_type: Option<Type>,

    pub(super) qualifier_ctx: QualifierCtx,

    pub(super) locals: FxHashMap<Name, Local>,

    /// The module depth of the current module of the cursor position.
    /// - crate-root
    ///  - mod foo
    ///   - mod bar
    /// Here depth will be 2
    pub(super) depth_from_crate_root: usize,
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
        let _p = profile::span("CompletionContext::process_all_names");
        self.scope.process_all_names(&mut |name, def| {
            if self.is_scope_def_hidden(def) {
                return;
            }
            let doc_aliases = self.doc_aliases_in_scope(def);
            f(name, def, doc_aliases);
        });
    }

    pub(crate) fn process_all_names_raw(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let _p = profile::span("CompletionContext::process_all_names_raw");
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
    pub(super) fn new(
        db: &'a RootDatabase,
        position @ FilePosition { file_id, offset }: FilePosition,
        config: &'a CompletionConfig,
    ) -> Option<(CompletionContext<'a>, CompletionAnalysis)> {
        let _p = profile::span("CompletionContext::new");
        let sema = Semantics::new(db);

        let original_file = sema.parse(file_id);

        // Insert a fake ident to get a valid parse tree. We will use this file
        // to determine context, though the original_file will be used for
        // actual completion.
        let file_with_fake_ident = {
            let parse = db.parse(file_id);
            let edit = Indel::insert(offset, COMPLETION_MARKER.to_string());
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

        let toolchain = db.crate_graph()[krate.into()].channel;
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
