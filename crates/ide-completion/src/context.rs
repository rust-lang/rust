//! See `CompletionContext` structure.

mod analysis;
#[cfg(test)]
mod tests;

use base_db::SourceDatabaseExt;
use hir::{
    HasAttrs, Local, Name, PathResolution, ScopeDef, Semantics, SemanticsScope, Type, TypeInfo,
};
use ide_db::{
    base_db::{FilePosition, SourceDatabase},
    famous_defs::FamousDefs,
    FxHashMap, FxHashSet, RootDatabase,
};
use syntax::{
    ast::{self, AttrKind, NameOrNameRef},
    AstNode,
    SyntaxKind::{self, *},
    SyntaxToken, TextRange, TextSize,
};
use text_edit::Indel;

use crate::CompletionConfig;

const COMPLETION_MARKER: &str = "intellijRulezz";

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PatternRefutability {
    Refutable,
    Irrefutable,
}

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
        in_block_expr: bool,
        in_loop_body: bool,
        after_if_expr: bool,
        /// Whether this expression is the direct condition of an if or while expression
        in_condition: bool,
        ref_expr_parent: Option<ast::RefExpr>,
        is_func_update: Option<ast::RecordExpr>,
        self_param: Option<hir::SelfParam>,
        innermost_ret_ty: Option<hir::Type>,
    },
    Type {
        location: TypeLocation,
    },
    Attr {
        kind: AttrKind,
        annotated_item_kind: Option<SyntaxKind>,
    },
    Derive {
        existing_derives: FxHashSet<hir::Macro>,
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum ItemListKind {
    SourceFile,
    Module,
    Impl,
    TraitImpl,
    Trait,
    ExternBlock,
}

#[derive(Debug)]
pub(super) enum Qualified {
    No,
    With {
        path: ast::Path,
        resolution: Option<PathResolution>,
        /// Whether this path consists solely of `super` segments
        is_super_chain: bool,
    },
    /// <_>::
    Infer,
    /// Whether the path is an absolute path
    Absolute,
}

/// The state of the pattern we are completing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PatternContext {
    pub(super) refutability: PatternRefutability,
    pub(super) param_ctx: Option<(ast::ParamList, ast::Param, ParamKind)>,
    pub(super) has_type_ascription: bool,
    pub(super) parent_pat: Option<ast::Pat>,
    pub(super) ref_token: Option<SyntaxToken>,
    pub(super) mut_token: Option<SyntaxToken>,
    /// The record pattern this name or ref is a field of
    pub(super) record_pat: Option<ast::RecordPat>,
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
    /// The record expression this nameref is a field of
    RecordExpr(ast::RecordExpr),
    Pattern(PatternContext),
}

/// The identifier we are currently completing.
#[derive(Debug)]
pub(super) enum IdentContext {
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

    /// The expected name of what we are completing.
    /// This is usually the parameter name of the function argument we are completing.
    pub(super) expected_name: Option<NameOrNameRef>,
    /// The expected type of what we are completing.
    pub(super) expected_type: Option<Type>,

    /// The parent impl of the cursor position if it exists.
    // FIXME: This probably doesn't belong here
    pub(super) impl_def: Option<ast::Impl>,
    /// Are we completing inside a let statement with a missing semicolon?
    // FIXME: This should be part of PathKind::Expr
    pub(super) incomplete_let: bool,

    // FIXME: This shouldn't exist
    pub(super) previous_token: Option<SyntaxToken>,

    // We might wanna split these out of CompletionContext
    pub(super) ident_ctx: IdentContext,
    pub(super) qualifier_ctx: QualifierCtx,

    pub(super) locals: FxHashMap<Name, Local>,
}

impl<'a> CompletionContext<'a> {
    /// The range of the identifier that is being completed.
    pub(crate) fn source_range(&self) -> TextRange {
        // check kind of macro-expanded token, but use range of original token
        let kind = self.token.kind();
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

    // FIXME: This shouldn't exist
    pub(crate) fn previous_token_is(&self, kind: SyntaxKind) -> bool {
        self.previous_token.as_ref().map_or(false, |tok| tok.kind() == kind)
    }

    pub(crate) fn famous_defs(&self) -> FamousDefs {
        FamousDefs(&self.sema, self.krate)
    }

    /// Checks if an item is visible and not `doc(hidden)` at the completion site.
    pub(crate) fn is_visible<I>(&self, item: &I) -> Visible
    where
        I: hir::HasVisibility + hir::HasAttrs + hir::HasCrate + Copy,
    {
        self.is_visible_impl(&item.visibility(self.db), &item.attrs(self.db), item.krate(self.db))
    }

    pub(crate) fn is_scope_def_hidden(&self, scope_def: ScopeDef) -> bool {
        if let (Some(attrs), Some(krate)) = (scope_def.attrs(self.db), scope_def.krate(self.db)) {
            return self.is_doc_hidden(&attrs, krate);
        }

        false
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

    /// A version of [`SemanticsScope::process_all_names`] that filters out `#[doc(hidden)]` items.
    pub(crate) fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let _p = profile::span("CompletionContext::process_all_names");
        self.scope.process_all_names(&mut |name, def| {
            if self.is_scope_def_hidden(def) {
                return;
            }

            f(name, def);
        });
    }

    pub(crate) fn process_all_names_raw(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let _p = profile::span("CompletionContext::process_all_names_raw");
        self.scope.process_all_names(&mut |name, def| f(name, def));
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
            let root_file = defining_crate.root_file(self.db);
            let source_root_id = self.db.file_source_root(root_file);
            let is_editable = !self.db.source_root(source_root_id).is_library;
            return if is_editable { Visible::Editable } else { Visible::No };
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
}

// CompletionContext construction
impl<'a> CompletionContext<'a> {
    pub(super) fn new(
        db: &'a RootDatabase,
        position @ FilePosition { file_id, offset }: FilePosition,
        config: &'a CompletionConfig,
    ) -> Option<CompletionContext<'a>> {
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
        let fake_ident_token =
            file_with_fake_ident.syntax().token_at_offset(offset).right_biased()?;

        let original_token = original_file.syntax().token_at_offset(offset).left_biased()?;
        let token = sema.descend_into_macros_single(original_token.clone());
        let scope = sema.scope_at_offset(&token.parent()?, offset)?;
        let krate = scope.krate();
        let module = scope.module();

        let mut locals = FxHashMap::default();
        scope.process_all_names(&mut |name, scope| {
            if let ScopeDef::Local(local) = scope {
                locals.insert(name, local);
            }
        });

        let mut ctx = CompletionContext {
            sema,
            scope,
            db,
            config,
            position,
            original_token,
            token,
            krate,
            module,
            expected_name: None,
            expected_type: None,
            impl_def: None,
            incomplete_let: false,
            previous_token: None,
            // dummy value, will be overwritten
            ident_ctx: IdentContext::UnexpandedAttrTT { fake_attribute_under_caret: None },
            qualifier_ctx: Default::default(),
            locals,
        };
        ctx.expand_and_fill(
            original_file.syntax().clone(),
            file_with_fake_ident.syntax().clone(),
            offset,
            fake_ident_token,
        )?;
        Some(ctx)
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
