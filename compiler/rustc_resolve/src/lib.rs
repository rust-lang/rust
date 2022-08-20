//! This crate is responsible for the part of name resolution that doesn't require type checker.
//!
//! Module structure of the crate is built here.
//! Paths in macros, imports, expressions, types, patterns are resolved here.
//! Label and lifetime names are resolved here as well.
//!
//! Type-relative name resolution (methods, fields, associated items) happens in `rustc_typeck`.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(never_type)]
#![recursion_limit = "256"]
#![allow(rustdoc::private_intra_doc_links)]
#![allow(rustc::potential_query_instability)]

#[macro_use]
extern crate tracing;

pub use rustc_hir::def::{Namespace, PerNS};

use rustc_arena::{DroplessArena, TypedArena};
use rustc_ast::node_id::NodeMap;
use rustc_ast::{self as ast, NodeId, CRATE_NODE_ID};
use rustc_ast::{AngleBracketedArg, Crate, Expr, ExprKind, GenericArg, GenericArgs, LitKind, Path};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_expand::base::{DeriveResolutions, SyntaxExtension, SyntaxExtensionKind};
use rustc_hir::def::Namespace::*;
use rustc_hir::def::{self, CtorOf, DefKind, LifetimeRes, PartialRes};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId};
use rustc_hir::def_id::{CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::definitions::{DefPathData, Definitions};
use rustc_hir::TraitCandidate;
use rustc_index::vec::IndexVec;
use rustc_metadata::creader::{CStore, CrateLoader};
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::span_bug;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, DefIdTree, MainDefinition, RegisteredTools, ResolverOutputs};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::cstore::{CrateStore, CrateStoreDyn, MetadataLoaderDyn};
use rustc_session::lint::LintBuffer;
use rustc_session::Session;
use rustc_span::hygiene::{ExpnId, LocalExpnId, MacroKind, SyntaxContext, Transparency};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

use smallvec::{smallvec, SmallVec};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::{cmp, fmt, ptr};
use tracing::debug;

use diagnostics::{ImportSuggestion, LabelSuggestion, Suggestion};
use imports::{Import, ImportKind, ImportResolver, NameResolution};
use late::{HasGenericParams, PathSource, PatternSource};
use macros::{MacroRulesBinding, MacroRulesScope, MacroRulesScopeRef};

use crate::access_levels::AccessLevelsVisitor;

type Res = def::Res<NodeId>;

mod access_levels;
mod build_reduced_graph;
mod check_unused;
mod def_collector;
mod diagnostics;
mod ident;
mod imports;
mod late;
mod macros;

enum Weak {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Determinacy {
    Determined,
    Undetermined,
}

impl Determinacy {
    fn determined(determined: bool) -> Determinacy {
        if determined { Determinacy::Determined } else { Determinacy::Undetermined }
    }
}

/// A specific scope in which a name can be looked up.
/// This enum is currently used only for early resolution (imports and macros),
/// but not for late resolution yet.
#[derive(Clone, Copy)]
enum Scope<'a> {
    DeriveHelpers(LocalExpnId),
    DeriveHelpersCompat,
    MacroRules(MacroRulesScopeRef<'a>),
    CrateRoot,
    // The node ID is for reporting the `PROC_MACRO_DERIVE_RESOLUTION_FALLBACK`
    // lint if it should be reported.
    Module(Module<'a>, Option<NodeId>),
    RegisteredAttrs,
    MacroUsePrelude,
    BuiltinAttrs,
    ExternPrelude,
    ToolPrelude,
    StdLibPrelude,
    BuiltinTypes,
}

/// Names from different contexts may want to visit different subsets of all specific scopes
/// with different restrictions when looking up the resolution.
/// This enum is currently used only for early resolution (imports and macros),
/// but not for late resolution yet.
#[derive(Clone, Copy)]
enum ScopeSet<'a> {
    /// All scopes with the given namespace.
    All(Namespace, /*is_import*/ bool),
    /// Crate root, then extern prelude (used for mixed 2015-2018 mode in macros).
    AbsolutePath(Namespace),
    /// All scopes with macro namespace and the given macro kind restriction.
    Macro(MacroKind),
    /// All scopes with the given namespace, used for partially performing late resolution.
    /// The node id enables lints and is used for reporting them.
    Late(Namespace, Module<'a>, Option<NodeId>),
}

/// Everything you need to know about a name's location to resolve it.
/// Serves as a starting point for the scope visitor.
/// This struct is currently used only for early resolution (imports and macros),
/// but not for late resolution yet.
#[derive(Clone, Copy, Debug)]
pub struct ParentScope<'a> {
    pub module: Module<'a>,
    expansion: LocalExpnId,
    pub macro_rules: MacroRulesScopeRef<'a>,
    derives: &'a [ast::Path],
}

impl<'a> ParentScope<'a> {
    /// Creates a parent scope with the passed argument used as the module scope component,
    /// and other scope components set to default empty values.
    pub fn module(module: Module<'a>, resolver: &Resolver<'a>) -> ParentScope<'a> {
        ParentScope {
            module,
            expansion: LocalExpnId::ROOT,
            macro_rules: resolver.arenas.alloc_macro_rules_scope(MacroRulesScope::Empty),
            derives: &[],
        }
    }
}

#[derive(Copy, Debug, Clone)]
enum ImplTraitContext {
    Existential,
    Universal(LocalDefId),
}

#[derive(Eq)]
struct BindingError {
    name: Symbol,
    origin: BTreeSet<Span>,
    target: BTreeSet<Span>,
    could_be_path: bool,
}

impl PartialOrd for BindingError {
    fn partial_cmp(&self, other: &BindingError) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BindingError {
    fn eq(&self, other: &BindingError) -> bool {
        self.name == other.name
    }
}

impl Ord for BindingError {
    fn cmp(&self, other: &BindingError) -> cmp::Ordering {
        self.name.cmp(&other.name)
    }
}

enum ResolutionError<'a> {
    /// Error E0401: can't use type or const parameters from outer function.
    GenericParamsFromOuterFunction(Res, HasGenericParams),
    /// Error E0403: the name is already used for a type or const parameter in this generic
    /// parameter list.
    NameAlreadyUsedInParameterList(Symbol, Span),
    /// Error E0407: method is not a member of trait.
    MethodNotMemberOfTrait(Ident, String, Option<Symbol>),
    /// Error E0437: type is not a member of trait.
    TypeNotMemberOfTrait(Ident, String, Option<Symbol>),
    /// Error E0438: const is not a member of trait.
    ConstNotMemberOfTrait(Ident, String, Option<Symbol>),
    /// Error E0408: variable `{}` is not bound in all patterns.
    VariableNotBoundInPattern(BindingError, ParentScope<'a>),
    /// Error E0409: variable `{}` is bound in inconsistent ways within the same match arm.
    VariableBoundWithDifferentMode(Symbol, Span),
    /// Error E0415: identifier is bound more than once in this parameter list.
    IdentifierBoundMoreThanOnceInParameterList(Symbol),
    /// Error E0416: identifier is bound more than once in the same pattern.
    IdentifierBoundMoreThanOnceInSamePattern(Symbol),
    /// Error E0426: use of undeclared label.
    UndeclaredLabel { name: Symbol, suggestion: Option<LabelSuggestion> },
    /// Error E0429: `self` imports are only allowed within a `{ }` list.
    SelfImportsOnlyAllowedWithin { root: bool, span_with_rename: Span },
    /// Error E0430: `self` import can only appear once in the list.
    SelfImportCanOnlyAppearOnceInTheList,
    /// Error E0431: `self` import can only appear in an import list with a non-empty prefix.
    SelfImportOnlyInImportListWithNonEmptyPrefix,
    /// Error E0433: failed to resolve.
    FailedToResolve { label: String, suggestion: Option<Suggestion> },
    /// Error E0434: can't capture dynamic environment in a fn item.
    CannotCaptureDynamicEnvironmentInFnItem,
    /// Error E0435: attempt to use a non-constant value in a constant.
    AttemptToUseNonConstantValueInConstant(
        Ident,
        /* suggestion */ &'static str,
        /* current */ &'static str,
    ),
    /// Error E0530: `X` bindings cannot shadow `Y`s.
    BindingShadowsSomethingUnacceptable {
        shadowing_binding: PatternSource,
        name: Symbol,
        participle: &'static str,
        article: &'static str,
        shadowed_binding: Res,
        shadowed_binding_span: Span,
    },
    /// Error E0128: generic parameters with a default cannot use forward-declared identifiers.
    ForwardDeclaredGenericParam,
    /// ERROR E0770: the type of const parameters must not depend on other generic parameters.
    ParamInTyOfConstParam(Symbol),
    /// generic parameters must not be used inside const evaluations.
    ///
    /// This error is only emitted when using `min_const_generics`.
    ParamInNonTrivialAnonConst { name: Symbol, is_type: bool },
    /// Error E0735: generic parameters with a default cannot use `Self`
    SelfInGenericParamDefault,
    /// Error E0767: use of unreachable label
    UnreachableLabel { name: Symbol, definition_span: Span, suggestion: Option<LabelSuggestion> },
    /// Error E0323, E0324, E0325: mismatch between trait item and impl item.
    TraitImplMismatch {
        name: Symbol,
        kind: &'static str,
        trait_path: String,
        trait_item_span: Span,
        code: rustc_errors::DiagnosticId,
    },
    /// Inline asm `sym` operand must refer to a `fn` or `static`.
    InvalidAsmSym,
}

enum VisResolutionError<'a> {
    Relative2018(Span, &'a ast::Path),
    AncestorOnly(Span),
    FailedToResolve(Span, String, Option<Suggestion>),
    ExpectedFound(Span, String, Res),
    Indeterminate(Span),
    ModuleOnly(Span),
}

/// A minimal representation of a path segment. We use this in resolve because we synthesize 'path
/// segments' which don't have the rest of an AST or HIR `PathSegment`.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    ident: Ident,
    id: Option<NodeId>,
    /// Signals whether this `PathSegment` has generic arguments. Used to avoid providing
    /// nonsensical suggestions.
    has_generic_args: bool,
    /// Signals whether this `PathSegment` has lifetime arguments.
    has_lifetime_args: bool,
    args_span: Span,
}

impl Segment {
    fn from_path(path: &Path) -> Vec<Segment> {
        path.segments.iter().map(|s| s.into()).collect()
    }

    fn from_ident(ident: Ident) -> Segment {
        Segment {
            ident,
            id: None,
            has_generic_args: false,
            has_lifetime_args: false,
            args_span: DUMMY_SP,
        }
    }

    fn from_ident_and_id(ident: Ident, id: NodeId) -> Segment {
        Segment {
            ident,
            id: Some(id),
            has_generic_args: false,
            has_lifetime_args: false,
            args_span: DUMMY_SP,
        }
    }

    fn names_to_string(segments: &[Segment]) -> String {
        names_to_string(&segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>())
    }
}

impl<'a> From<&'a ast::PathSegment> for Segment {
    fn from(seg: &'a ast::PathSegment) -> Segment {
        let has_generic_args = seg.args.is_some();
        let (args_span, has_lifetime_args) = if let Some(args) = seg.args.as_deref() {
            match args {
                GenericArgs::AngleBracketed(args) => {
                    let found_lifetimes = args
                        .args
                        .iter()
                        .any(|arg| matches!(arg, AngleBracketedArg::Arg(GenericArg::Lifetime(_))));
                    (args.span, found_lifetimes)
                }
                GenericArgs::Parenthesized(args) => (args.span, true),
            }
        } else {
            (DUMMY_SP, false)
        };
        Segment {
            ident: seg.ident,
            id: Some(seg.id),
            has_generic_args,
            has_lifetime_args,
            args_span,
        }
    }
}

/// An intermediate resolution result.
///
/// This refers to the thing referred by a name. The difference between `Res` and `Item` is that
/// items are visible in their whole block, while `Res`es only from the place they are defined
/// forward.
#[derive(Debug)]
enum LexicalScopeBinding<'a> {
    Item(&'a NameBinding<'a>),
    Res(Res),
}

impl<'a> LexicalScopeBinding<'a> {
    fn res(self) -> Res {
        match self {
            LexicalScopeBinding::Item(binding) => binding.res(),
            LexicalScopeBinding::Res(res) => res,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ModuleOrUniformRoot<'a> {
    /// Regular module.
    Module(Module<'a>),

    /// Virtual module that denotes resolution in crate root with fallback to extern prelude.
    CrateRootAndExternPrelude,

    /// Virtual module that denotes resolution in extern prelude.
    /// Used for paths starting with `::` on 2018 edition.
    ExternPrelude,

    /// Virtual module that denotes resolution in current scope.
    /// Used only for resolving single-segment imports. The reason it exists is that import paths
    /// are always split into two parts, the first of which should be some kind of module.
    CurrentScope,
}

impl ModuleOrUniformRoot<'_> {
    fn same_def(lhs: Self, rhs: Self) -> bool {
        match (lhs, rhs) {
            (ModuleOrUniformRoot::Module(lhs), ModuleOrUniformRoot::Module(rhs)) => {
                ptr::eq(lhs, rhs)
            }
            (
                ModuleOrUniformRoot::CrateRootAndExternPrelude,
                ModuleOrUniformRoot::CrateRootAndExternPrelude,
            )
            | (ModuleOrUniformRoot::ExternPrelude, ModuleOrUniformRoot::ExternPrelude)
            | (ModuleOrUniformRoot::CurrentScope, ModuleOrUniformRoot::CurrentScope) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
enum PathResult<'a> {
    Module(ModuleOrUniformRoot<'a>),
    NonModule(PartialRes),
    Indeterminate,
    Failed {
        span: Span,
        label: String,
        suggestion: Option<Suggestion>,
        is_error_from_last_segment: bool,
    },
}

impl<'a> PathResult<'a> {
    fn failed(
        span: Span,
        is_error_from_last_segment: bool,
        finalize: bool,
        label_and_suggestion: impl FnOnce() -> (String, Option<Suggestion>),
    ) -> PathResult<'a> {
        let (label, suggestion) =
            if finalize { label_and_suggestion() } else { (String::new(), None) };
        PathResult::Failed { span, label, suggestion, is_error_from_last_segment }
    }
}

#[derive(Debug)]
enum ModuleKind {
    /// An anonymous module; e.g., just a block.
    ///
    /// ```
    /// fn main() {
    ///     fn f() {} // (1)
    ///     { // This is an anonymous module
    ///         f(); // This resolves to (2) as we are inside the block.
    ///         fn f() {} // (2)
    ///     }
    ///     f(); // Resolves to (1)
    /// }
    /// ```
    Block,
    /// Any module with a name.
    ///
    /// This could be:
    ///
    /// * A normal module – either `mod from_file;` or `mod from_block { }` –
    ///   or the crate root (which is conceptually a top-level module).
    ///   Note that the crate root's [name][Self::name] will be [`kw::Empty`].
    /// * A trait or an enum (it implicitly contains associated types, methods and variant
    ///   constructors).
    Def(DefKind, DefId, Symbol),
}

impl ModuleKind {
    /// Get name of the module.
    pub fn name(&self) -> Option<Symbol> {
        match self {
            ModuleKind::Block => None,
            ModuleKind::Def(.., name) => Some(*name),
        }
    }
}

/// A key that identifies a binding in a given `Module`.
///
/// Multiple bindings in the same module can have the same key (in a valid
/// program) if all but one of them come from glob imports.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct BindingKey {
    /// The identifier for the binding, always the `normalize_to_macros_2_0` version of the
    /// identifier.
    ident: Ident,
    ns: Namespace,
    /// 0 if ident is not `_`, otherwise a value that's unique to the specific
    /// `_` in the expanded AST that introduced this binding.
    disambiguator: u32,
}

type Resolutions<'a> = RefCell<FxIndexMap<BindingKey, &'a RefCell<NameResolution<'a>>>>;

/// One node in the tree of modules.
///
/// Note that a "module" in resolve is broader than a `mod` that you declare in Rust code. It may be one of these:
///
/// * `mod`
/// * crate root (aka, top-level anonymous module)
/// * `enum`
/// * `trait`
/// * curly-braced block with statements
///
/// You can use [`ModuleData::kind`] to determine the kind of module this is.
pub struct ModuleData<'a> {
    /// The direct parent module (it may not be a `mod`, however).
    parent: Option<Module<'a>>,
    /// What kind of module this is, because this may not be a `mod`.
    kind: ModuleKind,

    /// Mapping between names and their (possibly in-progress) resolutions in this module.
    /// Resolutions in modules from other crates are not populated until accessed.
    lazy_resolutions: Resolutions<'a>,
    /// True if this is a module from other crate that needs to be populated on access.
    populate_on_access: Cell<bool>,

    /// Macro invocations that can expand into items in this module.
    unexpanded_invocations: RefCell<FxHashSet<LocalExpnId>>,

    /// Whether `#[no_implicit_prelude]` is active.
    no_implicit_prelude: bool,

    glob_importers: RefCell<Vec<&'a Import<'a>>>,
    globs: RefCell<Vec<&'a Import<'a>>>,

    /// Used to memoize the traits in this module for faster searches through all traits in scope.
    traits: RefCell<Option<Box<[(Ident, &'a NameBinding<'a>)]>>>,

    /// Span of the module itself. Used for error reporting.
    span: Span,

    expansion: ExpnId,
}

type Module<'a> = &'a ModuleData<'a>;

impl<'a> ModuleData<'a> {
    fn new(
        parent: Option<Module<'a>>,
        kind: ModuleKind,
        expansion: ExpnId,
        span: Span,
        no_implicit_prelude: bool,
    ) -> Self {
        let is_foreign = match kind {
            ModuleKind::Def(_, def_id, _) => !def_id.is_local(),
            ModuleKind::Block => false,
        };
        ModuleData {
            parent,
            kind,
            lazy_resolutions: Default::default(),
            populate_on_access: Cell::new(is_foreign),
            unexpanded_invocations: Default::default(),
            no_implicit_prelude,
            glob_importers: RefCell::new(Vec::new()),
            globs: RefCell::new(Vec::new()),
            traits: RefCell::new(None),
            span,
            expansion,
        }
    }

    fn for_each_child<R, F>(&'a self, resolver: &mut R, mut f: F)
    where
        R: AsMut<Resolver<'a>>,
        F: FnMut(&mut R, Ident, Namespace, &'a NameBinding<'a>),
    {
        for (key, name_resolution) in resolver.as_mut().resolutions(self).borrow().iter() {
            if let Some(binding) = name_resolution.borrow().binding {
                f(resolver, key.ident, key.ns, binding);
            }
        }
    }

    /// This modifies `self` in place. The traits will be stored in `self.traits`.
    fn ensure_traits<R>(&'a self, resolver: &mut R)
    where
        R: AsMut<Resolver<'a>>,
    {
        let mut traits = self.traits.borrow_mut();
        if traits.is_none() {
            let mut collected_traits = Vec::new();
            self.for_each_child(resolver, |_, name, ns, binding| {
                if ns != TypeNS {
                    return;
                }
                if let Res::Def(DefKind::Trait | DefKind::TraitAlias, _) = binding.res() {
                    collected_traits.push((name, binding))
                }
            });
            *traits = Some(collected_traits.into_boxed_slice());
        }
    }

    fn res(&self) -> Option<Res> {
        match self.kind {
            ModuleKind::Def(kind, def_id, _) => Some(Res::Def(kind, def_id)),
            _ => None,
        }
    }

    // Public for rustdoc.
    pub fn def_id(&self) -> DefId {
        self.opt_def_id().expect("`ModuleData::def_id` is called on a block module")
    }

    fn opt_def_id(&self) -> Option<DefId> {
        match self.kind {
            ModuleKind::Def(_, def_id, _) => Some(def_id),
            _ => None,
        }
    }

    // `self` resolves to the first module ancestor that `is_normal`.
    fn is_normal(&self) -> bool {
        matches!(self.kind, ModuleKind::Def(DefKind::Mod, _, _))
    }

    fn is_trait(&self) -> bool {
        matches!(self.kind, ModuleKind::Def(DefKind::Trait, _, _))
    }

    fn nearest_item_scope(&'a self) -> Module<'a> {
        match self.kind {
            ModuleKind::Def(DefKind::Enum | DefKind::Trait, ..) => {
                self.parent.expect("enum or trait module without a parent")
            }
            _ => self,
        }
    }

    /// The [`DefId`] of the nearest `mod` item ancestor (which may be this module).
    /// This may be the crate root.
    fn nearest_parent_mod(&self) -> DefId {
        match self.kind {
            ModuleKind::Def(DefKind::Mod, def_id, _) => def_id,
            _ => self.parent.expect("non-root module without parent").nearest_parent_mod(),
        }
    }

    fn is_ancestor_of(&self, mut other: &Self) -> bool {
        while !ptr::eq(self, other) {
            if let Some(parent) = other.parent {
                other = parent;
            } else {
                return false;
            }
        }
        true
    }
}

impl<'a> fmt::Debug for ModuleData<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.res())
    }
}

/// Records a possibly-private value, type, or module definition.
#[derive(Clone, Debug)]
pub struct NameBinding<'a> {
    kind: NameBindingKind<'a>,
    ambiguity: Option<(&'a NameBinding<'a>, AmbiguityKind)>,
    expansion: LocalExpnId,
    span: Span,
    vis: ty::Visibility,
}

pub trait ToNameBinding<'a> {
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a>;
}

impl<'a> ToNameBinding<'a> for &'a NameBinding<'a> {
    fn to_name_binding(self, _: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        self
    }
}

#[derive(Clone, Debug)]
enum NameBindingKind<'a> {
    Res(Res, /* is_macro_export */ bool),
    Module(Module<'a>),
    Import { binding: &'a NameBinding<'a>, import: &'a Import<'a>, used: Cell<bool> },
}

impl<'a> NameBindingKind<'a> {
    /// Is this a name binding of an import?
    fn is_import(&self) -> bool {
        matches!(*self, NameBindingKind::Import { .. })
    }
}

struct PrivacyError<'a> {
    ident: Ident,
    binding: &'a NameBinding<'a>,
    dedup_span: Span,
}

struct UseError<'a> {
    err: DiagnosticBuilder<'a, ErrorGuaranteed>,
    /// Candidates which user could `use` to access the missing type.
    candidates: Vec<ImportSuggestion>,
    /// The `DefId` of the module to place the use-statements in.
    def_id: DefId,
    /// Whether the diagnostic should say "instead" (as in `consider importing ... instead`).
    instead: bool,
    /// Extra free-form suggestion.
    suggestion: Option<(Span, &'static str, String, Applicability)>,
    /// Path `Segment`s at the place of use that failed. Used for accurate suggestion after telling
    /// the user to import the item directly.
    path: Vec<Segment>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum AmbiguityKind {
    Import,
    BuiltinAttr,
    DeriveHelper,
    MacroRulesVsModularized,
    GlobVsOuter,
    GlobVsGlob,
    GlobVsExpanded,
    MoreExpandedVsOuter,
}

impl AmbiguityKind {
    fn descr(self) -> &'static str {
        match self {
            AmbiguityKind::Import => "multiple potential import sources",
            AmbiguityKind::BuiltinAttr => "a name conflict with a builtin attribute",
            AmbiguityKind::DeriveHelper => "a name conflict with a derive helper attribute",
            AmbiguityKind::MacroRulesVsModularized => {
                "a conflict between a `macro_rules` name and a non-`macro_rules` name from another module"
            }
            AmbiguityKind::GlobVsOuter => {
                "a conflict between a name from a glob import and an outer scope during import or macro resolution"
            }
            AmbiguityKind::GlobVsGlob => "multiple glob imports of a name in the same module",
            AmbiguityKind::GlobVsExpanded => {
                "a conflict between a name from a glob import and a macro-expanded name in the same module during import or macro resolution"
            }
            AmbiguityKind::MoreExpandedVsOuter => {
                "a conflict between a macro-expanded name and a less macro-expanded name from outer scope during import or macro resolution"
            }
        }
    }
}

/// Miscellaneous bits of metadata for better ambiguity error reporting.
#[derive(Clone, Copy, PartialEq)]
enum AmbiguityErrorMisc {
    SuggestCrate,
    SuggestSelf,
    FromPrelude,
    None,
}

struct AmbiguityError<'a> {
    kind: AmbiguityKind,
    ident: Ident,
    b1: &'a NameBinding<'a>,
    b2: &'a NameBinding<'a>,
    misc1: AmbiguityErrorMisc,
    misc2: AmbiguityErrorMisc,
}

impl<'a> NameBinding<'a> {
    fn module(&self) -> Option<Module<'a>> {
        match self.kind {
            NameBindingKind::Module(module) => Some(module),
            NameBindingKind::Import { binding, .. } => binding.module(),
            _ => None,
        }
    }

    fn res(&self) -> Res {
        match self.kind {
            NameBindingKind::Res(res, _) => res,
            NameBindingKind::Module(module) => module.res().unwrap(),
            NameBindingKind::Import { binding, .. } => binding.res(),
        }
    }

    fn is_ambiguity(&self) -> bool {
        self.ambiguity.is_some()
            || match self.kind {
                NameBindingKind::Import { binding, .. } => binding.is_ambiguity(),
                _ => false,
            }
    }

    fn is_possibly_imported_variant(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { binding, .. } => binding.is_possibly_imported_variant(),
            NameBindingKind::Res(
                Res::Def(DefKind::Variant | DefKind::Ctor(CtorOf::Variant, ..), _),
                _,
            ) => true,
            NameBindingKind::Res(..) | NameBindingKind::Module(..) => false,
        }
    }

    fn is_extern_crate(&self) -> bool {
        match self.kind {
            NameBindingKind::Import {
                import: &Import { kind: ImportKind::ExternCrate { .. }, .. },
                ..
            } => true,
            NameBindingKind::Module(&ModuleData {
                kind: ModuleKind::Def(DefKind::Mod, def_id, _),
                ..
            }) => def_id.is_crate_root(),
            _ => false,
        }
    }

    fn is_import(&self) -> bool {
        matches!(self.kind, NameBindingKind::Import { .. })
    }

    fn is_glob_import(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { import, .. } => import.is_glob(),
            _ => false,
        }
    }

    fn is_importable(&self) -> bool {
        !matches!(
            self.res(),
            Res::Def(DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy, _)
        )
    }

    fn macro_kind(&self) -> Option<MacroKind> {
        self.res().macro_kind()
    }

    // Suppose that we resolved macro invocation with `invoc_parent_expansion` to binding `binding`
    // at some expansion round `max(invoc, binding)` when they both emerged from macros.
    // Then this function returns `true` if `self` may emerge from a macro *after* that
    // in some later round and screw up our previously found resolution.
    // See more detailed explanation in
    // https://github.com/rust-lang/rust/pull/53778#issuecomment-419224049
    fn may_appear_after(
        &self,
        invoc_parent_expansion: LocalExpnId,
        binding: &NameBinding<'_>,
    ) -> bool {
        // self > max(invoc, binding) => !(self <= invoc || self <= binding)
        // Expansions are partially ordered, so "may appear after" is an inversion of
        // "certainly appears before or simultaneously" and includes unordered cases.
        let self_parent_expansion = self.expansion;
        let other_parent_expansion = binding.expansion;
        let certainly_before_other_or_simultaneously =
            other_parent_expansion.is_descendant_of(self_parent_expansion);
        let certainly_before_invoc_or_simultaneously =
            invoc_parent_expansion.is_descendant_of(self_parent_expansion);
        !(certainly_before_other_or_simultaneously || certainly_before_invoc_or_simultaneously)
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExternPreludeEntry<'a> {
    extern_crate_item: Option<&'a NameBinding<'a>>,
    pub introduced_by_item: bool,
}

/// Used for better errors for E0773
enum BuiltinMacroState {
    NotYetSeen(SyntaxExtensionKind),
    AlreadySeen(Span),
}

struct DeriveData {
    resolutions: DeriveResolutions,
    helper_attrs: Vec<(usize, Ident)>,
    has_derive_copy: bool,
}

#[derive(Clone)]
struct MacroData {
    ext: Lrc<SyntaxExtension>,
    macro_rules: bool,
}

/// The main resolver class.
///
/// This is the visitor that walks the whole crate.
pub struct Resolver<'a> {
    session: &'a Session,

    definitions: Definitions,
    /// Item with a given `LocalDefId` was defined during macro expansion with ID `ExpnId`.
    expn_that_defined: FxHashMap<LocalDefId, ExpnId>,
    /// Reference span for definitions.
    source_span: IndexVec<LocalDefId, Span>,

    graph_root: Module<'a>,

    prelude: Option<Module<'a>>,
    extern_prelude: FxHashMap<Ident, ExternPreludeEntry<'a>>,

    /// N.B., this is used only for better diagnostics, not name resolution itself.
    has_self: FxHashSet<DefId>,

    /// Names of fields of an item `DefId` accessible with dot syntax.
    /// Used for hints during error reporting.
    field_names: FxHashMap<DefId, Vec<Spanned<Symbol>>>,

    /// All imports known to succeed or fail.
    determined_imports: Vec<&'a Import<'a>>,

    /// All non-determined imports.
    indeterminate_imports: Vec<&'a Import<'a>>,

    // Spans for local variables found during pattern resolution.
    // Used for suggestions during error reporting.
    pat_span_map: NodeMap<Span>,

    /// Resolutions for nodes that have a single resolution.
    partial_res_map: NodeMap<PartialRes>,
    /// Resolutions for import nodes, which have multiple resolutions in different namespaces.
    import_res_map: NodeMap<PerNS<Option<Res>>>,
    /// Resolutions for labels (node IDs of their corresponding blocks or loops).
    label_res_map: NodeMap<NodeId>,
    /// Resolutions for lifetimes.
    lifetimes_res_map: NodeMap<LifetimeRes>,
    /// Mapping from generics `def_id`s to TAIT generics `def_id`s.
    /// For each captured lifetime (e.g., 'a), we create a new lifetime parameter that is a generic
    /// defined on the TAIT, so we have type Foo<'a1> = ... and we establish a mapping in this
    /// field from the original parameter 'a to the new parameter 'a1.
    generics_def_id_map: Vec<FxHashMap<LocalDefId, LocalDefId>>,
    /// Lifetime parameters that lowering will have to introduce.
    extra_lifetime_params_map: NodeMap<Vec<(Ident, NodeId, LifetimeRes)>>,

    /// `CrateNum` resolutions of `extern crate` items.
    extern_crate_map: FxHashMap<LocalDefId, CrateNum>,
    reexport_map: FxHashMap<LocalDefId, Vec<ModChild>>,
    trait_map: NodeMap<Vec<TraitCandidate>>,

    /// A map from nodes to anonymous modules.
    /// Anonymous modules are pseudo-modules that are implicitly created around items
    /// contained within blocks.
    ///
    /// For example, if we have this:
    ///
    ///  fn f() {
    ///      fn g() {
    ///          ...
    ///      }
    ///  }
    ///
    /// There will be an anonymous module created around `g` with the ID of the
    /// entry block for `f`.
    block_map: NodeMap<Module<'a>>,
    /// A fake module that contains no definition and no prelude. Used so that
    /// some AST passes can generate identifiers that only resolve to local or
    /// language items.
    empty_module: Module<'a>,
    module_map: FxHashMap<DefId, Module<'a>>,
    binding_parent_modules: FxHashMap<Interned<'a, NameBinding<'a>>, Module<'a>>,
    underscore_disambiguator: u32,

    /// Maps glob imports to the names of items actually imported.
    glob_map: FxHashMap<LocalDefId, FxHashSet<Symbol>>,
    /// Visibilities in "lowered" form, for all entities that have them.
    visibilities: FxHashMap<LocalDefId, ty::Visibility>,
    has_pub_restricted: bool,
    used_imports: FxHashSet<NodeId>,
    maybe_unused_trait_imports: FxIndexSet<LocalDefId>,
    maybe_unused_extern_crates: Vec<(LocalDefId, Span)>,

    /// Privacy errors are delayed until the end in order to deduplicate them.
    privacy_errors: Vec<PrivacyError<'a>>,
    /// Ambiguity errors are delayed for deduplication.
    ambiguity_errors: Vec<AmbiguityError<'a>>,
    /// `use` injections are delayed for better placement and deduplication.
    use_injections: Vec<UseError<'a>>,
    /// Crate-local macro expanded `macro_export` referred to by a module-relative path.
    macro_expanded_macro_export_errors: BTreeSet<(Span, Span)>,

    arenas: &'a ResolverArenas<'a>,
    dummy_binding: &'a NameBinding<'a>,

    crate_loader: CrateLoader<'a>,
    macro_names: FxHashSet<Ident>,
    builtin_macros: FxHashMap<Symbol, BuiltinMacroState>,
    /// A small map keeping true kinds of built-in macros that appear to be fn-like on
    /// the surface (`macro` items in libcore), but are actually attributes or derives.
    builtin_macro_kinds: FxHashMap<LocalDefId, MacroKind>,
    registered_attrs: FxHashSet<Ident>,
    registered_tools: RegisteredTools,
    macro_use_prelude: FxHashMap<Symbol, &'a NameBinding<'a>>,
    macro_map: FxHashMap<DefId, MacroData>,
    dummy_ext_bang: Lrc<SyntaxExtension>,
    dummy_ext_derive: Lrc<SyntaxExtension>,
    non_macro_attr: Lrc<SyntaxExtension>,
    local_macro_def_scopes: FxHashMap<LocalDefId, Module<'a>>,
    ast_transform_scopes: FxHashMap<LocalExpnId, Module<'a>>,
    unused_macros: FxHashMap<LocalDefId, (NodeId, Ident)>,
    unused_macro_rules: FxHashMap<(LocalDefId, usize), (Ident, Span)>,
    proc_macro_stubs: FxHashSet<LocalDefId>,
    /// Traces collected during macro resolution and validated when it's complete.
    single_segment_macro_resolutions:
        Vec<(Ident, MacroKind, ParentScope<'a>, Option<&'a NameBinding<'a>>)>,
    multi_segment_macro_resolutions:
        Vec<(Vec<Segment>, Span, MacroKind, ParentScope<'a>, Option<Res>)>,
    builtin_attrs: Vec<(Ident, ParentScope<'a>)>,
    /// `derive(Copy)` marks items they are applied to so they are treated specially later.
    /// Derive macros cannot modify the item themselves and have to store the markers in the global
    /// context, so they attach the markers to derive container IDs using this resolver table.
    containers_deriving_copy: FxHashSet<LocalExpnId>,
    /// Parent scopes in which the macros were invoked.
    /// FIXME: `derives` are missing in these parent scopes and need to be taken from elsewhere.
    invocation_parent_scopes: FxHashMap<LocalExpnId, ParentScope<'a>>,
    /// `macro_rules` scopes *produced* by expanding the macro invocations,
    /// include all the `macro_rules` items and other invocations generated by them.
    output_macro_rules_scopes: FxHashMap<LocalExpnId, MacroRulesScopeRef<'a>>,
    /// `macro_rules` scopes produced by `macro_rules` item definitions.
    macro_rules_scopes: FxHashMap<LocalDefId, MacroRulesScopeRef<'a>>,
    /// Helper attributes that are in scope for the given expansion.
    helper_attrs: FxHashMap<LocalExpnId, Vec<Ident>>,
    /// Ready or in-progress results of resolving paths inside the `#[derive(...)]` attribute
    /// with the given `ExpnId`.
    derive_data: FxHashMap<LocalExpnId, DeriveData>,

    /// Avoid duplicated errors for "name already defined".
    name_already_seen: FxHashMap<Symbol, Span>,

    potentially_unused_imports: Vec<&'a Import<'a>>,

    /// Table for mapping struct IDs into struct constructor IDs,
    /// it's not used during normal resolution, only for better error reporting.
    /// Also includes of list of each fields visibility
    struct_constructors: DefIdMap<(Res, ty::Visibility, Vec<ty::Visibility>)>,

    /// Features enabled for this crate.
    active_features: FxHashSet<Symbol>,

    lint_buffer: LintBuffer,

    next_node_id: NodeId,

    node_id_to_def_id: FxHashMap<ast::NodeId, LocalDefId>,
    def_id_to_node_id: IndexVec<LocalDefId, ast::NodeId>,

    /// Indices of unnamed struct or variant fields with unresolved attributes.
    placeholder_field_indices: FxHashMap<NodeId, usize>,
    /// When collecting definitions from an AST fragment produced by a macro invocation `ExpnId`
    /// we know what parent node that fragment should be attached to thanks to this table,
    /// and how the `impl Trait` fragments were introduced.
    invocation_parents: FxHashMap<LocalExpnId, (LocalDefId, ImplTraitContext)>,

    /// Some way to know that we are in a *trait* impl in `visit_assoc_item`.
    /// FIXME: Replace with a more general AST map (together with some other fields).
    trait_impl_items: FxHashSet<LocalDefId>,

    legacy_const_generic_args: FxHashMap<DefId, Option<Vec<usize>>>,
    /// Amount of lifetime parameters for each item in the crate.
    item_generics_num_lifetimes: FxHashMap<LocalDefId, usize>,

    main_def: Option<MainDefinition>,
    trait_impls: FxIndexMap<DefId, Vec<LocalDefId>>,
    /// A list of proc macro LocalDefIds, written out in the order in which
    /// they are declared in the static array generated by proc_macro_harness.
    proc_macros: Vec<NodeId>,
    confused_type_with_std_module: FxHashMap<Span, Span>,

    access_levels: AccessLevels,
}

/// Nothing really interesting here; it just provides memory for the rest of the crate.
#[derive(Default)]
pub struct ResolverArenas<'a> {
    modules: TypedArena<ModuleData<'a>>,
    local_modules: RefCell<Vec<Module<'a>>>,
    imports: TypedArena<Import<'a>>,
    name_resolutions: TypedArena<RefCell<NameResolution<'a>>>,
    ast_paths: TypedArena<ast::Path>,
    dropless: DroplessArena,
}

impl<'a> ResolverArenas<'a> {
    fn new_module(
        &'a self,
        parent: Option<Module<'a>>,
        kind: ModuleKind,
        expn_id: ExpnId,
        span: Span,
        no_implicit_prelude: bool,
        module_map: &mut FxHashMap<DefId, Module<'a>>,
    ) -> Module<'a> {
        let module =
            self.modules.alloc(ModuleData::new(parent, kind, expn_id, span, no_implicit_prelude));
        let def_id = module.opt_def_id();
        if def_id.map_or(true, |def_id| def_id.is_local()) {
            self.local_modules.borrow_mut().push(module);
        }
        if let Some(def_id) = def_id {
            module_map.insert(def_id, module);
        }
        module
    }
    fn local_modules(&'a self) -> std::cell::Ref<'a, Vec<Module<'a>>> {
        self.local_modules.borrow()
    }
    fn alloc_name_binding(&'a self, name_binding: NameBinding<'a>) -> &'a NameBinding<'a> {
        self.dropless.alloc(name_binding)
    }
    fn alloc_import(&'a self, import: Import<'a>) -> &'a Import<'_> {
        self.imports.alloc(import)
    }
    fn alloc_name_resolution(&'a self) -> &'a RefCell<NameResolution<'a>> {
        self.name_resolutions.alloc(Default::default())
    }
    fn alloc_macro_rules_scope(&'a self, scope: MacroRulesScope<'a>) -> MacroRulesScopeRef<'a> {
        Interned::new_unchecked(self.dropless.alloc(Cell::new(scope)))
    }
    fn alloc_macro_rules_binding(
        &'a self,
        binding: MacroRulesBinding<'a>,
    ) -> &'a MacroRulesBinding<'a> {
        self.dropless.alloc(binding)
    }
    fn alloc_ast_paths(&'a self, paths: &[ast::Path]) -> &'a [ast::Path] {
        self.ast_paths.alloc_from_iter(paths.iter().cloned())
    }
    fn alloc_pattern_spans(&'a self, spans: impl Iterator<Item = Span>) -> &'a [Span] {
        self.dropless.alloc_from_iter(spans)
    }
}

impl<'a> AsMut<Resolver<'a>> for Resolver<'a> {
    fn as_mut(&mut self) -> &mut Resolver<'a> {
        self
    }
}

impl<'a, 'b> DefIdTree for &'a Resolver<'b> {
    #[inline]
    fn opt_parent(self, id: DefId) -> Option<DefId> {
        match id.as_local() {
            Some(id) => self.definitions.def_key(id).parent,
            None => self.cstore().def_key(id).parent,
        }
        .map(|index| DefId { index, ..id })
    }
}

impl Resolver<'_> {
    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.node_id_to_def_id.get(&node).copied()
    }

    pub fn local_def_id(&self, node: NodeId) -> LocalDefId {
        self.opt_local_def_id(node).unwrap_or_else(|| panic!("no entry for node id: `{:?}`", node))
    }

    /// Adds a definition with a parent definition.
    fn create_def(
        &mut self,
        parent: LocalDefId,
        node_id: ast::NodeId,
        data: DefPathData,
        expn_id: ExpnId,
        span: Span,
    ) -> LocalDefId {
        assert!(
            !self.node_id_to_def_id.contains_key(&node_id),
            "adding a def'n for node-id {:?} and data {:?} but a previous def'n exists: {:?}",
            node_id,
            data,
            self.definitions.def_key(self.node_id_to_def_id[&node_id]),
        );

        let def_id = self.definitions.create_def(parent, data);

        // Create the definition.
        if expn_id != ExpnId::root() {
            self.expn_that_defined.insert(def_id, expn_id);
        }

        // A relative span's parent must be an absolute span.
        debug_assert_eq!(span.data_untracked().parent, None);
        let _id = self.source_span.push(span);
        debug_assert_eq!(_id, def_id);

        // Some things for which we allocate `LocalDefId`s don't correspond to
        // anything in the AST, so they don't have a `NodeId`. For these cases
        // we don't need a mapping from `NodeId` to `LocalDefId`.
        if node_id != ast::DUMMY_NODE_ID {
            debug!("create_def: def_id_to_node_id[{:?}] <-> {:?}", def_id, node_id);
            self.node_id_to_def_id.insert(node_id, def_id);
        }
        assert_eq!(self.def_id_to_node_id.push(node_id), def_id);

        def_id
    }

    fn item_generics_num_lifetimes(&self, def_id: DefId) -> usize {
        if let Some(def_id) = def_id.as_local() {
            self.item_generics_num_lifetimes[&def_id]
        } else {
            self.cstore().item_generics_num_lifetimes(def_id, self.session)
        }
    }
}

impl<'a> Resolver<'a> {
    pub fn new(
        session: &'a Session,
        krate: &Crate,
        crate_name: &str,
        metadata_loader: Box<MetadataLoaderDyn>,
        arenas: &'a ResolverArenas<'a>,
    ) -> Resolver<'a> {
        let root_def_id = CRATE_DEF_ID.to_def_id();
        let mut module_map = FxHashMap::default();
        let graph_root = arenas.new_module(
            None,
            ModuleKind::Def(DefKind::Mod, root_def_id, kw::Empty),
            ExpnId::root(),
            krate.spans.inner_span,
            session.contains_name(&krate.attrs, sym::no_implicit_prelude),
            &mut module_map,
        );
        let empty_module = arenas.new_module(
            None,
            ModuleKind::Def(DefKind::Mod, root_def_id, kw::Empty),
            ExpnId::root(),
            DUMMY_SP,
            true,
            &mut FxHashMap::default(),
        );

        let definitions = Definitions::new(session.local_stable_crate_id());

        let mut visibilities = FxHashMap::default();
        visibilities.insert(CRATE_DEF_ID, ty::Visibility::Public);

        let mut def_id_to_node_id = IndexVec::default();
        assert_eq!(def_id_to_node_id.push(CRATE_NODE_ID), CRATE_DEF_ID);
        let mut node_id_to_def_id = FxHashMap::default();
        node_id_to_def_id.insert(CRATE_NODE_ID, CRATE_DEF_ID);

        let mut invocation_parents = FxHashMap::default();
        invocation_parents.insert(LocalExpnId::ROOT, (CRATE_DEF_ID, ImplTraitContext::Existential));

        let mut source_span = IndexVec::default();
        let _id = source_span.push(krate.spans.inner_span);
        debug_assert_eq!(_id, CRATE_DEF_ID);

        let mut extern_prelude: FxHashMap<Ident, ExternPreludeEntry<'_>> = session
            .opts
            .externs
            .iter()
            .filter(|(_, entry)| entry.add_prelude)
            .map(|(name, _)| (Ident::from_str(name), Default::default()))
            .collect();

        if !session.contains_name(&krate.attrs, sym::no_core) {
            extern_prelude.insert(Ident::with_dummy_span(sym::core), Default::default());
            if !session.contains_name(&krate.attrs, sym::no_std) {
                extern_prelude.insert(Ident::with_dummy_span(sym::std), Default::default());
            }
        }

        let (registered_attrs, registered_tools) =
            macros::registered_attrs_and_tools(session, &krate.attrs);

        let features = session.features_untracked();

        let mut resolver = Resolver {
            session,

            definitions,
            expn_that_defined: Default::default(),
            source_span,

            // The outermost module has def ID 0; this is not reflected in the
            // AST.
            graph_root,
            prelude: None,
            extern_prelude,

            has_self: FxHashSet::default(),
            field_names: FxHashMap::default(),

            determined_imports: Vec::new(),
            indeterminate_imports: Vec::new(),

            pat_span_map: Default::default(),
            partial_res_map: Default::default(),
            import_res_map: Default::default(),
            label_res_map: Default::default(),
            lifetimes_res_map: Default::default(),
            generics_def_id_map: Vec::new(),
            extra_lifetime_params_map: Default::default(),
            extern_crate_map: Default::default(),
            reexport_map: FxHashMap::default(),
            trait_map: NodeMap::default(),
            underscore_disambiguator: 0,
            empty_module,
            module_map,
            block_map: Default::default(),
            binding_parent_modules: FxHashMap::default(),
            ast_transform_scopes: FxHashMap::default(),

            glob_map: Default::default(),
            visibilities,
            has_pub_restricted: false,
            used_imports: FxHashSet::default(),
            maybe_unused_trait_imports: Default::default(),
            maybe_unused_extern_crates: Vec::new(),

            privacy_errors: Vec::new(),
            ambiguity_errors: Vec::new(),
            use_injections: Vec::new(),
            macro_expanded_macro_export_errors: BTreeSet::new(),

            arenas,
            dummy_binding: arenas.alloc_name_binding(NameBinding {
                kind: NameBindingKind::Res(Res::Err, false),
                ambiguity: None,
                expansion: LocalExpnId::ROOT,
                span: DUMMY_SP,
                vis: ty::Visibility::Public,
            }),

            crate_loader: CrateLoader::new(session, metadata_loader, crate_name),
            macro_names: FxHashSet::default(),
            builtin_macros: Default::default(),
            builtin_macro_kinds: Default::default(),
            registered_attrs,
            registered_tools,
            macro_use_prelude: FxHashMap::default(),
            macro_map: FxHashMap::default(),
            dummy_ext_bang: Lrc::new(SyntaxExtension::dummy_bang(session.edition())),
            dummy_ext_derive: Lrc::new(SyntaxExtension::dummy_derive(session.edition())),
            non_macro_attr: Lrc::new(SyntaxExtension::non_macro_attr(session.edition())),
            invocation_parent_scopes: Default::default(),
            output_macro_rules_scopes: Default::default(),
            macro_rules_scopes: Default::default(),
            helper_attrs: Default::default(),
            derive_data: Default::default(),
            local_macro_def_scopes: FxHashMap::default(),
            name_already_seen: FxHashMap::default(),
            potentially_unused_imports: Vec::new(),
            struct_constructors: Default::default(),
            unused_macros: Default::default(),
            unused_macro_rules: Default::default(),
            proc_macro_stubs: Default::default(),
            single_segment_macro_resolutions: Default::default(),
            multi_segment_macro_resolutions: Default::default(),
            builtin_attrs: Default::default(),
            containers_deriving_copy: Default::default(),
            active_features: features
                .declared_lib_features
                .iter()
                .map(|(feat, ..)| *feat)
                .chain(features.declared_lang_features.iter().map(|(feat, ..)| *feat))
                .collect(),
            lint_buffer: LintBuffer::default(),
            next_node_id: CRATE_NODE_ID,
            node_id_to_def_id,
            def_id_to_node_id,
            placeholder_field_indices: Default::default(),
            invocation_parents,
            trait_impl_items: Default::default(),
            legacy_const_generic_args: Default::default(),
            item_generics_num_lifetimes: Default::default(),
            main_def: Default::default(),
            trait_impls: Default::default(),
            proc_macros: Default::default(),
            confused_type_with_std_module: Default::default(),
            access_levels: Default::default(),
        };

        let root_parent_scope = ParentScope::module(graph_root, &resolver);
        resolver.invocation_parent_scopes.insert(LocalExpnId::ROOT, root_parent_scope);

        resolver
    }

    fn new_module(
        &mut self,
        parent: Option<Module<'a>>,
        kind: ModuleKind,
        expn_id: ExpnId,
        span: Span,
        no_implicit_prelude: bool,
    ) -> Module<'a> {
        let module_map = &mut self.module_map;
        self.arenas.new_module(parent, kind, expn_id, span, no_implicit_prelude, module_map)
    }

    pub fn next_node_id(&mut self) -> NodeId {
        let start = self.next_node_id;
        let next = start.as_u32().checked_add(1).expect("input too large; ran out of NodeIds");
        self.next_node_id = ast::NodeId::from_u32(next);
        start
    }

    pub fn next_node_ids(&mut self, count: usize) -> std::ops::Range<NodeId> {
        let start = self.next_node_id;
        let end = start.as_usize().checked_add(count).expect("input too large; ran out of NodeIds");
        self.next_node_id = ast::NodeId::from_usize(end);
        start..self.next_node_id
    }

    pub fn lint_buffer(&mut self) -> &mut LintBuffer {
        &mut self.lint_buffer
    }

    pub fn arenas() -> ResolverArenas<'a> {
        Default::default()
    }

    pub fn into_outputs(
        self,
    ) -> (Definitions, Box<CrateStoreDyn>, ResolverOutputs, ty::ResolverAstLowering) {
        let proc_macros = self.proc_macros.iter().map(|id| self.local_def_id(*id)).collect();
        let definitions = self.definitions;
        let cstore = Box::new(self.crate_loader.into_cstore());
        let source_span = self.source_span;
        let expn_that_defined = self.expn_that_defined;
        let visibilities = self.visibilities;
        let has_pub_restricted = self.has_pub_restricted;
        let extern_crate_map = self.extern_crate_map;
        let reexport_map = self.reexport_map;
        let maybe_unused_trait_imports = self.maybe_unused_trait_imports;
        let maybe_unused_extern_crates = self.maybe_unused_extern_crates;
        let glob_map = self.glob_map;
        let main_def = self.main_def;
        let confused_type_with_std_module = self.confused_type_with_std_module;
        let access_levels = self.access_levels;
        let resolutions = ResolverOutputs {
            source_span,
            expn_that_defined,
            visibilities,
            has_pub_restricted,
            access_levels,
            extern_crate_map,
            reexport_map,
            glob_map,
            maybe_unused_trait_imports,
            maybe_unused_extern_crates,
            extern_prelude: self
                .extern_prelude
                .iter()
                .map(|(ident, entry)| (ident.name, entry.introduced_by_item))
                .collect(),
            main_def,
            trait_impls: self.trait_impls,
            proc_macros,
            confused_type_with_std_module,
            registered_tools: self.registered_tools,
        };
        let resolutions_lowering = ty::ResolverAstLowering {
            legacy_const_generic_args: self.legacy_const_generic_args,
            partial_res_map: self.partial_res_map,
            import_res_map: self.import_res_map,
            label_res_map: self.label_res_map,
            lifetimes_res_map: self.lifetimes_res_map,
            generics_def_id_map: self.generics_def_id_map,
            extra_lifetime_params_map: self.extra_lifetime_params_map,
            next_node_id: self.next_node_id,
            node_id_to_def_id: self.node_id_to_def_id,
            def_id_to_node_id: self.def_id_to_node_id,
            trait_map: self.trait_map,
            builtin_macro_kinds: self.builtin_macro_kinds,
        };
        (definitions, cstore, resolutions, resolutions_lowering)
    }

    pub fn clone_outputs(
        &self,
    ) -> (Definitions, Box<CrateStoreDyn>, ResolverOutputs, ty::ResolverAstLowering) {
        let proc_macros = self.proc_macros.iter().map(|id| self.local_def_id(*id)).collect();
        let definitions = self.definitions.clone();
        let cstore = Box::new(self.cstore().clone());
        let resolutions = ResolverOutputs {
            source_span: self.source_span.clone(),
            expn_that_defined: self.expn_that_defined.clone(),
            visibilities: self.visibilities.clone(),
            has_pub_restricted: self.has_pub_restricted,
            extern_crate_map: self.extern_crate_map.clone(),
            reexport_map: self.reexport_map.clone(),
            glob_map: self.glob_map.clone(),
            maybe_unused_trait_imports: self.maybe_unused_trait_imports.clone(),
            maybe_unused_extern_crates: self.maybe_unused_extern_crates.clone(),
            extern_prelude: self
                .extern_prelude
                .iter()
                .map(|(ident, entry)| (ident.name, entry.introduced_by_item))
                .collect(),
            main_def: self.main_def,
            trait_impls: self.trait_impls.clone(),
            proc_macros,
            confused_type_with_std_module: self.confused_type_with_std_module.clone(),
            registered_tools: self.registered_tools.clone(),
            access_levels: self.access_levels.clone(),
        };
        let resolutions_lowering = ty::ResolverAstLowering {
            legacy_const_generic_args: self.legacy_const_generic_args.clone(),
            partial_res_map: self.partial_res_map.clone(),
            import_res_map: self.import_res_map.clone(),
            label_res_map: self.label_res_map.clone(),
            lifetimes_res_map: self.lifetimes_res_map.clone(),
            generics_def_id_map: self.generics_def_id_map.clone(),
            extra_lifetime_params_map: self.extra_lifetime_params_map.clone(),
            next_node_id: self.next_node_id.clone(),
            node_id_to_def_id: self.node_id_to_def_id.clone(),
            def_id_to_node_id: self.def_id_to_node_id.clone(),
            trait_map: self.trait_map.clone(),
            builtin_macro_kinds: self.builtin_macro_kinds.clone(),
        };
        (definitions, cstore, resolutions, resolutions_lowering)
    }

    fn create_stable_hashing_context(&self) -> StableHashingContext<'_> {
        StableHashingContext::new(
            self.session,
            &self.definitions,
            self.crate_loader.cstore(),
            &self.source_span,
        )
    }

    pub fn cstore(&self) -> &CStore {
        self.crate_loader.cstore()
    }

    fn dummy_ext(&self, macro_kind: MacroKind) -> Lrc<SyntaxExtension> {
        match macro_kind {
            MacroKind::Bang => self.dummy_ext_bang.clone(),
            MacroKind::Derive => self.dummy_ext_derive.clone(),
            MacroKind::Attr => self.non_macro_attr.clone(),
        }
    }

    /// Runs the function on each namespace.
    fn per_ns<F: FnMut(&mut Self, Namespace)>(&mut self, mut f: F) {
        f(self, TypeNS);
        f(self, ValueNS);
        f(self, MacroNS);
    }

    fn is_builtin_macro(&mut self, res: Res) -> bool {
        self.get_macro(res).map_or(false, |macro_data| macro_data.ext.builtin_name.is_some())
    }

    fn macro_def(&self, mut ctxt: SyntaxContext) -> DefId {
        loop {
            match ctxt.outer_expn_data().macro_def_id {
                Some(def_id) => return def_id,
                None => ctxt.remove_mark(),
            };
        }
    }

    /// Entry point to crate resolution.
    pub fn resolve_crate(&mut self, krate: &Crate) {
        self.session.time("resolve_crate", || {
            self.session.time("finalize_imports", || ImportResolver { r: self }.finalize_imports());
            self.session.time("resolve_access_levels", || {
                AccessLevelsVisitor::compute_access_levels(self, krate)
            });
            self.session.time("finalize_macro_resolutions", || self.finalize_macro_resolutions());
            self.session.time("late_resolve_crate", || self.late_resolve_crate(krate));
            self.session.time("resolve_main", || self.resolve_main());
            self.session.time("resolve_check_unused", || self.check_unused(krate));
            self.session.time("resolve_report_errors", || self.report_errors(krate));
            self.session.time("resolve_postprocess", || self.crate_loader.postprocess(krate));
        });
    }

    pub fn traits_in_scope(
        &mut self,
        current_trait: Option<Module<'a>>,
        parent_scope: &ParentScope<'a>,
        ctxt: SyntaxContext,
        assoc_item: Option<(Symbol, Namespace)>,
    ) -> Vec<TraitCandidate> {
        let mut found_traits = Vec::new();

        if let Some(module) = current_trait {
            if self.trait_may_have_item(Some(module), assoc_item) {
                let def_id = module.def_id();
                found_traits.push(TraitCandidate { def_id, import_ids: smallvec![] });
            }
        }

        self.visit_scopes(ScopeSet::All(TypeNS, false), parent_scope, ctxt, |this, scope, _, _| {
            match scope {
                Scope::Module(module, _) => {
                    this.traits_in_module(module, assoc_item, &mut found_traits);
                }
                Scope::StdLibPrelude => {
                    if let Some(module) = this.prelude {
                        this.traits_in_module(module, assoc_item, &mut found_traits);
                    }
                }
                Scope::ExternPrelude | Scope::ToolPrelude | Scope::BuiltinTypes => {}
                _ => unreachable!(),
            }
            None::<()>
        });

        found_traits
    }

    fn traits_in_module(
        &mut self,
        module: Module<'a>,
        assoc_item: Option<(Symbol, Namespace)>,
        found_traits: &mut Vec<TraitCandidate>,
    ) {
        module.ensure_traits(self);
        let traits = module.traits.borrow();
        for (trait_name, trait_binding) in traits.as_ref().unwrap().iter() {
            if self.trait_may_have_item(trait_binding.module(), assoc_item) {
                let def_id = trait_binding.res().def_id();
                let import_ids = self.find_transitive_imports(&trait_binding.kind, *trait_name);
                found_traits.push(TraitCandidate { def_id, import_ids });
            }
        }
    }

    // List of traits in scope is pruned on best effort basis. We reject traits not having an
    // associated item with the given name and namespace (if specified). This is a conservative
    // optimization, proper hygienic type-based resolution of associated items is done in typeck.
    // We don't reject trait aliases (`trait_module == None`) because we don't have access to their
    // associated items.
    fn trait_may_have_item(
        &mut self,
        trait_module: Option<Module<'a>>,
        assoc_item: Option<(Symbol, Namespace)>,
    ) -> bool {
        match (trait_module, assoc_item) {
            (Some(trait_module), Some((name, ns))) => {
                self.resolutions(trait_module).borrow().iter().any(|resolution| {
                    let (&BindingKey { ident: assoc_ident, ns: assoc_ns, .. }, _) = resolution;
                    assoc_ns == ns && assoc_ident.name == name
                })
            }
            _ => true,
        }
    }

    fn find_transitive_imports(
        &mut self,
        mut kind: &NameBindingKind<'_>,
        trait_name: Ident,
    ) -> SmallVec<[LocalDefId; 1]> {
        let mut import_ids = smallvec![];
        while let NameBindingKind::Import { import, binding, .. } = kind {
            let id = self.local_def_id(import.id);
            self.maybe_unused_trait_imports.insert(id);
            self.add_to_glob_map(&import, trait_name);
            import_ids.push(id);
            kind = &binding.kind;
        }
        import_ids
    }

    fn new_key(&mut self, ident: Ident, ns: Namespace) -> BindingKey {
        let ident = ident.normalize_to_macros_2_0();
        let disambiguator = if ident.name == kw::Underscore {
            self.underscore_disambiguator += 1;
            self.underscore_disambiguator
        } else {
            0
        };
        BindingKey { ident, ns, disambiguator }
    }

    fn resolutions(&mut self, module: Module<'a>) -> &'a Resolutions<'a> {
        if module.populate_on_access.get() {
            module.populate_on_access.set(false);
            self.build_reduced_graph_external(module);
        }
        &module.lazy_resolutions
    }

    fn resolution(
        &mut self,
        module: Module<'a>,
        key: BindingKey,
    ) -> &'a RefCell<NameResolution<'a>> {
        *self
            .resolutions(module)
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| self.arenas.alloc_name_resolution())
    }

    fn record_use(
        &mut self,
        ident: Ident,
        used_binding: &'a NameBinding<'a>,
        is_lexical_scope: bool,
    ) {
        if let Some((b2, kind)) = used_binding.ambiguity {
            self.ambiguity_errors.push(AmbiguityError {
                kind,
                ident,
                b1: used_binding,
                b2,
                misc1: AmbiguityErrorMisc::None,
                misc2: AmbiguityErrorMisc::None,
            });
        }
        if let NameBindingKind::Import { import, binding, ref used } = used_binding.kind {
            // Avoid marking `extern crate` items that refer to a name from extern prelude,
            // but not introduce it, as used if they are accessed from lexical scope.
            if is_lexical_scope {
                if let Some(entry) = self.extern_prelude.get(&ident.normalize_to_macros_2_0()) {
                    if let Some(crate_item) = entry.extern_crate_item {
                        if ptr::eq(used_binding, crate_item) && !entry.introduced_by_item {
                            return;
                        }
                    }
                }
            }
            used.set(true);
            import.used.set(true);
            self.used_imports.insert(import.id);
            self.add_to_glob_map(&import, ident);
            self.record_use(ident, binding, false);
        }
    }

    #[inline]
    fn add_to_glob_map(&mut self, import: &Import<'_>, ident: Ident) {
        if import.is_glob() {
            let def_id = self.local_def_id(import.id);
            self.glob_map.entry(def_id).or_default().insert(ident.name);
        }
    }

    fn resolve_crate_root(&mut self, ident: Ident) -> Module<'a> {
        debug!("resolve_crate_root({:?})", ident);
        let mut ctxt = ident.span.ctxt();
        let mark = if ident.name == kw::DollarCrate {
            // When resolving `$crate` from a `macro_rules!` invoked in a `macro`,
            // we don't want to pretend that the `macro_rules!` definition is in the `macro`
            // as described in `SyntaxContext::apply_mark`, so we ignore prepended opaque marks.
            // FIXME: This is only a guess and it doesn't work correctly for `macro_rules!`
            // definitions actually produced by `macro` and `macro` definitions produced by
            // `macro_rules!`, but at least such configurations are not stable yet.
            ctxt = ctxt.normalize_to_macro_rules();
            debug!(
                "resolve_crate_root: marks={:?}",
                ctxt.marks().into_iter().map(|(i, t)| (i.expn_data(), t)).collect::<Vec<_>>()
            );
            let mut iter = ctxt.marks().into_iter().rev().peekable();
            let mut result = None;
            // Find the last opaque mark from the end if it exists.
            while let Some(&(mark, transparency)) = iter.peek() {
                if transparency == Transparency::Opaque {
                    result = Some(mark);
                    iter.next();
                } else {
                    break;
                }
            }
            debug!(
                "resolve_crate_root: found opaque mark {:?} {:?}",
                result,
                result.map(|r| r.expn_data())
            );
            // Then find the last semi-transparent mark from the end if it exists.
            for (mark, transparency) in iter {
                if transparency == Transparency::SemiTransparent {
                    result = Some(mark);
                } else {
                    break;
                }
            }
            debug!(
                "resolve_crate_root: found semi-transparent mark {:?} {:?}",
                result,
                result.map(|r| r.expn_data())
            );
            result
        } else {
            debug!("resolve_crate_root: not DollarCrate");
            ctxt = ctxt.normalize_to_macros_2_0();
            ctxt.adjust(ExpnId::root())
        };
        let module = match mark {
            Some(def) => self.expn_def_scope(def),
            None => {
                debug!(
                    "resolve_crate_root({:?}): found no mark (ident.span = {:?})",
                    ident, ident.span
                );
                return self.graph_root;
            }
        };
        let module = self.expect_module(
            module.opt_def_id().map_or(LOCAL_CRATE, |def_id| def_id.krate).as_def_id(),
        );
        debug!(
            "resolve_crate_root({:?}): got module {:?} ({:?}) (ident.span = {:?})",
            ident,
            module,
            module.kind.name(),
            ident.span
        );
        module
    }

    fn resolve_self(&mut self, ctxt: &mut SyntaxContext, module: Module<'a>) -> Module<'a> {
        let mut module = self.expect_module(module.nearest_parent_mod());
        while module.span.ctxt().normalize_to_macros_2_0() != *ctxt {
            let parent = module.parent.unwrap_or_else(|| self.expn_def_scope(ctxt.remove_mark()));
            module = self.expect_module(parent.nearest_parent_mod());
        }
        module
    }

    fn record_partial_res(&mut self, node_id: NodeId, resolution: PartialRes) {
        debug!("(recording res) recording {:?} for {}", resolution, node_id);
        if let Some(prev_res) = self.partial_res_map.insert(node_id, resolution) {
            panic!("path resolved multiple times ({:?} before, {:?} now)", prev_res, resolution);
        }
    }

    fn record_pat_span(&mut self, node: NodeId, span: Span) {
        debug!("(recording pat) recording {:?} for {:?}", node, span);
        self.pat_span_map.insert(node, span);
    }

    fn is_accessible_from(&self, vis: ty::Visibility, module: Module<'a>) -> bool {
        vis.is_accessible_from(module.nearest_parent_mod(), self)
    }

    fn set_binding_parent_module(&mut self, binding: &'a NameBinding<'a>, module: Module<'a>) {
        if let Some(old_module) =
            self.binding_parent_modules.insert(Interned::new_unchecked(binding), module)
        {
            if !ptr::eq(module, old_module) {
                span_bug!(binding.span, "parent module is reset for binding");
            }
        }
    }

    fn disambiguate_macro_rules_vs_modularized(
        &self,
        macro_rules: &'a NameBinding<'a>,
        modularized: &'a NameBinding<'a>,
    ) -> bool {
        // Some non-controversial subset of ambiguities "modularized macro name" vs "macro_rules"
        // is disambiguated to mitigate regressions from macro modularization.
        // Scoping for `macro_rules` behaves like scoping for `let` at module level, in general.
        match (
            self.binding_parent_modules.get(&Interned::new_unchecked(macro_rules)),
            self.binding_parent_modules.get(&Interned::new_unchecked(modularized)),
        ) {
            (Some(macro_rules), Some(modularized)) => {
                macro_rules.nearest_parent_mod() == modularized.nearest_parent_mod()
                    && modularized.is_ancestor_of(macro_rules)
            }
            _ => false,
        }
    }

    fn extern_prelude_get(&mut self, ident: Ident, finalize: bool) -> Option<&'a NameBinding<'a>> {
        if ident.is_path_segment_keyword() {
            // Make sure `self`, `super` etc produce an error when passed to here.
            return None;
        }
        self.extern_prelude.get(&ident.normalize_to_macros_2_0()).cloned().and_then(|entry| {
            if let Some(binding) = entry.extern_crate_item {
                if finalize && entry.introduced_by_item {
                    self.record_use(ident, binding, false);
                }
                Some(binding)
            } else {
                let crate_id = if finalize {
                    let Some(crate_id) =
                        self.crate_loader.process_path_extern(ident.name, ident.span) else { return Some(self.dummy_binding); };
                    crate_id
                } else {
                    self.crate_loader.maybe_process_path_extern(ident.name)?
                };
                let crate_root = self.expect_module(crate_id.as_def_id());
                Some(
                    (crate_root, ty::Visibility::Public, DUMMY_SP, LocalExpnId::ROOT)
                        .to_name_binding(self.arenas),
                )
            }
        })
    }

    /// Rustdoc uses this to resolve doc link paths in a recoverable way. `PathResult<'a>`
    /// isn't something that can be returned because it can't be made to live that long,
    /// and also it's a private type. Fortunately rustdoc doesn't need to know the error,
    /// just that an error occurred.
    pub fn resolve_rustdoc_path(
        &mut self,
        path_str: &str,
        ns: Namespace,
        mut parent_scope: ParentScope<'a>,
    ) -> Option<Res> {
        let mut segments =
            Vec::from_iter(path_str.split("::").map(Ident::from_str).map(Segment::from_ident));
        if let Some(segment) = segments.first_mut() {
            if segment.ident.name == kw::Crate {
                // FIXME: `resolve_path` always resolves `crate` to the current crate root, but
                // rustdoc wants it to resolve to the `parent_scope`'s crate root. This trick of
                // replacing `crate` with `self` and changing the current module should achieve
                // the same effect.
                segment.ident.name = kw::SelfLower;
                parent_scope.module =
                    self.expect_module(parent_scope.module.def_id().krate.as_def_id());
            } else if segment.ident.name == kw::Empty {
                segment.ident.name = kw::PathRoot;
            }
        }

        match self.maybe_resolve_path(&segments, Some(ns), &parent_scope) {
            PathResult::Module(ModuleOrUniformRoot::Module(module)) => Some(module.res().unwrap()),
            PathResult::NonModule(path_res) if path_res.unresolved_segments() == 0 => {
                Some(path_res.base_res())
            }
            PathResult::Module(ModuleOrUniformRoot::ExternPrelude)
            | PathResult::NonModule(..)
            | PathResult::Failed { .. } => None,
            PathResult::Module(..) | PathResult::Indeterminate => unreachable!(),
        }
    }

    /// For rustdoc.
    /// For local modules returns only reexports, for external modules returns all children.
    pub fn module_children_or_reexports(&self, def_id: DefId) -> Vec<ModChild> {
        if let Some(def_id) = def_id.as_local() {
            self.reexport_map.get(&def_id).cloned().unwrap_or_default()
        } else {
            self.cstore().module_children_untracked(def_id, self.session)
        }
    }

    /// For rustdoc.
    pub fn macro_rules_scope(&self, def_id: LocalDefId) -> (MacroRulesScopeRef<'a>, Res) {
        let scope = *self.macro_rules_scopes.get(&def_id).expect("not a `macro_rules` item");
        match scope.get() {
            MacroRulesScope::Binding(mb) => (scope, mb.binding.res()),
            _ => unreachable!(),
        }
    }

    /// Retrieves the span of the given `DefId` if `DefId` is in the local crate.
    #[inline]
    pub fn opt_span(&self, def_id: DefId) -> Option<Span> {
        def_id.as_local().map(|def_id| self.source_span[def_id])
    }

    /// Retrieves the name of the given `DefId`.
    #[inline]
    pub fn opt_name(&self, def_id: DefId) -> Option<Symbol> {
        let def_key = match def_id.as_local() {
            Some(def_id) => self.definitions.def_key(def_id),
            None => self.cstore().def_key(def_id),
        };
        def_key.get_opt_name()
    }

    /// Checks if an expression refers to a function marked with
    /// `#[rustc_legacy_const_generics]` and returns the argument index list
    /// from the attribute.
    pub fn legacy_const_generic_args(&mut self, expr: &Expr) -> Option<Vec<usize>> {
        if let ExprKind::Path(None, path) = &expr.kind {
            // Don't perform legacy const generics rewriting if the path already
            // has generic arguments.
            if path.segments.last().unwrap().args.is_some() {
                return None;
            }

            let partial_res = self.partial_res_map.get(&expr.id)?;
            if partial_res.unresolved_segments() != 0 {
                return None;
            }

            if let Res::Def(def::DefKind::Fn, def_id) = partial_res.base_res() {
                // We only support cross-crate argument rewriting. Uses
                // within the same crate should be updated to use the new
                // const generics style.
                if def_id.is_local() {
                    return None;
                }

                if let Some(v) = self.legacy_const_generic_args.get(&def_id) {
                    return v.clone();
                }

                let attr = self
                    .cstore()
                    .item_attrs_untracked(def_id, self.session)
                    .find(|a| a.has_name(sym::rustc_legacy_const_generics))?;
                let mut ret = Vec::new();
                for meta in attr.meta_item_list()? {
                    match meta.literal()?.kind {
                        LitKind::Int(a, _) => ret.push(a as usize),
                        _ => panic!("invalid arg index"),
                    }
                }
                // Cache the lookup to avoid parsing attributes for an iterm multiple times.
                self.legacy_const_generic_args.insert(def_id, Some(ret.clone()));
                return Some(ret);
            }
        }
        None
    }

    fn resolve_main(&mut self) {
        let module = self.graph_root;
        let ident = Ident::with_dummy_span(sym::main);
        let parent_scope = &ParentScope::module(module, self);

        let Ok(name_binding) = self.maybe_resolve_ident_in_module(
            ModuleOrUniformRoot::Module(module),
            ident,
            ValueNS,
            parent_scope,
        ) else {
            return;
        };

        let res = name_binding.res();
        let is_import = name_binding.is_import();
        let span = name_binding.span;
        if let Res::Def(DefKind::Fn, _) = res {
            self.record_use(ident, name_binding, false);
        }
        self.main_def = Some(MainDefinition { res, is_import, span });
    }
}

fn names_to_string(names: &[Symbol]) -> String {
    let mut result = String::new();
    for (i, name) in names.iter().filter(|name| **name != kw::PathRoot).enumerate() {
        if i > 0 {
            result.push_str("::");
        }
        if Ident::with_dummy_span(*name).is_raw_guess() {
            result.push_str("r#");
        }
        result.push_str(name.as_str());
    }
    result
}

fn path_names_to_string(path: &Path) -> String {
    names_to_string(&path.segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>())
}

/// A somewhat inefficient routine to obtain the name of a module.
fn module_to_string(module: Module<'_>) -> Option<String> {
    let mut names = Vec::new();

    fn collect_mod(names: &mut Vec<Symbol>, module: Module<'_>) {
        if let ModuleKind::Def(.., name) = module.kind {
            if let Some(parent) = module.parent {
                names.push(name);
                collect_mod(names, parent);
            }
        } else {
            names.push(Symbol::intern("<opaque>"));
            collect_mod(names, module.parent.unwrap());
        }
    }
    collect_mod(&mut names, module);

    if names.is_empty() {
        return None;
    }
    names.reverse();
    Some(names_to_string(&names))
}

#[derive(Copy, Clone, Debug)]
struct Finalize {
    /// Node ID for linting.
    node_id: NodeId,
    /// Span of the whole path or some its characteristic fragment.
    /// E.g. span of `b` in `foo::{a, b, c}`, or full span for regular paths.
    path_span: Span,
    /// Span of the path start, suitable for prepending something to to it.
    /// E.g. span of `foo` in `foo::{a, b, c}`, or full span for regular paths.
    root_span: Span,
    /// Whether to report privacy errors or silently return "no resolution" for them,
    /// similarly to speculative resolution.
    report_private: bool,
}

impl Finalize {
    fn new(node_id: NodeId, path_span: Span) -> Finalize {
        Finalize::with_root_span(node_id, path_span, path_span)
    }

    fn with_root_span(node_id: NodeId, path_span: Span, root_span: Span) -> Finalize {
        Finalize { node_id, path_span, root_span, report_private: true }
    }
}

pub fn provide(providers: &mut Providers) {
    late::lifetimes::provide(providers);
}
