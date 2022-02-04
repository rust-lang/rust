// ignore-tidy-filelength

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
#![feature(bool_to_option)]
#![feature(crate_visibility_modifier)]
#![feature(let_else)]
#![feature(never_type)]
#![feature(nll)]
#![recursion_limit = "256"]
#![allow(rustdoc::private_intra_doc_links)]
#![cfg_attr(not(bootstrap), allow(rustc::potential_query_instability))]

#[macro_use]
extern crate tracing;

pub use rustc_hir::def::{Namespace, PerNS};

use Determinacy::*;

use rustc_arena::{DroplessArena, TypedArena};
use rustc_ast::node_id::NodeMap;
use rustc_ast::ptr::P;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, NodeId};
use rustc_ast::{Crate, CRATE_NODE_ID};
use rustc_ast::{Expr, ExprKind, LitKind};
use rustc_ast::{ItemKind, ModKind, Path};
use rustc_ast_lowering::ResolverAstLowering;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_expand::base::{DeriveResolutions, SyntaxExtension, SyntaxExtensionKind};
use rustc_hir::def::Namespace::*;
use rustc_hir::def::{self, CtorOf, DefKind, NonMacroAttrKind, PartialRes};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefPathHash, LocalDefId};
use rustc_hir::def_id::{CRATE_DEF_ID, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPathData, Definitions};
use rustc_hir::TraitCandidate;
use rustc_index::vec::IndexVec;
use rustc_metadata::creader::{CStore, CrateLoader};
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::span_bug;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, DefIdTree, MainDefinition, RegisteredTools, ResolverOutputs};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::cstore::{CrateStore, MetadataLoaderDyn};
use rustc_session::lint;
use rustc_session::lint::{BuiltinLintDiagnostics, LintBuffer};
use rustc_session::Session;
use rustc_span::edition::Edition;
use rustc_span::hygiene::{ExpnId, ExpnKind, LocalExpnId, MacroKind, SyntaxContext, Transparency};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

use smallvec::{smallvec, SmallVec};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::{cmp, fmt, iter, mem, ptr};
use tracing::debug;

use diagnostics::{extend_span_to_previous_binding, find_span_of_binding_until_next_binding};
use diagnostics::{ImportSuggestion, LabelSuggestion, Suggestion};
use imports::{Import, ImportKind, ImportResolver, NameResolution};
use late::{ConstantItemKind, HasGenericParams, PathSource, Rib, RibKind::*};
use macros::{MacroRulesBinding, MacroRulesScope, MacroRulesScopeRef};

use crate::access_levels::AccessLevelsVisitor;

type Res = def::Res<NodeId>;

mod access_levels;
mod build_reduced_graph;
mod check_unused;
mod def_collector;
mod diagnostics;
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
    module: Module<'a>,
    expansion: LocalExpnId,
    macro_rules: MacroRulesScopeRef<'a>,
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
    MethodNotMemberOfTrait(Ident, &'a str, Option<Symbol>),
    /// Error E0437: type is not a member of trait.
    TypeNotMemberOfTrait(Ident, &'a str, Option<Symbol>),
    /// Error E0438: const is not a member of trait.
    ConstNotMemberOfTrait(Ident, &'a str, Option<Symbol>),
    /// Error E0408: variable `{}` is not bound in all patterns.
    VariableNotBoundInPattern(&'a BindingError),
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
        shadowing_binding_descr: &'static str,
        name: Symbol,
        participle: &'static str,
        article: &'static str,
        shadowed_binding_descr: &'static str,
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
}

impl Segment {
    fn from_path(path: &Path) -> Vec<Segment> {
        path.segments.iter().map(|s| s.into()).collect()
    }

    fn from_ident(ident: Ident) -> Segment {
        Segment { ident, id: None, has_generic_args: false }
    }

    fn names_to_string(segments: &[Segment]) -> String {
        names_to_string(&segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>())
    }
}

impl<'a> From<&'a ast::PathSegment> for Segment {
    fn from(seg: &'a ast::PathSegment) -> Segment {
        Segment { ident: seg.ident, id: Some(seg.id), has_generic_args: seg.args.is_some() }
    }
}

struct UsePlacementFinder {
    target_module: NodeId,
    span: Option<Span>,
    found_use: bool,
}

impl UsePlacementFinder {
    fn check(krate: &Crate, target_module: NodeId) -> (Option<Span>, bool) {
        let mut finder = UsePlacementFinder { target_module, span: None, found_use: false };
        if let ControlFlow::Continue(..) = finder.check_mod(&krate.items, CRATE_NODE_ID) {
            visit::walk_crate(&mut finder, krate);
        }
        (finder.span, finder.found_use)
    }

    fn check_mod(&mut self, items: &[P<ast::Item>], node_id: NodeId) -> ControlFlow<()> {
        if self.span.is_some() {
            return ControlFlow::Break(());
        }
        if node_id != self.target_module {
            return ControlFlow::Continue(());
        }
        // find a use statement
        for item in items {
            match item.kind {
                ItemKind::Use(..) => {
                    // don't suggest placing a use before the prelude
                    // import or other generated ones
                    if !item.span.from_expansion() {
                        self.span = Some(item.span.shrink_to_lo());
                        self.found_use = true;
                        return ControlFlow::Break(());
                    }
                }
                // don't place use before extern crate
                ItemKind::ExternCrate(_) => {}
                // but place them before the first other item
                _ => {
                    if self.span.map_or(true, |span| item.span < span)
                        && !item.span.from_expansion()
                    {
                        self.span = Some(item.span.shrink_to_lo());
                        // don't insert between attributes and an item
                        // find the first attribute on the item
                        // FIXME: This is broken for active attributes.
                        for attr in &item.attrs {
                            if !attr.span.is_dummy()
                                && self.span.map_or(true, |span| attr.span < span)
                            {
                                self.span = Some(attr.span.shrink_to_lo());
                            }
                        }
                    }
                }
            }
        }
        ControlFlow::Continue(())
    }
}

impl<'tcx> Visitor<'tcx> for UsePlacementFinder {
    fn visit_item(&mut self, item: &'tcx ast::Item) {
        if let ItemKind::Mod(_, ModKind::Loaded(items, ..)) = &item.kind {
            if let ControlFlow::Break(..) = self.check_mod(items, item.id) {
                return;
            }
        }
        visit::walk_item(self, item);
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
    Block(NodeId),
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
            ModuleKind::Block(..) => None,
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
    /// The identifier for the binding, aways the `normalize_to_macros_2_0` version of the
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
            ModuleKind::Block(_) => false,
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
    err: DiagnosticBuilder<'a>,
    /// Candidates which user could `use` to access the missing type.
    candidates: Vec<ImportSuggestion>,
    /// The `DefId` of the module to place the use-statements in.
    def_id: DefId,
    /// Whether the diagnostic should say "instead" (as in `consider importing ... instead`).
    instead: bool,
    /// Extra free-form suggestion.
    suggestion: Option<(Span, &'static str, String, Applicability)>,
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
            }) => def_id.index == CRATE_DEF_INDEX,
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

    fn is_macro_def(&self) -> bool {
        matches!(self.kind, NameBindingKind::Res(Res::Def(DefKind::Macro(..), _), _))
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

/// The main resolver class.
///
/// This is the visitor that walks the whole crate.
pub struct Resolver<'a> {
    session: &'a Session,

    definitions: Definitions,

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

    /// FIXME: Refactor things so that these fields are passed through arguments and not resolver.
    /// We are resolving a last import segment during import validation.
    last_import_segment: bool,
    /// This binding should be ignored during in-module resolution, so that we don't get
    /// "self-confirming" import resolutions during import validation.
    unusable_binding: Option<&'a NameBinding<'a>>,

    // Spans for local variables found during pattern resolution.
    // Used for suggestions during error reporting.
    pat_span_map: NodeMap<Span>,

    /// Resolutions for nodes that have a single resolution.
    partial_res_map: NodeMap<PartialRes>,
    /// Resolutions for import nodes, which have multiple resolutions in different namespaces.
    import_res_map: NodeMap<PerNS<Option<Res>>>,
    /// Resolutions for labels (node IDs of their corresponding blocks or loops).
    label_res_map: NodeMap<NodeId>,

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
    used_imports: FxHashSet<NodeId>,
    maybe_unused_trait_imports: FxHashSet<LocalDefId>,
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
    registered_attrs: FxHashSet<Ident>,
    registered_tools: RegisteredTools,
    macro_use_prelude: FxHashMap<Symbol, &'a NameBinding<'a>>,
    all_macros: FxHashMap<Symbol, Res>,
    macro_map: FxHashMap<DefId, Lrc<SyntaxExtension>>,
    dummy_ext_bang: Lrc<SyntaxExtension>,
    dummy_ext_derive: Lrc<SyntaxExtension>,
    non_macro_attr: Lrc<SyntaxExtension>,
    local_macro_def_scopes: FxHashMap<LocalDefId, Module<'a>>,
    ast_transform_scopes: FxHashMap<LocalExpnId, Module<'a>>,
    unused_macros: FxHashMap<LocalDefId, (NodeId, Ident)>,
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

    next_disambiguator: FxHashMap<(LocalDefId, DefPathData), u32>,
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
    fn parent(self, id: DefId) -> Option<DefId> {
        match id.as_local() {
            Some(id) => self.definitions.def_key(id).parent,
            None => self.cstore().def_key(id).parent,
        }
        .map(|index| DefId { index, ..id })
    }
}

/// This interface is used through the AST→HIR step, to embed full paths into the HIR. After that
/// the resolver is no longer needed as all the relevant information is inline.
impl ResolverAstLowering for Resolver<'_> {
    fn def_key(&mut self, id: DefId) -> DefKey {
        if let Some(id) = id.as_local() {
            self.definitions().def_key(id)
        } else {
            self.cstore().def_key(id)
        }
    }

    #[inline]
    fn def_span(&self, id: LocalDefId) -> Span {
        self.definitions.def_span(id)
    }

    fn item_generics_num_lifetimes(&self, def_id: DefId) -> usize {
        if let Some(def_id) = def_id.as_local() {
            self.item_generics_num_lifetimes[&def_id]
        } else {
            self.cstore().item_generics_num_lifetimes(def_id, self.session)
        }
    }

    fn legacy_const_generic_args(&mut self, expr: &Expr) -> Option<Vec<usize>> {
        self.legacy_const_generic_args(expr)
    }

    fn get_partial_res(&self, id: NodeId) -> Option<PartialRes> {
        self.partial_res_map.get(&id).cloned()
    }

    fn get_import_res(&mut self, id: NodeId) -> PerNS<Option<Res>> {
        self.import_res_map.get(&id).cloned().unwrap_or_default()
    }

    fn get_label_res(&mut self, id: NodeId) -> Option<NodeId> {
        self.label_res_map.get(&id).cloned()
    }

    fn definitions(&mut self) -> &mut Definitions {
        &mut self.definitions
    }

    fn create_stable_hashing_context(&self) -> StableHashingContext<'_> {
        StableHashingContext::new(self.session, &self.definitions, self.crate_loader.cstore())
    }

    fn lint_buffer(&mut self) -> &mut LintBuffer {
        &mut self.lint_buffer
    }

    fn next_node_id(&mut self) -> NodeId {
        self.next_node_id()
    }

    fn take_trait_map(&mut self, node: NodeId) -> Option<Vec<TraitCandidate>> {
        self.trait_map.remove(&node)
    }

    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.node_id_to_def_id.get(&node).copied()
    }

    fn local_def_id(&self, node: NodeId) -> LocalDefId {
        self.opt_local_def_id(node).unwrap_or_else(|| panic!("no entry for node id: `{:?}`", node))
    }

    fn def_path_hash(&self, def_id: DefId) -> DefPathHash {
        match def_id.as_local() {
            Some(def_id) => self.definitions.def_path_hash(def_id),
            None => self.cstore().def_path_hash(def_id),
        }
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

        // Find the next free disambiguator for this key.
        let next_disambiguator = &mut self.next_disambiguator;
        let next_disambiguator = |parent, data| {
            let next_disamb = next_disambiguator.entry((parent, data)).or_insert(0);
            let disambiguator = *next_disamb;
            *next_disamb = next_disamb.checked_add(1).expect("disambiguator overflow");
            disambiguator
        };

        let def_id = self.definitions.create_def(parent, data, expn_id, next_disambiguator, span);

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
            krate.span,
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

        let definitions = Definitions::new(session.local_stable_crate_id(), krate.span);
        let root = definitions.get_root_def();

        let mut visibilities = FxHashMap::default();
        visibilities.insert(CRATE_DEF_ID, ty::Visibility::Public);

        let mut def_id_to_node_id = IndexVec::default();
        assert_eq!(def_id_to_node_id.push(CRATE_NODE_ID), root);
        let mut node_id_to_def_id = FxHashMap::default();
        node_id_to_def_id.insert(CRATE_NODE_ID, root);

        let mut invocation_parents = FxHashMap::default();
        invocation_parents.insert(LocalExpnId::ROOT, (root, ImplTraitContext::Existential));

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

            // The outermost module has def ID 0; this is not reflected in the
            // AST.
            graph_root,
            prelude: None,
            extern_prelude,

            has_self: FxHashSet::default(),
            field_names: FxHashMap::default(),

            determined_imports: Vec::new(),
            indeterminate_imports: Vec::new(),

            last_import_segment: false,
            unusable_binding: None,

            pat_span_map: Default::default(),
            partial_res_map: Default::default(),
            import_res_map: Default::default(),
            label_res_map: Default::default(),
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
            registered_attrs,
            registered_tools,
            macro_use_prelude: FxHashMap::default(),
            all_macros: FxHashMap::default(),
            macro_map: FxHashMap::default(),
            dummy_ext_bang: Lrc::new(SyntaxExtension::dummy_bang(session.edition())),
            dummy_ext_derive: Lrc::new(SyntaxExtension::dummy_derive(session.edition())),
            non_macro_attr: Lrc::new(SyntaxExtension::non_macro_attr(session.edition())),
            invocation_parent_scopes: Default::default(),
            output_macro_rules_scopes: Default::default(),
            helper_attrs: Default::default(),
            derive_data: Default::default(),
            local_macro_def_scopes: FxHashMap::default(),
            name_already_seen: FxHashMap::default(),
            potentially_unused_imports: Vec::new(),
            struct_constructors: Default::default(),
            unused_macros: Default::default(),
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
            next_disambiguator: Default::default(),
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
        let next =
            self.next_node_id.as_u32().checked_add(1).expect("input too large; ran out of NodeIds");
        mem::replace(&mut self.next_node_id, ast::NodeId::from_u32(next))
    }

    pub fn lint_buffer(&mut self) -> &mut LintBuffer {
        &mut self.lint_buffer
    }

    pub fn arenas() -> ResolverArenas<'a> {
        Default::default()
    }

    pub fn into_outputs(self) -> ResolverOutputs {
        let proc_macros = self.proc_macros.iter().map(|id| self.local_def_id(*id)).collect();
        let definitions = self.definitions;
        let visibilities = self.visibilities;
        let extern_crate_map = self.extern_crate_map;
        let reexport_map = self.reexport_map;
        let maybe_unused_trait_imports = self.maybe_unused_trait_imports;
        let maybe_unused_extern_crates = self.maybe_unused_extern_crates;
        let glob_map = self.glob_map;
        let main_def = self.main_def;
        let confused_type_with_std_module = self.confused_type_with_std_module;
        let access_levels = self.access_levels;
        ResolverOutputs {
            definitions,
            cstore: Box::new(self.crate_loader.into_cstore()),
            visibilities,
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
        }
    }

    pub fn clone_outputs(&self) -> ResolverOutputs {
        let proc_macros = self.proc_macros.iter().map(|id| self.local_def_id(*id)).collect();
        ResolverOutputs {
            definitions: self.definitions.clone(),
            access_levels: self.access_levels.clone(),
            cstore: Box::new(self.cstore().clone()),
            visibilities: self.visibilities.clone(),
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
        }
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
        self.get_macro(res).map_or(false, |ext| ext.builtin_name.is_some())
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

    /// A generic scope visitor.
    /// Visits scopes in order to resolve some identifier in them or perform other actions.
    /// If the callback returns `Some` result, we stop visiting scopes and return it.
    fn visit_scopes<T>(
        &mut self,
        scope_set: ScopeSet<'a>,
        parent_scope: &ParentScope<'a>,
        ctxt: SyntaxContext,
        mut visitor: impl FnMut(
            &mut Self,
            Scope<'a>,
            /*use_prelude*/ bool,
            SyntaxContext,
        ) -> Option<T>,
    ) -> Option<T> {
        // General principles:
        // 1. Not controlled (user-defined) names should have higher priority than controlled names
        //    built into the language or standard library. This way we can add new names into the
        //    language or standard library without breaking user code.
        // 2. "Closed set" below means new names cannot appear after the current resolution attempt.
        // Places to search (in order of decreasing priority):
        // (Type NS)
        // 1. FIXME: Ribs (type parameters), there's no necessary infrastructure yet
        //    (open set, not controlled).
        // 2. Names in modules (both normal `mod`ules and blocks), loop through hygienic parents
        //    (open, not controlled).
        // 3. Extern prelude (open, the open part is from macro expansions, not controlled).
        // 4. Tool modules (closed, controlled right now, but not in the future).
        // 5. Standard library prelude (de-facto closed, controlled).
        // 6. Language prelude (closed, controlled).
        // (Value NS)
        // 1. FIXME: Ribs (local variables), there's no necessary infrastructure yet
        //    (open set, not controlled).
        // 2. Names in modules (both normal `mod`ules and blocks), loop through hygienic parents
        //    (open, not controlled).
        // 3. Standard library prelude (de-facto closed, controlled).
        // (Macro NS)
        // 1-3. Derive helpers (open, not controlled). All ambiguities with other names
        //    are currently reported as errors. They should be higher in priority than preludes
        //    and probably even names in modules according to the "general principles" above. They
        //    also should be subject to restricted shadowing because are effectively produced by
        //    derives (you need to resolve the derive first to add helpers into scope), but they
        //    should be available before the derive is expanded for compatibility.
        //    It's mess in general, so we are being conservative for now.
        // 1-3. `macro_rules` (open, not controlled), loop through `macro_rules` scopes. Have higher
        //    priority than prelude macros, but create ambiguities with macros in modules.
        // 1-3. Names in modules (both normal `mod`ules and blocks), loop through hygienic parents
        //    (open, not controlled). Have higher priority than prelude macros, but create
        //    ambiguities with `macro_rules`.
        // 4. `macro_use` prelude (open, the open part is from macro expansions, not controlled).
        // 4a. User-defined prelude from macro-use
        //    (open, the open part is from macro expansions, not controlled).
        // 4b. "Standard library prelude" part implemented through `macro-use` (closed, controlled).
        // 4c. Standard library prelude (de-facto closed, controlled).
        // 6. Language prelude: builtin attributes (closed, controlled).

        let rust_2015 = ctxt.edition() == Edition::Edition2015;
        let (ns, macro_kind, is_absolute_path) = match scope_set {
            ScopeSet::All(ns, _) => (ns, None, false),
            ScopeSet::AbsolutePath(ns) => (ns, None, true),
            ScopeSet::Macro(macro_kind) => (MacroNS, Some(macro_kind), false),
            ScopeSet::Late(ns, ..) => (ns, None, false),
        };
        let module = match scope_set {
            // Start with the specified module.
            ScopeSet::Late(_, module, _) => module,
            // Jump out of trait or enum modules, they do not act as scopes.
            _ => parent_scope.module.nearest_item_scope(),
        };
        let mut scope = match ns {
            _ if is_absolute_path => Scope::CrateRoot,
            TypeNS | ValueNS => Scope::Module(module, None),
            MacroNS => Scope::DeriveHelpers(parent_scope.expansion),
        };
        let mut ctxt = ctxt.normalize_to_macros_2_0();
        let mut use_prelude = !module.no_implicit_prelude;

        loop {
            let visit = match scope {
                // Derive helpers are not in scope when resolving derives in the same container.
                Scope::DeriveHelpers(expn_id) => {
                    !(expn_id == parent_scope.expansion && macro_kind == Some(MacroKind::Derive))
                }
                Scope::DeriveHelpersCompat => true,
                Scope::MacroRules(macro_rules_scope) => {
                    // Use "path compression" on `macro_rules` scope chains. This is an optimization
                    // used to avoid long scope chains, see the comments on `MacroRulesScopeRef`.
                    // As another consequence of this optimization visitors never observe invocation
                    // scopes for macros that were already expanded.
                    while let MacroRulesScope::Invocation(invoc_id) = macro_rules_scope.get() {
                        if let Some(next_scope) = self.output_macro_rules_scopes.get(&invoc_id) {
                            macro_rules_scope.set(next_scope.get());
                        } else {
                            break;
                        }
                    }
                    true
                }
                Scope::CrateRoot => true,
                Scope::Module(..) => true,
                Scope::RegisteredAttrs => use_prelude,
                Scope::MacroUsePrelude => use_prelude || rust_2015,
                Scope::BuiltinAttrs => true,
                Scope::ExternPrelude => use_prelude || is_absolute_path,
                Scope::ToolPrelude => use_prelude,
                Scope::StdLibPrelude => use_prelude || ns == MacroNS,
                Scope::BuiltinTypes => true,
            };

            if visit {
                if let break_result @ Some(..) = visitor(self, scope, use_prelude, ctxt) {
                    return break_result;
                }
            }

            scope = match scope {
                Scope::DeriveHelpers(LocalExpnId::ROOT) => Scope::DeriveHelpersCompat,
                Scope::DeriveHelpers(expn_id) => {
                    // Derive helpers are not visible to code generated by bang or derive macros.
                    let expn_data = expn_id.expn_data();
                    match expn_data.kind {
                        ExpnKind::Root
                        | ExpnKind::Macro(MacroKind::Bang | MacroKind::Derive, _) => {
                            Scope::DeriveHelpersCompat
                        }
                        _ => Scope::DeriveHelpers(expn_data.parent.expect_local()),
                    }
                }
                Scope::DeriveHelpersCompat => Scope::MacroRules(parent_scope.macro_rules),
                Scope::MacroRules(macro_rules_scope) => match macro_rules_scope.get() {
                    MacroRulesScope::Binding(binding) => {
                        Scope::MacroRules(binding.parent_macro_rules_scope)
                    }
                    MacroRulesScope::Invocation(invoc_id) => {
                        Scope::MacroRules(self.invocation_parent_scopes[&invoc_id].macro_rules)
                    }
                    MacroRulesScope::Empty => Scope::Module(module, None),
                },
                Scope::CrateRoot => match ns {
                    TypeNS => {
                        ctxt.adjust(ExpnId::root());
                        Scope::ExternPrelude
                    }
                    ValueNS | MacroNS => break,
                },
                Scope::Module(module, prev_lint_id) => {
                    use_prelude = !module.no_implicit_prelude;
                    let derive_fallback_lint_id = match scope_set {
                        ScopeSet::Late(.., lint_id) => lint_id,
                        _ => None,
                    };
                    match self.hygienic_lexical_parent(module, &mut ctxt, derive_fallback_lint_id) {
                        Some((parent_module, lint_id)) => {
                            Scope::Module(parent_module, lint_id.or(prev_lint_id))
                        }
                        None => {
                            ctxt.adjust(ExpnId::root());
                            match ns {
                                TypeNS => Scope::ExternPrelude,
                                ValueNS => Scope::StdLibPrelude,
                                MacroNS => Scope::RegisteredAttrs,
                            }
                        }
                    }
                }
                Scope::RegisteredAttrs => Scope::MacroUsePrelude,
                Scope::MacroUsePrelude => Scope::StdLibPrelude,
                Scope::BuiltinAttrs => break, // nowhere else to search
                Scope::ExternPrelude if is_absolute_path => break,
                Scope::ExternPrelude => Scope::ToolPrelude,
                Scope::ToolPrelude => Scope::StdLibPrelude,
                Scope::StdLibPrelude => match ns {
                    TypeNS => Scope::BuiltinTypes,
                    ValueNS => break, // nowhere else to search
                    MacroNS => Scope::BuiltinAttrs,
                },
                Scope::BuiltinTypes => break, // nowhere else to search
            };
        }

        None
    }

    /// This resolves the identifier `ident` in the namespace `ns` in the current lexical scope.
    /// More specifically, we proceed up the hierarchy of scopes and return the binding for
    /// `ident` in the first scope that defines it (or None if no scopes define it).
    ///
    /// A block's items are above its local variables in the scope hierarchy, regardless of where
    /// the items are defined in the block. For example,
    /// ```rust
    /// fn f() {
    ///    g(); // Since there are no local variables in scope yet, this resolves to the item.
    ///    let g = || {};
    ///    fn g() {}
    ///    g(); // This resolves to the local variable `g` since it shadows the item.
    /// }
    /// ```
    ///
    /// Invariant: This must only be called during main resolution, not during
    /// import resolution.
    fn resolve_ident_in_lexical_scope(
        &mut self,
        mut ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'a>,
        record_used_id: Option<NodeId>,
        path_span: Span,
        ribs: &[Rib<'a>],
    ) -> Option<LexicalScopeBinding<'a>> {
        assert!(ns == TypeNS || ns == ValueNS);
        let orig_ident = ident;
        if ident.name == kw::Empty {
            return Some(LexicalScopeBinding::Res(Res::Err));
        }
        let (general_span, normalized_span) = if ident.name == kw::SelfUpper {
            // FIXME(jseyfried) improve `Self` hygiene
            let empty_span = ident.span.with_ctxt(SyntaxContext::root());
            (empty_span, empty_span)
        } else if ns == TypeNS {
            let normalized_span = ident.span.normalize_to_macros_2_0();
            (normalized_span, normalized_span)
        } else {
            (ident.span.normalize_to_macro_rules(), ident.span.normalize_to_macros_2_0())
        };
        ident.span = general_span;
        let normalized_ident = Ident { span: normalized_span, ..ident };

        // Walk backwards up the ribs in scope.
        let record_used = record_used_id.is_some();
        let mut module = self.graph_root;
        for i in (0..ribs.len()).rev() {
            debug!("walk rib\n{:?}", ribs[i].bindings);
            // Use the rib kind to determine whether we are resolving parameters
            // (macro 2.0 hygiene) or local variables (`macro_rules` hygiene).
            let rib_ident = if ribs[i].kind.contains_params() { normalized_ident } else { ident };
            if let Some((original_rib_ident_def, res)) = ribs[i].bindings.get_key_value(&rib_ident)
            {
                // The ident resolves to a type parameter or local variable.
                return Some(LexicalScopeBinding::Res(self.validate_res_from_ribs(
                    i,
                    rib_ident,
                    *res,
                    record_used,
                    path_span,
                    *original_rib_ident_def,
                    ribs,
                )));
            }

            module = match ribs[i].kind {
                ModuleRibKind(module) => module,
                MacroDefinition(def) if def == self.macro_def(ident.span.ctxt()) => {
                    // If an invocation of this macro created `ident`, give up on `ident`
                    // and switch to `ident`'s source from the macro definition.
                    ident.span.remove_mark();
                    continue;
                }
                _ => continue,
            };

            match module.kind {
                ModuleKind::Block(..) => {} // We can see through blocks
                _ => break,
            }

            let item = self.resolve_ident_in_module_unadjusted(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                parent_scope,
                record_used,
                path_span,
            );
            if let Ok(binding) = item {
                // The ident resolves to an item.
                return Some(LexicalScopeBinding::Item(binding));
            }
        }
        self.early_resolve_ident_in_lexical_scope(
            orig_ident,
            ScopeSet::Late(ns, module, record_used_id),
            parent_scope,
            record_used,
            record_used,
            path_span,
        )
        .ok()
        .map(LexicalScopeBinding::Item)
    }

    fn hygienic_lexical_parent(
        &mut self,
        module: Module<'a>,
        ctxt: &mut SyntaxContext,
        derive_fallback_lint_id: Option<NodeId>,
    ) -> Option<(Module<'a>, Option<NodeId>)> {
        if !module.expansion.outer_expn_is_descendant_of(*ctxt) {
            return Some((self.expn_def_scope(ctxt.remove_mark()), None));
        }

        if let ModuleKind::Block(..) = module.kind {
            return Some((module.parent.unwrap().nearest_item_scope(), None));
        }

        // We need to support the next case under a deprecation warning
        // ```
        // struct MyStruct;
        // ---- begin: this comes from a proc macro derive
        // mod implementation_details {
        //     // Note that `MyStruct` is not in scope here.
        //     impl SomeTrait for MyStruct { ... }
        // }
        // ---- end
        // ```
        // So we have to fall back to the module's parent during lexical resolution in this case.
        if derive_fallback_lint_id.is_some() {
            if let Some(parent) = module.parent {
                // Inner module is inside the macro, parent module is outside of the macro.
                if module.expansion != parent.expansion
                    && module.expansion.is_descendant_of(parent.expansion)
                {
                    // The macro is a proc macro derive
                    if let Some(def_id) = module.expansion.expn_data().macro_def_id {
                        let ext = self.get_macro_by_def_id(def_id);
                        if ext.builtin_name.is_none()
                            && ext.macro_kind() == MacroKind::Derive
                            && parent.expansion.outer_expn_is_descendant_of(*ctxt)
                        {
                            return Some((parent, derive_fallback_lint_id));
                        }
                    }
                }
            }
        }

        None
    }

    fn resolve_ident_in_module(
        &mut self,
        module: ModuleOrUniformRoot<'a>,
        ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'a>,
        record_used: bool,
        path_span: Span,
    ) -> Result<&'a NameBinding<'a>, Determinacy> {
        self.resolve_ident_in_module_ext(module, ident, ns, parent_scope, record_used, path_span)
            .map_err(|(determinacy, _)| determinacy)
    }

    fn resolve_ident_in_module_ext(
        &mut self,
        module: ModuleOrUniformRoot<'a>,
        mut ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'a>,
        record_used: bool,
        path_span: Span,
    ) -> Result<&'a NameBinding<'a>, (Determinacy, Weak)> {
        let tmp_parent_scope;
        let mut adjusted_parent_scope = parent_scope;
        match module {
            ModuleOrUniformRoot::Module(m) => {
                if let Some(def) = ident.span.normalize_to_macros_2_0_and_adjust(m.expansion) {
                    tmp_parent_scope =
                        ParentScope { module: self.expn_def_scope(def), ..*parent_scope };
                    adjusted_parent_scope = &tmp_parent_scope;
                }
            }
            ModuleOrUniformRoot::ExternPrelude => {
                ident.span.normalize_to_macros_2_0_and_adjust(ExpnId::root());
            }
            ModuleOrUniformRoot::CrateRootAndExternPrelude | ModuleOrUniformRoot::CurrentScope => {
                // No adjustments
            }
        }
        self.resolve_ident_in_module_unadjusted_ext(
            module,
            ident,
            ns,
            adjusted_parent_scope,
            false,
            record_used,
            path_span,
        )
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

    fn resolve_path(
        &mut self,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        parent_scope: &ParentScope<'a>,
        record_used: bool,
        path_span: Span,
        crate_lint: CrateLint,
    ) -> PathResult<'a> {
        self.resolve_path_with_ribs(
            path,
            opt_ns,
            parent_scope,
            record_used,
            path_span,
            crate_lint,
            None,
        )
    }

    fn resolve_path_with_ribs(
        &mut self,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        parent_scope: &ParentScope<'a>,
        record_used: bool,
        path_span: Span,
        crate_lint: CrateLint,
        ribs: Option<&PerNS<Vec<Rib<'a>>>>,
    ) -> PathResult<'a> {
        let mut module = None;
        let mut allow_super = true;
        let mut second_binding = None;

        debug!(
            "resolve_path(path={:?}, opt_ns={:?}, record_used={:?}, \
             path_span={:?}, crate_lint={:?})",
            path, opt_ns, record_used, path_span, crate_lint,
        );

        for (i, &Segment { ident, id, has_generic_args: _ }) in path.iter().enumerate() {
            debug!("resolve_path ident {} {:?} {:?}", i, ident, id);
            let record_segment_res = |this: &mut Self, res| {
                if record_used {
                    if let Some(id) = id {
                        if !this.partial_res_map.contains_key(&id) {
                            assert!(id != ast::DUMMY_NODE_ID, "Trying to resolve dummy id");
                            this.record_partial_res(id, PartialRes::new(res));
                        }
                    }
                }
            };

            let is_last = i == path.len() - 1;
            let ns = if is_last { opt_ns.unwrap_or(TypeNS) } else { TypeNS };
            let name = ident.name;

            allow_super &= ns == TypeNS && (name == kw::SelfLower || name == kw::Super);

            if ns == TypeNS {
                if allow_super && name == kw::Super {
                    let mut ctxt = ident.span.ctxt().normalize_to_macros_2_0();
                    let self_module = match i {
                        0 => Some(self.resolve_self(&mut ctxt, parent_scope.module)),
                        _ => match module {
                            Some(ModuleOrUniformRoot::Module(module)) => Some(module),
                            _ => None,
                        },
                    };
                    if let Some(self_module) = self_module {
                        if let Some(parent) = self_module.parent {
                            module = Some(ModuleOrUniformRoot::Module(
                                self.resolve_self(&mut ctxt, parent),
                            ));
                            continue;
                        }
                    }
                    let msg = "there are too many leading `super` keywords".to_string();
                    return PathResult::Failed {
                        span: ident.span,
                        label: msg,
                        suggestion: None,
                        is_error_from_last_segment: false,
                    };
                }
                if i == 0 {
                    if name == kw::SelfLower {
                        let mut ctxt = ident.span.ctxt().normalize_to_macros_2_0();
                        module = Some(ModuleOrUniformRoot::Module(
                            self.resolve_self(&mut ctxt, parent_scope.module),
                        ));
                        continue;
                    }
                    if name == kw::PathRoot && ident.span.rust_2018() {
                        module = Some(ModuleOrUniformRoot::ExternPrelude);
                        continue;
                    }
                    if name == kw::PathRoot && ident.span.rust_2015() && self.session.rust_2018() {
                        // `::a::b` from 2015 macro on 2018 global edition
                        module = Some(ModuleOrUniformRoot::CrateRootAndExternPrelude);
                        continue;
                    }
                    if name == kw::PathRoot || name == kw::Crate || name == kw::DollarCrate {
                        // `::a::b`, `crate::a::b` or `$crate::a::b`
                        module = Some(ModuleOrUniformRoot::Module(self.resolve_crate_root(ident)));
                        continue;
                    }
                }
            }

            // Report special messages for path segment keywords in wrong positions.
            if ident.is_path_segment_keyword() && i != 0 {
                let name_str = if name == kw::PathRoot {
                    "crate root".to_string()
                } else {
                    format!("`{}`", name)
                };
                let label = if i == 1 && path[0].ident.name == kw::PathRoot {
                    format!("global paths cannot start with {}", name_str)
                } else {
                    format!("{} in paths can only be used in start position", name_str)
                };
                return PathResult::Failed {
                    span: ident.span,
                    label,
                    suggestion: None,
                    is_error_from_last_segment: false,
                };
            }

            enum FindBindingResult<'a> {
                Binding(Result<&'a NameBinding<'a>, Determinacy>),
                PathResult(PathResult<'a>),
            }
            let find_binding_in_ns = |this: &mut Self, ns| {
                let binding = if let Some(module) = module {
                    this.resolve_ident_in_module(
                        module,
                        ident,
                        ns,
                        parent_scope,
                        record_used,
                        path_span,
                    )
                } else if ribs.is_none() || opt_ns.is_none() || opt_ns == Some(MacroNS) {
                    let scopes = ScopeSet::All(ns, opt_ns.is_none());
                    this.early_resolve_ident_in_lexical_scope(
                        ident,
                        scopes,
                        parent_scope,
                        record_used,
                        record_used,
                        path_span,
                    )
                } else {
                    let record_used_id = if record_used {
                        crate_lint.node_id().or(Some(CRATE_NODE_ID))
                    } else {
                        None
                    };
                    match this.resolve_ident_in_lexical_scope(
                        ident,
                        ns,
                        parent_scope,
                        record_used_id,
                        path_span,
                        &ribs.unwrap()[ns],
                    ) {
                        // we found a locally-imported or available item/module
                        Some(LexicalScopeBinding::Item(binding)) => Ok(binding),
                        // we found a local variable or type param
                        Some(LexicalScopeBinding::Res(res))
                            if opt_ns == Some(TypeNS) || opt_ns == Some(ValueNS) =>
                        {
                            record_segment_res(this, res);
                            return FindBindingResult::PathResult(PathResult::NonModule(
                                PartialRes::with_unresolved_segments(res, path.len() - 1),
                            ));
                        }
                        _ => Err(Determinacy::determined(record_used)),
                    }
                };
                FindBindingResult::Binding(binding)
            };
            let binding = match find_binding_in_ns(self, ns) {
                FindBindingResult::PathResult(x) => return x,
                FindBindingResult::Binding(binding) => binding,
            };
            match binding {
                Ok(binding) => {
                    if i == 1 {
                        second_binding = Some(binding);
                    }
                    let res = binding.res();
                    let maybe_assoc = opt_ns != Some(MacroNS) && PathSource::Type.is_expected(res);
                    if let Some(next_module) = binding.module() {
                        module = Some(ModuleOrUniformRoot::Module(next_module));
                        record_segment_res(self, res);
                    } else if res == Res::ToolMod && i + 1 != path.len() {
                        if binding.is_import() {
                            self.session
                                .struct_span_err(
                                    ident.span,
                                    "cannot use a tool module through an import",
                                )
                                .span_note(binding.span, "the tool module imported here")
                                .emit();
                        }
                        let res = Res::NonMacroAttr(NonMacroAttrKind::Tool);
                        return PathResult::NonModule(PartialRes::new(res));
                    } else if res == Res::Err {
                        return PathResult::NonModule(PartialRes::new(Res::Err));
                    } else if opt_ns.is_some() && (is_last || maybe_assoc) {
                        self.lint_if_path_starts_with_module(
                            crate_lint,
                            path,
                            path_span,
                            second_binding,
                        );
                        return PathResult::NonModule(PartialRes::with_unresolved_segments(
                            res,
                            path.len() - i - 1,
                        ));
                    } else {
                        let label = format!(
                            "`{}` is {} {}, not a module",
                            ident,
                            res.article(),
                            res.descr(),
                        );

                        return PathResult::Failed {
                            span: ident.span,
                            label,
                            suggestion: None,
                            is_error_from_last_segment: is_last,
                        };
                    }
                }
                Err(Undetermined) => return PathResult::Indeterminate,
                Err(Determined) => {
                    if let Some(ModuleOrUniformRoot::Module(module)) = module {
                        if opt_ns.is_some() && !module.is_normal() {
                            return PathResult::NonModule(PartialRes::with_unresolved_segments(
                                module.res().unwrap(),
                                path.len() - i,
                            ));
                        }
                    }
                    let module_res = match module {
                        Some(ModuleOrUniformRoot::Module(module)) => module.res(),
                        _ => None,
                    };
                    let (label, suggestion) = if module_res == self.graph_root.res() {
                        let is_mod = |res| matches!(res, Res::Def(DefKind::Mod, _));
                        // Don't look up import candidates if this is a speculative resolve
                        let mut candidates = if record_used {
                            self.lookup_import_candidates(ident, TypeNS, parent_scope, is_mod)
                        } else {
                            Vec::new()
                        };
                        candidates.sort_by_cached_key(|c| {
                            (c.path.segments.len(), pprust::path_to_string(&c.path))
                        });
                        if let Some(candidate) = candidates.get(0) {
                            (
                                String::from("unresolved import"),
                                Some((
                                    vec![(ident.span, pprust::path_to_string(&candidate.path))],
                                    String::from("a similar path exists"),
                                    Applicability::MaybeIncorrect,
                                )),
                            )
                        } else if self.session.edition() == Edition::Edition2015 {
                            (format!("maybe a missing crate `{}`?", ident), None)
                        } else {
                            (format!("could not find `{}` in the crate root", ident), None)
                        }
                    } else if i == 0 {
                        if ident
                            .name
                            .as_str()
                            .chars()
                            .next()
                            .map_or(false, |c| c.is_ascii_uppercase())
                        {
                            // Check whether the name refers to an item in the value namespace.
                            let suggestion = if ribs.is_some() {
                                let match_span = match self.resolve_ident_in_lexical_scope(
                                    ident,
                                    ValueNS,
                                    parent_scope,
                                    None,
                                    path_span,
                                    &ribs.unwrap()[ValueNS],
                                ) {
                                    // Name matches a local variable. For example:
                                    // ```
                                    // fn f() {
                                    //     let Foo: &str = "";
                                    //     println!("{}", Foo::Bar); // Name refers to local
                                    //                               // variable `Foo`.
                                    // }
                                    // ```
                                    Some(LexicalScopeBinding::Res(Res::Local(id))) => {
                                        Some(*self.pat_span_map.get(&id).unwrap())
                                    }

                                    // Name matches item from a local name binding
                                    // created by `use` declaration. For example:
                                    // ```
                                    // pub Foo: &str = "";
                                    //
                                    // mod submod {
                                    //     use super::Foo;
                                    //     println!("{}", Foo::Bar); // Name refers to local
                                    //                               // binding `Foo`.
                                    // }
                                    // ```
                                    Some(LexicalScopeBinding::Item(name_binding)) => {
                                        Some(name_binding.span)
                                    }
                                    _ => None,
                                };

                                if let Some(span) = match_span {
                                    Some((
                                        vec![(span, String::from(""))],
                                        format!("`{}` is defined here, but is not a type", ident),
                                        Applicability::MaybeIncorrect,
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            (format!("use of undeclared type `{}`", ident), suggestion)
                        } else {
                            (
                                format!("use of undeclared crate or module `{}`", ident),
                                if ident.name == sym::alloc {
                                    Some((
                                        vec![],
                                        String::from(
                                            "add `extern crate alloc` to use the `alloc` crate",
                                        ),
                                        Applicability::MaybeIncorrect,
                                    ))
                                } else {
                                    self.find_similarly_named_module_or_crate(
                                        ident.name,
                                        &parent_scope.module,
                                    )
                                    .map(|sugg| {
                                        (
                                            vec![(ident.span, sugg.to_string())],
                                            String::from(
                                                "there is a crate or module with a similar name",
                                            ),
                                            Applicability::MaybeIncorrect,
                                        )
                                    })
                                },
                            )
                        }
                    } else {
                        let parent = path[i - 1].ident.name;
                        let parent = match parent {
                            // ::foo is mounted at the crate root for 2015, and is the extern
                            // prelude for 2018+
                            kw::PathRoot if self.session.edition() > Edition::Edition2015 => {
                                "the list of imported crates".to_owned()
                            }
                            kw::PathRoot | kw::Crate => "the crate root".to_owned(),
                            _ => {
                                format!("`{}`", parent)
                            }
                        };

                        let mut msg = format!("could not find `{}` in {}", ident, parent);
                        if ns == TypeNS || ns == ValueNS {
                            let ns_to_try = if ns == TypeNS { ValueNS } else { TypeNS };
                            if let FindBindingResult::Binding(Ok(binding)) =
                                find_binding_in_ns(self, ns_to_try)
                            {
                                let mut found = |what| {
                                    msg = format!(
                                        "expected {}, found {} `{}` in {}",
                                        ns.descr(),
                                        what,
                                        ident,
                                        parent
                                    )
                                };
                                if binding.module().is_some() {
                                    found("module")
                                } else {
                                    match binding.res() {
                                        def::Res::<NodeId>::Def(kind, id) => found(kind.descr(id)),
                                        _ => found(ns_to_try.descr()),
                                    }
                                }
                            };
                        }
                        (msg, None)
                    };
                    return PathResult::Failed {
                        span: ident.span,
                        label,
                        suggestion,
                        is_error_from_last_segment: is_last,
                    };
                }
            }
        }

        self.lint_if_path_starts_with_module(crate_lint, path, path_span, second_binding);

        PathResult::Module(match module {
            Some(module) => module,
            None if path.is_empty() => ModuleOrUniformRoot::CurrentScope,
            _ => span_bug!(path_span, "resolve_path: non-empty path `{:?}` has no module", path),
        })
    }

    fn lint_if_path_starts_with_module(
        &mut self,
        crate_lint: CrateLint,
        path: &[Segment],
        path_span: Span,
        second_binding: Option<&NameBinding<'_>>,
    ) {
        let (diag_id, diag_span) = match crate_lint {
            CrateLint::No => return,
            CrateLint::SimplePath(id) => (id, path_span),
            CrateLint::UsePath { root_id, root_span } => (root_id, root_span),
            CrateLint::QPathTrait { qpath_id, qpath_span } => (qpath_id, qpath_span),
        };

        let first_name = match path.get(0) {
            // In the 2018 edition this lint is a hard error, so nothing to do
            Some(seg) if seg.ident.span.rust_2015() && self.session.rust_2015() => seg.ident.name,
            _ => return,
        };

        // We're only interested in `use` paths which should start with
        // `{{root}}` currently.
        if first_name != kw::PathRoot {
            return;
        }

        match path.get(1) {
            // If this import looks like `crate::...` it's already good
            Some(Segment { ident, .. }) if ident.name == kw::Crate => return,
            // Otherwise go below to see if it's an extern crate
            Some(_) => {}
            // If the path has length one (and it's `PathRoot` most likely)
            // then we don't know whether we're gonna be importing a crate or an
            // item in our crate. Defer this lint to elsewhere
            None => return,
        }

        // If the first element of our path was actually resolved to an
        // `ExternCrate` (also used for `crate::...`) then no need to issue a
        // warning, this looks all good!
        if let Some(binding) = second_binding {
            if let NameBindingKind::Import { import, .. } = binding.kind {
                // Careful: we still want to rewrite paths from renamed extern crates.
                if let ImportKind::ExternCrate { source: None, .. } = import.kind {
                    return;
                }
            }
        }

        let diag = BuiltinLintDiagnostics::AbsPathWithModule(diag_span);
        self.lint_buffer.buffer_lint_with_diagnostic(
            lint::builtin::ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,
            diag_id,
            diag_span,
            "absolute paths must start with `self`, `super`, \
             `crate`, or an external crate name in the 2018 edition",
            diag,
        );
    }

    // Validate a local resolution (from ribs).
    fn validate_res_from_ribs(
        &mut self,
        rib_index: usize,
        rib_ident: Ident,
        mut res: Res,
        record_used: bool,
        span: Span,
        original_rib_ident_def: Ident,
        all_ribs: &[Rib<'a>],
    ) -> Res {
        const CG_BUG_STR: &str = "min_const_generics resolve check didn't stop compilation";
        debug!("validate_res_from_ribs({:?})", res);
        let ribs = &all_ribs[rib_index + 1..];

        // An invalid forward use of a generic parameter from a previous default.
        if let ForwardGenericParamBanRibKind = all_ribs[rib_index].kind {
            if record_used {
                let res_error = if rib_ident.name == kw::SelfUpper {
                    ResolutionError::SelfInGenericParamDefault
                } else {
                    ResolutionError::ForwardDeclaredGenericParam
                };
                self.report_error(span, res_error);
            }
            assert_eq!(res, Res::Err);
            return Res::Err;
        }

        match res {
            Res::Local(_) => {
                use ResolutionError::*;
                let mut res_err = None;

                for rib in ribs {
                    match rib.kind {
                        NormalRibKind
                        | ClosureOrAsyncRibKind
                        | ModuleRibKind(..)
                        | MacroDefinition(..)
                        | ForwardGenericParamBanRibKind => {
                            // Nothing to do. Continue.
                        }
                        ItemRibKind(_) | FnItemRibKind | AssocItemRibKind => {
                            // This was an attempt to access an upvar inside a
                            // named function item. This is not allowed, so we
                            // report an error.
                            if record_used {
                                // We don't immediately trigger a resolve error, because
                                // we want certain other resolution errors (namely those
                                // emitted for `ConstantItemRibKind` below) to take
                                // precedence.
                                res_err = Some(CannotCaptureDynamicEnvironmentInFnItem);
                            }
                        }
                        ConstantItemRibKind(_, item) => {
                            // Still doesn't deal with upvars
                            if record_used {
                                let (span, resolution_error) =
                                    if let Some((ident, constant_item_kind)) = item {
                                        let kind_str = match constant_item_kind {
                                            ConstantItemKind::Const => "const",
                                            ConstantItemKind::Static => "static",
                                        };
                                        (
                                            span,
                                            AttemptToUseNonConstantValueInConstant(
                                                ident, "let", kind_str,
                                            ),
                                        )
                                    } else {
                                        (
                                            rib_ident.span,
                                            AttemptToUseNonConstantValueInConstant(
                                                original_rib_ident_def,
                                                "const",
                                                "let",
                                            ),
                                        )
                                    };
                                self.report_error(span, resolution_error);
                            }
                            return Res::Err;
                        }
                        ConstParamTyRibKind => {
                            if record_used {
                                self.report_error(span, ParamInTyOfConstParam(rib_ident.name));
                            }
                            return Res::Err;
                        }
                    }
                }
                if let Some(res_err) = res_err {
                    self.report_error(span, res_err);
                    return Res::Err;
                }
            }
            Res::Def(DefKind::TyParam, _) | Res::SelfTy { .. } => {
                for rib in ribs {
                    let has_generic_params: HasGenericParams = match rib.kind {
                        NormalRibKind
                        | ClosureOrAsyncRibKind
                        | AssocItemRibKind
                        | ModuleRibKind(..)
                        | MacroDefinition(..)
                        | ForwardGenericParamBanRibKind => {
                            // Nothing to do. Continue.
                            continue;
                        }

                        ConstantItemRibKind(trivial, _) => {
                            let features = self.session.features_untracked();
                            // HACK(min_const_generics): We currently only allow `N` or `{ N }`.
                            if !(trivial || features.generic_const_exprs) {
                                // HACK(min_const_generics): If we encounter `Self` in an anonymous constant
                                // we can't easily tell if it's generic at this stage, so we instead remember
                                // this and then enforce the self type to be concrete later on.
                                if let Res::SelfTy { trait_, alias_to: Some((def, _)) } = res {
                                    res = Res::SelfTy { trait_, alias_to: Some((def, true)) }
                                } else {
                                    if record_used {
                                        self.report_error(
                                            span,
                                            ResolutionError::ParamInNonTrivialAnonConst {
                                                name: rib_ident.name,
                                                is_type: true,
                                            },
                                        );
                                    }

                                    self.session.delay_span_bug(span, CG_BUG_STR);
                                    return Res::Err;
                                }
                            }

                            continue;
                        }

                        // This was an attempt to use a type parameter outside its scope.
                        ItemRibKind(has_generic_params) => has_generic_params,
                        FnItemRibKind => HasGenericParams::Yes,
                        ConstParamTyRibKind => {
                            if record_used {
                                self.report_error(
                                    span,
                                    ResolutionError::ParamInTyOfConstParam(rib_ident.name),
                                );
                            }
                            return Res::Err;
                        }
                    };

                    if record_used {
                        self.report_error(
                            span,
                            ResolutionError::GenericParamsFromOuterFunction(
                                res,
                                has_generic_params,
                            ),
                        );
                    }
                    return Res::Err;
                }
            }
            Res::Def(DefKind::ConstParam, _) => {
                let mut ribs = ribs.iter().peekable();
                if let Some(Rib { kind: FnItemRibKind, .. }) = ribs.peek() {
                    // When declaring const parameters inside function signatures, the first rib
                    // is always a `FnItemRibKind`. In this case, we can skip it, to avoid it
                    // (spuriously) conflicting with the const param.
                    ribs.next();
                }

                for rib in ribs {
                    let has_generic_params = match rib.kind {
                        NormalRibKind
                        | ClosureOrAsyncRibKind
                        | AssocItemRibKind
                        | ModuleRibKind(..)
                        | MacroDefinition(..)
                        | ForwardGenericParamBanRibKind => continue,

                        ConstantItemRibKind(trivial, _) => {
                            let features = self.session.features_untracked();
                            // HACK(min_const_generics): We currently only allow `N` or `{ N }`.
                            if !(trivial || features.generic_const_exprs) {
                                if record_used {
                                    self.report_error(
                                        span,
                                        ResolutionError::ParamInNonTrivialAnonConst {
                                            name: rib_ident.name,
                                            is_type: false,
                                        },
                                    );
                                }

                                self.session.delay_span_bug(span, CG_BUG_STR);
                                return Res::Err;
                            }

                            continue;
                        }

                        ItemRibKind(has_generic_params) => has_generic_params,
                        FnItemRibKind => HasGenericParams::Yes,
                        ConstParamTyRibKind => {
                            if record_used {
                                self.report_error(
                                    span,
                                    ResolutionError::ParamInTyOfConstParam(rib_ident.name),
                                );
                            }
                            return Res::Err;
                        }
                    };

                    // This was an attempt to use a const parameter outside its scope.
                    if record_used {
                        self.report_error(
                            span,
                            ResolutionError::GenericParamsFromOuterFunction(
                                res,
                                has_generic_params,
                            ),
                        );
                    }
                    return Res::Err;
                }
            }
            _ => {}
        }
        res
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

    fn report_errors(&mut self, krate: &Crate) {
        self.report_with_use_injections(krate);

        for &(span_use, span_def) in &self.macro_expanded_macro_export_errors {
            let msg = "macro-expanded `macro_export` macros from the current crate \
                       cannot be referred to by absolute paths";
            self.lint_buffer.buffer_lint_with_diagnostic(
                lint::builtin::MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
                CRATE_NODE_ID,
                span_use,
                msg,
                BuiltinLintDiagnostics::MacroExpandedMacroExportsAccessedByAbsolutePaths(span_def),
            );
        }

        for ambiguity_error in &self.ambiguity_errors {
            self.report_ambiguity_error(ambiguity_error);
        }

        let mut reported_spans = FxHashSet::default();
        for error in &self.privacy_errors {
            if reported_spans.insert(error.dedup_span) {
                self.report_privacy_error(error);
            }
        }
    }

    fn report_with_use_injections(&mut self, krate: &Crate) {
        for UseError { mut err, candidates, def_id, instead, suggestion } in
            self.use_injections.drain(..)
        {
            let (span, found_use) = if let Some(def_id) = def_id.as_local() {
                UsePlacementFinder::check(krate, self.def_id_to_node_id[def_id])
            } else {
                (None, false)
            };
            if !candidates.is_empty() {
                diagnostics::show_candidates(
                    &self.definitions,
                    self.session,
                    &mut err,
                    span,
                    &candidates,
                    instead,
                    found_use,
                );
            } else if let Some((span, msg, sugg, appl)) = suggestion {
                err.span_suggestion(span, msg, sugg, appl);
            }
            err.emit();
        }
    }

    fn report_conflict<'b>(
        &mut self,
        parent: Module<'_>,
        ident: Ident,
        ns: Namespace,
        new_binding: &NameBinding<'b>,
        old_binding: &NameBinding<'b>,
    ) {
        // Error on the second of two conflicting names
        if old_binding.span.lo() > new_binding.span.lo() {
            return self.report_conflict(parent, ident, ns, old_binding, new_binding);
        }

        let container = match parent.kind {
            ModuleKind::Def(kind, _, _) => kind.descr(parent.def_id()),
            ModuleKind::Block(..) => "block",
        };

        let old_noun = match old_binding.is_import() {
            true => "import",
            false => "definition",
        };

        let new_participle = match new_binding.is_import() {
            true => "imported",
            false => "defined",
        };

        let (name, span) =
            (ident.name, self.session.source_map().guess_head_span(new_binding.span));

        if let Some(s) = self.name_already_seen.get(&name) {
            if s == &span {
                return;
            }
        }

        let old_kind = match (ns, old_binding.module()) {
            (ValueNS, _) => "value",
            (MacroNS, _) => "macro",
            (TypeNS, _) if old_binding.is_extern_crate() => "extern crate",
            (TypeNS, Some(module)) if module.is_normal() => "module",
            (TypeNS, Some(module)) if module.is_trait() => "trait",
            (TypeNS, _) => "type",
        };

        let msg = format!("the name `{}` is defined multiple times", name);

        let mut err = match (old_binding.is_extern_crate(), new_binding.is_extern_crate()) {
            (true, true) => struct_span_err!(self.session, span, E0259, "{}", msg),
            (true, _) | (_, true) => match new_binding.is_import() && old_binding.is_import() {
                true => struct_span_err!(self.session, span, E0254, "{}", msg),
                false => struct_span_err!(self.session, span, E0260, "{}", msg),
            },
            _ => match (old_binding.is_import(), new_binding.is_import()) {
                (false, false) => struct_span_err!(self.session, span, E0428, "{}", msg),
                (true, true) => struct_span_err!(self.session, span, E0252, "{}", msg),
                _ => struct_span_err!(self.session, span, E0255, "{}", msg),
            },
        };

        err.note(&format!(
            "`{}` must be defined only once in the {} namespace of this {}",
            name,
            ns.descr(),
            container
        ));

        err.span_label(span, format!("`{}` re{} here", name, new_participle));
        err.span_label(
            self.session.source_map().guess_head_span(old_binding.span),
            format!("previous {} of the {} `{}` here", old_noun, old_kind, name),
        );

        // See https://github.com/rust-lang/rust/issues/32354
        use NameBindingKind::Import;
        let import = match (&new_binding.kind, &old_binding.kind) {
            // If there are two imports where one or both have attributes then prefer removing the
            // import without attributes.
            (Import { import: new, .. }, Import { import: old, .. })
                if {
                    !new_binding.span.is_dummy()
                        && !old_binding.span.is_dummy()
                        && (new.has_attributes || old.has_attributes)
                } =>
            {
                if old.has_attributes {
                    Some((new, new_binding.span, true))
                } else {
                    Some((old, old_binding.span, true))
                }
            }
            // Otherwise prioritize the new binding.
            (Import { import, .. }, other) if !new_binding.span.is_dummy() => {
                Some((import, new_binding.span, other.is_import()))
            }
            (other, Import { import, .. }) if !old_binding.span.is_dummy() => {
                Some((import, old_binding.span, other.is_import()))
            }
            _ => None,
        };

        // Check if the target of the use for both bindings is the same.
        let duplicate = new_binding.res().opt_def_id() == old_binding.res().opt_def_id();
        let has_dummy_span = new_binding.span.is_dummy() || old_binding.span.is_dummy();
        let from_item =
            self.extern_prelude.get(&ident).map_or(true, |entry| entry.introduced_by_item);
        // Only suggest removing an import if both bindings are to the same def, if both spans
        // aren't dummy spans. Further, if both bindings are imports, then the ident must have
        // been introduced by an item.
        let should_remove_import = duplicate
            && !has_dummy_span
            && ((new_binding.is_extern_crate() || old_binding.is_extern_crate()) || from_item);

        match import {
            Some((import, span, true)) if should_remove_import && import.is_nested() => {
                self.add_suggestion_for_duplicate_nested_use(&mut err, import, span)
            }
            Some((import, _, true)) if should_remove_import && !import.is_glob() => {
                // Simple case - remove the entire import. Due to the above match arm, this can
                // only be a single use so just remove it entirely.
                err.tool_only_span_suggestion(
                    import.use_span_with_attributes,
                    "remove unnecessary import",
                    String::new(),
                    Applicability::MaybeIncorrect,
                );
            }
            Some((import, span, _)) => {
                self.add_suggestion_for_rename_of_use(&mut err, name, import, span)
            }
            _ => {}
        }

        err.emit();
        self.name_already_seen.insert(name, span);
    }

    /// This function adds a suggestion to change the binding name of a new import that conflicts
    /// with an existing import.
    ///
    /// ```text,ignore (diagnostic)
    /// help: you can use `as` to change the binding name of the import
    ///    |
    /// LL | use foo::bar as other_bar;
    ///    |     ^^^^^^^^^^^^^^^^^^^^^
    /// ```
    fn add_suggestion_for_rename_of_use(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        name: Symbol,
        import: &Import<'_>,
        binding_span: Span,
    ) {
        let suggested_name = if name.as_str().chars().next().unwrap().is_uppercase() {
            format!("Other{}", name)
        } else {
            format!("other_{}", name)
        };

        let mut suggestion = None;
        match import.kind {
            ImportKind::Single { type_ns_only: true, .. } => {
                suggestion = Some(format!("self as {}", suggested_name))
            }
            ImportKind::Single { source, .. } => {
                if let Some(pos) =
                    source.span.hi().0.checked_sub(binding_span.lo().0).map(|pos| pos as usize)
                {
                    if let Ok(snippet) = self.session.source_map().span_to_snippet(binding_span) {
                        if pos <= snippet.len() {
                            suggestion = Some(format!(
                                "{} as {}{}",
                                &snippet[..pos],
                                suggested_name,
                                if snippet.ends_with(';') { ";" } else { "" }
                            ))
                        }
                    }
                }
            }
            ImportKind::ExternCrate { source, target, .. } => {
                suggestion = Some(format!(
                    "extern crate {} as {};",
                    source.unwrap_or(target.name),
                    suggested_name,
                ))
            }
            _ => unreachable!(),
        }

        let rename_msg = "you can use `as` to change the binding name of the import";
        if let Some(suggestion) = suggestion {
            err.span_suggestion(
                binding_span,
                rename_msg,
                suggestion,
                Applicability::MaybeIncorrect,
            );
        } else {
            err.span_label(binding_span, rename_msg);
        }
    }

    /// This function adds a suggestion to remove an unnecessary binding from an import that is
    /// nested. In the following example, this function will be invoked to remove the `a` binding
    /// in the second use statement:
    ///
    /// ```ignore (diagnostic)
    /// use issue_52891::a;
    /// use issue_52891::{d, a, e};
    /// ```
    ///
    /// The following suggestion will be added:
    ///
    /// ```ignore (diagnostic)
    /// use issue_52891::{d, a, e};
    ///                      ^-- help: remove unnecessary import
    /// ```
    ///
    /// If the nested use contains only one import then the suggestion will remove the entire
    /// line.
    ///
    /// It is expected that the provided import is nested - this isn't checked by the
    /// function. If this invariant is not upheld, this function's behaviour will be unexpected
    /// as characters expected by span manipulations won't be present.
    fn add_suggestion_for_duplicate_nested_use(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        import: &Import<'_>,
        binding_span: Span,
    ) {
        assert!(import.is_nested());
        let message = "remove unnecessary import";

        // Two examples will be used to illustrate the span manipulations we're doing:
        //
        // - Given `use issue_52891::{d, a, e};` where `a` is a duplicate then `binding_span` is
        //   `a` and `import.use_span` is `issue_52891::{d, a, e};`.
        // - Given `use issue_52891::{d, e, a};` where `a` is a duplicate then `binding_span` is
        //   `a` and `import.use_span` is `issue_52891::{d, e, a};`.

        let (found_closing_brace, span) =
            find_span_of_binding_until_next_binding(self.session, binding_span, import.use_span);

        // If there was a closing brace then identify the span to remove any trailing commas from
        // previous imports.
        if found_closing_brace {
            if let Some(span) = extend_span_to_previous_binding(self.session, span) {
                err.tool_only_span_suggestion(
                    span,
                    message,
                    String::new(),
                    Applicability::MaybeIncorrect,
                );
            } else {
                // Remove the entire line if we cannot extend the span back, this indicates an
                // `issue_52891::{self}` case.
                err.span_suggestion(
                    import.use_span_with_attributes,
                    message,
                    String::new(),
                    Applicability::MaybeIncorrect,
                );
            }

            return;
        }

        err.span_suggestion(span, message, String::new(), Applicability::MachineApplicable);
    }

    fn extern_prelude_get(
        &mut self,
        ident: Ident,
        speculative: bool,
    ) -> Option<&'a NameBinding<'a>> {
        if ident.is_path_segment_keyword() {
            // Make sure `self`, `super` etc produce an error when passed to here.
            return None;
        }
        self.extern_prelude.get(&ident.normalize_to_macros_2_0()).cloned().and_then(|entry| {
            if let Some(binding) = entry.extern_crate_item {
                if !speculative && entry.introduced_by_item {
                    self.record_use(ident, binding, false);
                }
                Some(binding)
            } else {
                let crate_id = if !speculative {
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

    /// Rustdoc uses this to resolve things in a recoverable way. `ResolutionError<'a>`
    /// isn't something that can be returned because it can't be made to live that long,
    /// and also it's a private type. Fortunately rustdoc doesn't need to know the error,
    /// just that an error occurred.
    // FIXME(Manishearth): intra-doc links won't get warned of epoch changes.
    pub fn resolve_str_path_error(
        &mut self,
        span: Span,
        path_str: &str,
        ns: Namespace,
        module_id: DefId,
    ) -> Result<(ast::Path, Res), ()> {
        let path = if path_str.starts_with("::") {
            ast::Path {
                span,
                segments: iter::once(Ident::with_dummy_span(kw::PathRoot))
                    .chain(path_str.split("::").skip(1).map(Ident::from_str))
                    .map(|i| self.new_ast_path_segment(i))
                    .collect(),
                tokens: None,
            }
        } else {
            ast::Path {
                span,
                segments: path_str
                    .split("::")
                    .map(Ident::from_str)
                    .map(|i| self.new_ast_path_segment(i))
                    .collect(),
                tokens: None,
            }
        };
        let module = self.expect_module(module_id);
        let parent_scope = &ParentScope::module(module, self);
        let res = self.resolve_ast_path(&path, ns, parent_scope).map_err(|_| ())?;
        Ok((path, res))
    }

    // Resolve a path passed from rustdoc or HIR lowering.
    fn resolve_ast_path(
        &mut self,
        path: &ast::Path,
        ns: Namespace,
        parent_scope: &ParentScope<'a>,
    ) -> Result<Res, (Span, ResolutionError<'a>)> {
        match self.resolve_path(
            &Segment::from_path(path),
            Some(ns),
            parent_scope,
            false,
            path.span,
            CrateLint::No,
        ) {
            PathResult::Module(ModuleOrUniformRoot::Module(module)) => Ok(module.res().unwrap()),
            PathResult::NonModule(path_res) if path_res.unresolved_segments() == 0 => {
                Ok(path_res.base_res())
            }
            PathResult::NonModule(..) => Err((
                path.span,
                ResolutionError::FailedToResolve {
                    label: String::from("type-relative paths are not supported in this context"),
                    suggestion: None,
                },
            )),
            PathResult::Module(..) | PathResult::Indeterminate => unreachable!(),
            PathResult::Failed { span, label, suggestion, .. } => {
                Err((span, ResolutionError::FailedToResolve { label, suggestion }))
            }
        }
    }

    fn new_ast_path_segment(&mut self, ident: Ident) -> ast::PathSegment {
        let mut seg = ast::PathSegment::from_ident(ident);
        seg.id = self.next_node_id();
        seg
    }

    // For rustdoc.
    pub fn graph_root(&self) -> Module<'a> {
        self.graph_root
    }

    // For rustdoc.
    pub fn all_macros(&self) -> &FxHashMap<Symbol, Res> {
        &self.all_macros
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

    /// Retrieves the span of the given `DefId` if `DefId` is in the local crate.
    #[inline]
    pub fn opt_span(&self, def_id: DefId) -> Option<Span> {
        def_id.as_local().map(|def_id| self.definitions.def_span(def_id))
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

        let name_binding = match self.resolve_ident_in_module(
            ModuleOrUniformRoot::Module(module),
            ident,
            ValueNS,
            parent_scope,
            false,
            DUMMY_SP,
        ) {
            Ok(name_binding) => name_binding,
            _ => return,
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
enum CrateLint {
    /// Do not issue the lint.
    No,

    /// This lint applies to some arbitrary path; e.g., `impl ::foo::Bar`.
    /// In this case, we can take the span of that path.
    SimplePath(NodeId),

    /// This lint comes from a `use` statement. In this case, what we
    /// care about really is the *root* `use` statement; e.g., if we
    /// have nested things like `use a::{b, c}`, we care about the
    /// `use a` part.
    UsePath { root_id: NodeId, root_span: Span },

    /// This is the "trait item" from a fully qualified path. For example,
    /// we might be resolving  `X::Y::Z` from a path like `<T as X::Y>::Z`.
    /// The `path_span` is the span of the to the trait itself (`X::Y`).
    QPathTrait { qpath_id: NodeId, qpath_span: Span },
}

impl CrateLint {
    fn node_id(&self) -> Option<NodeId> {
        match *self {
            CrateLint::No => None,
            CrateLint::SimplePath(id)
            | CrateLint::UsePath { root_id: id, .. }
            | CrateLint::QPathTrait { qpath_id: id, .. } => Some(id),
        }
    }
}

pub fn provide(providers: &mut Providers) {
    late::lifetimes::provide(providers);
}
