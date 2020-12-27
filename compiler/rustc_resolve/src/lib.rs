// ignore-tidy-filelength

//! This crate is responsible for the part of name resolution that doesn't require type checker.
//!
//! Module structure of the crate is built here.
//! Paths in macros, imports, expressions, types, patterns are resolved here.
//! Label and lifetime names are resolved here as well.
//!
//! Type-relative name resolution (methods, fields, associated items) happens in `librustc_typeck`.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(bool_to_option)]
#![feature(crate_visibility_modifier)]
#![feature(format_args_capture)]
#![feature(nll)]
#![feature(or_patterns)]
#![recursion_limit = "256"]

pub use rustc_hir::def::{Namespace, PerNS};

use Determinacy::*;

use rustc_arena::{DroplessArena, TypedArena};
use rustc_ast::node_id::NodeMap;
use rustc_ast::unwrap_or;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, FloatTy, IntTy, NodeId, UintTy};
use rustc_ast::{Crate, CRATE_NODE_ID};
use rustc_ast::{ItemKind, Path};
use rustc_ast_lowering::ResolverAstLowering;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::ptr_key::PtrKey;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_expand::base::SyntaxExtension;
use rustc_hir::def::Namespace::*;
use rustc_hir::def::{self, CtorOf, DefKind, NonMacroAttrKind, PartialRes};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::definitions::{DefKey, DefPathData, Definitions};
use rustc_hir::PrimTy::{self, Bool, Char, Float, Int, Str, Uint};
use rustc_hir::TraitCandidate;
use rustc_index::vec::IndexVec;
use rustc_metadata::creader::{CStore, CrateLoader};
use rustc_middle::hir::exports::ExportMap;
use rustc_middle::middle::cstore::{CrateStore, MetadataLoaderDyn};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, DefIdTree, ResolverOutputs};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_session::lint::{BuiltinLintDiagnostics, LintBuffer};
use rustc_session::Session;
use rustc_span::hygiene::{ExpnId, ExpnKind, MacroKind, SyntaxContext, Transparency};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

use smallvec::{smallvec, SmallVec};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::{cmp, fmt, iter, ptr};
use tracing::debug;

use diagnostics::{extend_span_to_previous_binding, find_span_of_binding_until_next_binding};
use diagnostics::{ImportSuggestion, LabelSuggestion, Suggestion};
use imports::{Import, ImportKind, ImportResolver, NameResolution};
use late::{HasGenericParams, PathSource, Rib, RibKind::*};
use macros::{MacroRulesBinding, MacroRulesScope, MacroRulesScopeRef};

type Res = def::Res<NodeId>;

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
    DeriveHelpers(ExpnId),
    DeriveHelpersCompat,
    MacroRules(MacroRulesScopeRef<'a>),
    CrateRoot,
    Module(Module<'a>),
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
enum ScopeSet {
    /// All scopes with the given namespace.
    All(Namespace, /*is_import*/ bool),
    /// Crate root, then extern prelude (used for mixed 2015-2018 mode in macros).
    AbsolutePath(Namespace),
    /// All scopes with macro namespace and the given macro kind restriction.
    Macro(MacroKind),
}

/// Everything you need to know about a name's location to resolve it.
/// Serves as a starting point for the scope visitor.
/// This struct is currently used only for early resolution (imports and macros),
/// but not for late resolution yet.
#[derive(Clone, Copy, Debug)]
pub struct ParentScope<'a> {
    module: Module<'a>,
    expansion: ExpnId,
    macro_rules: MacroRulesScopeRef<'a>,
    derives: &'a [ast::Path],
}

impl<'a> ParentScope<'a> {
    /// Creates a parent scope with the passed argument used as the module scope component,
    /// and other scope components set to default empty values.
    pub fn module(module: Module<'a>, resolver: &Resolver<'a>) -> ParentScope<'a> {
        ParentScope {
            module,
            expansion: ExpnId::root(),
            macro_rules: resolver.arenas.alloc_macro_rules_scope(MacroRulesScope::Empty),
            derives: &[],
        }
    }
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
    MethodNotMemberOfTrait(Symbol, &'a str),
    /// Error E0437: type is not a member of trait.
    TypeNotMemberOfTrait(Symbol, &'a str),
    /// Error E0438: const is not a member of trait.
    ConstNotMemberOfTrait(Symbol, &'a str),
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
    AttemptToUseNonConstantValueInConstant,
    /// Error E0530: `X` bindings cannot shadow `Y`s.
    BindingShadowsSomethingUnacceptable(&'static str, Symbol, &'a NameBinding<'a>),
    /// Error E0128: type parameters with a default cannot use forward-declared identifiers.
    ForwardDeclaredTyParam, // FIXME(const_generics:defaults)
    /// ERROR E0770: the type of const parameters must not depend on other generic parameters.
    ParamInTyOfConstParam(Symbol),
    /// constant values inside of type parameter defaults must not depend on generic parameters.
    ParamInAnonConstInTyDefault(Symbol),
    /// generic parameters must not be used inside const evaluations.
    ///
    /// This error is only emitted when using `min_const_generics`.
    ParamInNonTrivialAnonConst { name: Symbol, is_type: bool },
    /// Error E0735: type parameters with a default cannot use `Self`
    SelfInTyParamDefault,
    /// Error E0767: use of unreachable label
    UnreachableLabel { name: Symbol, definition_span: Span, suggestion: Option<LabelSuggestion> },
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
        visit::walk_crate(&mut finder, krate);
        (finder.span, finder.found_use)
    }
}

impl<'tcx> Visitor<'tcx> for UsePlacementFinder {
    fn visit_mod(
        &mut self,
        module: &'tcx ast::Mod,
        _: Span,
        _: &[ast::Attribute],
        node_id: NodeId,
    ) {
        if self.span.is_some() {
            return;
        }
        if node_id != self.target_module {
            visit::walk_mod(self, module);
            return;
        }
        // find a use statement
        for item in &module.items {
            match item.kind {
                ItemKind::Use(..) => {
                    // don't suggest placing a use before the prelude
                    // import or other generated ones
                    if !item.span.from_expansion() {
                        self.span = Some(item.span.shrink_to_lo());
                        self.found_use = true;
                        return;
                    }
                }
                // don't place use before extern crate
                ItemKind::ExternCrate(_) => {}
                // but place them before the first other item
                _ => {
                    if self.span.map_or(true, |span| item.span < span)
                        && !item.span.from_expansion()
                    {
                        // don't insert between attributes and an item
                        if item.attrs.is_empty() {
                            self.span = Some(item.span.shrink_to_lo());
                        } else {
                            // find the first attribute on the item
                            for attr in &item.attrs {
                                if self.span.map_or(true, |span| attr.span < span) {
                                    self.span = Some(attr.span.shrink_to_lo());
                                }
                            }
                        }
                    }
                }
            }
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
                lhs.def_id() == rhs.def_id()
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
    /// * A normal module ‒ either `mod from_file;` or `mod from_block { }`.
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
pub struct ModuleData<'a> {
    parent: Option<Module<'a>>,
    kind: ModuleKind,

    // The def id of the closest normal module (`mod`) ancestor (including this module).
    normal_ancestor_id: DefId,

    // Mapping between names and their (possibly in-progress) resolutions in this module.
    // Resolutions in modules from other crates are not populated until accessed.
    lazy_resolutions: Resolutions<'a>,
    // True if this is a module from other crate that needs to be populated on access.
    populate_on_access: Cell<bool>,

    // Macro invocations that can expand into items in this module.
    unexpanded_invocations: RefCell<FxHashSet<ExpnId>>,

    no_implicit_prelude: bool,

    glob_importers: RefCell<Vec<&'a Import<'a>>>,
    globs: RefCell<Vec<&'a Import<'a>>>,

    // Used to memoize the traits in this module for faster searches through all traits in scope.
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
        normal_ancestor_id: DefId,
        expansion: ExpnId,
        span: Span,
    ) -> Self {
        ModuleData {
            parent,
            kind,
            normal_ancestor_id,
            lazy_resolutions: Default::default(),
            populate_on_access: Cell::new(!normal_ancestor_id.is_local()),
            unexpanded_invocations: Default::default(),
            no_implicit_prelude: false,
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

    fn def_id(&self) -> Option<DefId> {
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
    expansion: ExpnId,
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
    /// Is this a name binding of a import?
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
            AmbiguityKind::Import => "name vs any other name during import resolution",
            AmbiguityKind::BuiltinAttr => "built-in attribute vs any other name",
            AmbiguityKind::DeriveHelper => "derive helper attribute vs any other name",
            AmbiguityKind::MacroRulesVsModularized => {
                "`macro_rules` vs non-`macro_rules` from other module"
            }
            AmbiguityKind::GlobVsOuter => {
                "glob import vs any other name from outer scope during import/macro resolution"
            }
            AmbiguityKind::GlobVsGlob => "glob import vs glob import in the same module",
            AmbiguityKind::GlobVsExpanded => {
                "glob import vs macro-expanded name in the same \
                 module during import/macro resolution"
            }
            AmbiguityKind::MoreExpandedVsOuter => {
                "macro-expanded name vs less macro-expanded name \
                 from outer scope during import/macro resolution"
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
            _ => self.is_variant(),
        }
    }

    // We sometimes need to treat variants as `pub` for backwards compatibility.
    fn pseudo_vis(&self) -> ty::Visibility {
        if self.is_variant() && self.res().def_id().is_local() {
            ty::Visibility::Public
        } else {
            self.vis
        }
    }

    fn is_variant(&self) -> bool {
        matches!(self.kind, NameBindingKind::Res(
                Res::Def(DefKind::Variant | DefKind::Ctor(CtorOf::Variant, ..), _),
                _,
            ))
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
    fn may_appear_after(&self, invoc_parent_expansion: ExpnId, binding: &NameBinding<'_>) -> bool {
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

/// Interns the names of the primitive types.
///
/// All other types are defined somewhere and possibly imported, but the primitive ones need
/// special handling, since they have no place of origin.
struct PrimitiveTypeTable {
    primitive_types: FxHashMap<Symbol, PrimTy>,
}

impl PrimitiveTypeTable {
    fn new() -> PrimitiveTypeTable {
        let mut table = FxHashMap::default();

        table.insert(sym::bool, Bool);
        table.insert(sym::char, Char);
        table.insert(sym::f32, Float(FloatTy::F32));
        table.insert(sym::f64, Float(FloatTy::F64));
        table.insert(sym::isize, Int(IntTy::Isize));
        table.insert(sym::i8, Int(IntTy::I8));
        table.insert(sym::i16, Int(IntTy::I16));
        table.insert(sym::i32, Int(IntTy::I32));
        table.insert(sym::i64, Int(IntTy::I64));
        table.insert(sym::i128, Int(IntTy::I128));
        table.insert(sym::str, Str);
        table.insert(sym::usize, Uint(UintTy::Usize));
        table.insert(sym::u8, Uint(UintTy::U8));
        table.insert(sym::u16, Uint(UintTy::U16));
        table.insert(sym::u32, Uint(UintTy::U32));
        table.insert(sym::u64, Uint(UintTy::U64));
        table.insert(sym::u128, Uint(UintTy::U128));
        Self { primitive_types: table }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExternPreludeEntry<'a> {
    extern_crate_item: Option<&'a NameBinding<'a>>,
    pub introduced_by_item: bool,
}

/// Used for better errors for E0773
enum BuiltinMacroState {
    NotYetSeen(SyntaxExtension),
    AlreadySeen(Span),
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

    /// The idents for the primitive types.
    primitive_type_table: PrimitiveTypeTable,

    /// Resolutions for nodes that have a single resolution.
    partial_res_map: NodeMap<PartialRes>,
    /// Resolutions for import nodes, which have multiple resolutions in different namespaces.
    import_res_map: NodeMap<PerNS<Option<Res>>>,
    /// Resolutions for labels (node IDs of their corresponding blocks or loops).
    label_res_map: NodeMap<NodeId>,

    /// `CrateNum` resolutions of `extern crate` items.
    extern_crate_map: FxHashMap<LocalDefId, CrateNum>,
    export_map: ExportMap<LocalDefId>,
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
    module_map: FxHashMap<LocalDefId, Module<'a>>,
    extern_module_map: FxHashMap<DefId, Module<'a>>,
    binding_parent_modules: FxHashMap<PtrKey<'a, NameBinding<'a>>, Module<'a>>,
    underscore_disambiguator: u32,

    /// Maps glob imports to the names of items actually imported.
    glob_map: FxHashMap<LocalDefId, FxHashSet<Symbol>>,
    /// Visibilities in "lowered" form, for all entities that have them.
    visibilities: FxHashMap<LocalDefId, ty::Visibility>,
    used_imports: FxHashSet<(NodeId, Namespace)>,
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
    registered_tools: FxHashSet<Ident>,
    macro_use_prelude: FxHashMap<Symbol, &'a NameBinding<'a>>,
    all_macros: FxHashMap<Symbol, Res>,
    macro_map: FxHashMap<DefId, Lrc<SyntaxExtension>>,
    dummy_ext_bang: Lrc<SyntaxExtension>,
    dummy_ext_derive: Lrc<SyntaxExtension>,
    non_macro_attrs: [Lrc<SyntaxExtension>; 2],
    local_macro_def_scopes: FxHashMap<LocalDefId, Module<'a>>,
    ast_transform_scopes: FxHashMap<ExpnId, Module<'a>>,
    unused_macros: FxHashMap<LocalDefId, (NodeId, Span)>,
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
    containers_deriving_copy: FxHashSet<ExpnId>,
    /// Parent scopes in which the macros were invoked.
    /// FIXME: `derives` are missing in these parent scopes and need to be taken from elsewhere.
    invocation_parent_scopes: FxHashMap<ExpnId, ParentScope<'a>>,
    /// `macro_rules` scopes *produced* by expanding the macro invocations,
    /// include all the `macro_rules` items and other invocations generated by them.
    output_macro_rules_scopes: FxHashMap<ExpnId, MacroRulesScopeRef<'a>>,
    /// Helper attributes that are in scope for the given expansion.
    helper_attrs: FxHashMap<ExpnId, Vec<Ident>>,

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

    def_id_to_span: IndexVec<LocalDefId, Span>,

    node_id_to_def_id: FxHashMap<ast::NodeId, LocalDefId>,
    def_id_to_node_id: IndexVec<LocalDefId, ast::NodeId>,

    /// Indices of unnamed struct or variant fields with unresolved attributes.
    placeholder_field_indices: FxHashMap<NodeId, usize>,
    /// When collecting definitions from an AST fragment produced by a macro invocation `ExpnId`
    /// we know what parent node that fragment should be attached to thanks to this table.
    invocation_parents: FxHashMap<ExpnId, LocalDefId>,

    next_disambiguator: FxHashMap<(LocalDefId, DefPathData), u32>,
    /// Some way to know that we are in a *trait* impl in `visit_assoc_item`.
    /// FIXME: Replace with a more general AST map (together with some other fields).
    trait_impl_items: FxHashSet<LocalDefId>,
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
    fn alloc_module(&'a self, module: ModuleData<'a>) -> Module<'a> {
        let module = self.modules.alloc(module);
        if module.def_id().map(|def_id| def_id.is_local()).unwrap_or(true) {
            self.local_modules.borrow_mut().push(module);
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
        PtrKey(self.dropless.alloc(Cell::new(scope)))
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

    fn item_generics_num_lifetimes(&self, def_id: DefId, sess: &Session) -> usize {
        self.cstore().item_generics_num_lifetimes(def_id, sess)
    }

    fn get_partial_res(&mut self, id: NodeId) -> Option<PartialRes> {
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

    fn lint_buffer(&mut self) -> &mut LintBuffer {
        &mut self.lint_buffer
    }

    fn next_node_id(&mut self) -> NodeId {
        self.next_node_id()
    }

    fn trait_map(&self) -> &NodeMap<Vec<TraitCandidate>> {
        &self.trait_map
    }

    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.node_id_to_def_id.get(&node).copied()
    }

    fn local_def_id(&self, node: NodeId) -> LocalDefId {
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

        // Find the next free disambiguator for this key.
        let next_disambiguator = &mut self.next_disambiguator;
        let next_disambiguator = |parent, data| {
            let next_disamb = next_disambiguator.entry((parent, data)).or_insert(0);
            let disambiguator = *next_disamb;
            *next_disamb = next_disamb.checked_add(1).expect("disambiguator overflow");
            disambiguator
        };

        let def_id = self.definitions.create_def(parent, data, expn_id, next_disambiguator);

        assert_eq!(self.def_id_to_span.push(span), def_id);

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
        metadata_loader: &'a MetadataLoaderDyn,
        arenas: &'a ResolverArenas<'a>,
    ) -> Resolver<'a> {
        let root_local_def_id = LocalDefId { local_def_index: CRATE_DEF_INDEX };
        let root_def_id = root_local_def_id.to_def_id();
        let root_module_kind = ModuleKind::Def(DefKind::Mod, root_def_id, kw::Invalid);
        let graph_root = arenas.alloc_module(ModuleData {
            no_implicit_prelude: session.contains_name(&krate.attrs, sym::no_implicit_prelude),
            ..ModuleData::new(None, root_module_kind, root_def_id, ExpnId::root(), krate.span)
        });
        let empty_module_kind = ModuleKind::Def(DefKind::Mod, root_def_id, kw::Invalid);
        let empty_module = arenas.alloc_module(ModuleData {
            no_implicit_prelude: true,
            ..ModuleData::new(
                Some(graph_root),
                empty_module_kind,
                root_def_id,
                ExpnId::root(),
                DUMMY_SP,
            )
        });
        let mut module_map = FxHashMap::default();
        module_map.insert(root_local_def_id, graph_root);

        let definitions = Definitions::new(crate_name, session.local_crate_disambiguator());
        let root = definitions.get_root_def();

        let mut visibilities = FxHashMap::default();
        visibilities.insert(root_local_def_id, ty::Visibility::Public);

        let mut def_id_to_span = IndexVec::default();
        assert_eq!(def_id_to_span.push(rustc_span::DUMMY_SP), root);
        let mut def_id_to_node_id = IndexVec::default();
        assert_eq!(def_id_to_node_id.push(CRATE_NODE_ID), root);
        let mut node_id_to_def_id = FxHashMap::default();
        node_id_to_def_id.insert(CRATE_NODE_ID, root);

        let mut invocation_parents = FxHashMap::default();
        invocation_parents.insert(ExpnId::root(), root);

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
        let non_macro_attr =
            |mark_used| Lrc::new(SyntaxExtension::non_macro_attr(mark_used, session.edition()));

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

            primitive_type_table: PrimitiveTypeTable::new(),

            partial_res_map: Default::default(),
            import_res_map: Default::default(),
            label_res_map: Default::default(),
            extern_crate_map: Default::default(),
            export_map: FxHashMap::default(),
            trait_map: Default::default(),
            underscore_disambiguator: 0,
            empty_module,
            module_map,
            block_map: Default::default(),
            extern_module_map: FxHashMap::default(),
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
                expansion: ExpnId::root(),
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
            non_macro_attrs: [non_macro_attr(false), non_macro_attr(true)],
            invocation_parent_scopes: Default::default(),
            output_macro_rules_scopes: Default::default(),
            helper_attrs: Default::default(),
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
            next_node_id: NodeId::from_u32(1),
            def_id_to_span,
            node_id_to_def_id,
            def_id_to_node_id,
            placeholder_field_indices: Default::default(),
            invocation_parents,
            next_disambiguator: Default::default(),
            trait_impl_items: Default::default(),
        };

        let root_parent_scope = ParentScope::module(graph_root, &resolver);
        resolver.invocation_parent_scopes.insert(ExpnId::root(), root_parent_scope);

        resolver
    }

    pub fn next_node_id(&mut self) -> NodeId {
        let next = self
            .next_node_id
            .as_usize()
            .checked_add(1)
            .expect("input too large; ran out of NodeIds");
        self.next_node_id = ast::NodeId::from_usize(next);
        self.next_node_id
    }

    pub fn lint_buffer(&mut self) -> &mut LintBuffer {
        &mut self.lint_buffer
    }

    pub fn arenas() -> ResolverArenas<'a> {
        Default::default()
    }

    pub fn into_outputs(self) -> ResolverOutputs {
        let definitions = self.definitions;
        let visibilities = self.visibilities;
        let extern_crate_map = self.extern_crate_map;
        let export_map = self.export_map;
        let maybe_unused_trait_imports = self.maybe_unused_trait_imports;
        let maybe_unused_extern_crates = self.maybe_unused_extern_crates;
        let glob_map = self.glob_map;
        ResolverOutputs {
            definitions,
            cstore: Box::new(self.crate_loader.into_cstore()),
            visibilities,
            extern_crate_map,
            export_map,
            glob_map,
            maybe_unused_trait_imports,
            maybe_unused_extern_crates,
            extern_prelude: self
                .extern_prelude
                .iter()
                .map(|(ident, entry)| (ident.name, entry.introduced_by_item))
                .collect(),
        }
    }

    pub fn clone_outputs(&self) -> ResolverOutputs {
        ResolverOutputs {
            definitions: self.definitions.clone(),
            cstore: Box::new(self.cstore().clone()),
            visibilities: self.visibilities.clone(),
            extern_crate_map: self.extern_crate_map.clone(),
            export_map: self.export_map.clone(),
            glob_map: self.glob_map.clone(),
            maybe_unused_trait_imports: self.maybe_unused_trait_imports.clone(),
            maybe_unused_extern_crates: self.maybe_unused_extern_crates.clone(),
            extern_prelude: self
                .extern_prelude
                .iter()
                .map(|(ident, entry)| (ident.name, entry.introduced_by_item))
                .collect(),
        }
    }

    pub fn cstore(&self) -> &CStore {
        self.crate_loader.cstore()
    }

    fn non_macro_attr(&self, mark_used: bool) -> Lrc<SyntaxExtension> {
        self.non_macro_attrs[mark_used as usize].clone()
    }

    fn dummy_ext(&self, macro_kind: MacroKind) -> Lrc<SyntaxExtension> {
        match macro_kind {
            MacroKind::Bang => self.dummy_ext_bang.clone(),
            MacroKind::Derive => self.dummy_ext_derive.clone(),
            MacroKind::Attr => self.non_macro_attr(true),
        }
    }

    /// Runs the function on each namespace.
    fn per_ns<F: FnMut(&mut Self, Namespace)>(&mut self, mut f: F) {
        f(self, TypeNS);
        f(self, ValueNS);
        f(self, MacroNS);
    }

    fn is_builtin_macro(&mut self, res: Res) -> bool {
        self.get_macro(res).map_or(false, |ext| ext.is_builtin)
    }

    fn macro_def(&self, mut ctxt: SyntaxContext) -> DefId {
        loop {
            match ctxt.outer_expn().expn_data().macro_def_id {
                Some(def_id) => return def_id,
                None => ctxt.remove_mark(),
            };
        }
    }

    /// Entry point to crate resolution.
    pub fn resolve_crate(&mut self, krate: &Crate) {
        let _prof_timer = self.session.prof.generic_activity("resolve_crate");

        ImportResolver { r: self }.finalize_imports();
        self.finalize_macro_resolutions();

        self.late_resolve_crate(krate);

        self.check_unused(krate);
        self.report_errors(krate);
        self.crate_loader.postprocess(krate);
    }

    fn get_traits_in_module_containing_item(
        &mut self,
        ident: Ident,
        ns: Namespace,
        module: Module<'a>,
        found_traits: &mut Vec<TraitCandidate>,
        parent_scope: &ParentScope<'a>,
    ) {
        assert!(ns == TypeNS || ns == ValueNS);
        module.ensure_traits(self);
        let traits = module.traits.borrow();

        for &(trait_name, binding) in traits.as_ref().unwrap().iter() {
            // Traits have pseudo-modules that can be used to search for the given ident.
            if let Some(module) = binding.module() {
                let mut ident = ident;
                if ident.span.glob_adjust(module.expansion, binding.span).is_none() {
                    continue;
                }
                if self
                    .resolve_ident_in_module_unadjusted(
                        ModuleOrUniformRoot::Module(module),
                        ident,
                        ns,
                        parent_scope,
                        false,
                        module.span,
                    )
                    .is_ok()
                {
                    let import_ids = self.find_transitive_imports(&binding.kind, trait_name);
                    let trait_def_id = module.def_id().unwrap();
                    found_traits.push(TraitCandidate { def_id: trait_def_id, import_ids });
                }
            } else if let Res::Def(DefKind::TraitAlias, _) = binding.res() {
                // For now, just treat all trait aliases as possible candidates, since we don't
                // know if the ident is somewhere in the transitive bounds.
                let import_ids = self.find_transitive_imports(&binding.kind, trait_name);
                let trait_def_id = binding.res().def_id();
                found_traits.push(TraitCandidate { def_id: trait_def_id, import_ids });
            } else {
                bug!("candidate is not trait or trait alias?")
            }
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

    fn new_module(
        &self,
        parent: Module<'a>,
        kind: ModuleKind,
        normal_ancestor_id: DefId,
        expn_id: ExpnId,
        span: Span,
    ) -> Module<'a> {
        let module = ModuleData::new(Some(parent), kind, normal_ancestor_id, expn_id, span);
        self.arenas.alloc_module(module)
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
        ns: Namespace,
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
            self.used_imports.insert((import.id, ns));
            self.add_to_glob_map(&import, ident);
            self.record_use(ident, ns, binding, false);
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
        scope_set: ScopeSet,
        parent_scope: &ParentScope<'a>,
        ident: Ident,
        mut visitor: impl FnMut(&mut Self, Scope<'a>, /*use_prelude*/ bool, Ident) -> Option<T>,
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

        let rust_2015 = ident.span.rust_2015();
        let (ns, macro_kind, is_absolute_path) = match scope_set {
            ScopeSet::All(ns, _) => (ns, None, false),
            ScopeSet::AbsolutePath(ns) => (ns, None, true),
            ScopeSet::Macro(macro_kind) => (MacroNS, Some(macro_kind), false),
        };
        // Jump out of trait or enum modules, they do not act as scopes.
        let module = parent_scope.module.nearest_item_scope();
        let mut scope = match ns {
            _ if is_absolute_path => Scope::CrateRoot,
            TypeNS | ValueNS => Scope::Module(module),
            MacroNS => Scope::DeriveHelpers(parent_scope.expansion),
        };
        let mut ident = ident.normalize_to_macros_2_0();
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
                if let break_result @ Some(..) = visitor(self, scope, use_prelude, ident) {
                    return break_result;
                }
            }

            scope = match scope {
                Scope::DeriveHelpers(expn_id) if expn_id != ExpnId::root() => {
                    // Derive helpers are not visible to code generated by bang or derive macros.
                    let expn_data = expn_id.expn_data();
                    match expn_data.kind {
                        ExpnKind::Root
                        | ExpnKind::Macro(MacroKind::Bang | MacroKind::Derive, _) => {
                            Scope::DeriveHelpersCompat
                        }
                        _ => Scope::DeriveHelpers(expn_data.parent),
                    }
                }
                Scope::DeriveHelpers(..) => Scope::DeriveHelpersCompat,
                Scope::DeriveHelpersCompat => Scope::MacroRules(parent_scope.macro_rules),
                Scope::MacroRules(macro_rules_scope) => match macro_rules_scope.get() {
                    MacroRulesScope::Binding(binding) => {
                        Scope::MacroRules(binding.parent_macro_rules_scope)
                    }
                    MacroRulesScope::Invocation(invoc_id) => {
                        Scope::MacroRules(self.invocation_parent_scopes[&invoc_id].macro_rules)
                    }
                    MacroRulesScope::Empty => Scope::Module(module),
                },
                Scope::CrateRoot => match ns {
                    TypeNS => {
                        ident.span.adjust(ExpnId::root());
                        Scope::ExternPrelude
                    }
                    ValueNS | MacroNS => break,
                },
                Scope::Module(module) => {
                    use_prelude = !module.no_implicit_prelude;
                    match self.hygienic_lexical_parent(module, &mut ident.span) {
                        Some(parent_module) => Scope::Module(parent_module),
                        None => {
                            ident.span.adjust(ExpnId::root());
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
        if ident.name == kw::Invalid {
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
            if let Some(res) = ribs[i].bindings.get(&rib_ident).cloned() {
                // The ident resolves to a type parameter or local variable.
                return Some(LexicalScopeBinding::Res(self.validate_res_from_ribs(
                    i,
                    rib_ident,
                    res,
                    record_used,
                    path_span,
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

            match module.kind {
                ModuleKind::Block(..) => {} // We can see through blocks
                _ => break,
            }
        }

        ident = normalized_ident;
        let mut poisoned = None;
        loop {
            let opt_module = if let Some(node_id) = record_used_id {
                self.hygienic_lexical_parent_with_compatibility_fallback(
                    module,
                    &mut ident.span,
                    node_id,
                    &mut poisoned,
                )
            } else {
                self.hygienic_lexical_parent(module, &mut ident.span)
            };
            module = unwrap_or!(opt_module, break);
            let adjusted_parent_scope = &ParentScope { module, ..*parent_scope };
            let result = self.resolve_ident_in_module_unadjusted(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                adjusted_parent_scope,
                record_used,
                path_span,
            );

            match result {
                Ok(binding) => {
                    if let Some(node_id) = poisoned {
                        self.lint_buffer.buffer_lint_with_diagnostic(
                            lint::builtin::PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
                            node_id,
                            ident.span,
                            &format!("cannot find {} `{}` in this scope", ns.descr(), ident),
                            BuiltinLintDiagnostics::ProcMacroDeriveResolutionFallback(ident.span),
                        );
                    }
                    return Some(LexicalScopeBinding::Item(binding));
                }
                Err(Determined) => continue,
                Err(Undetermined) => {
                    span_bug!(ident.span, "undetermined resolution during main resolution pass")
                }
            }
        }

        if !module.no_implicit_prelude {
            ident.span.adjust(ExpnId::root());
            if ns == TypeNS {
                if let Some(binding) = self.extern_prelude_get(ident, !record_used) {
                    return Some(LexicalScopeBinding::Item(binding));
                }
                if let Some(ident) = self.registered_tools.get(&ident) {
                    let binding =
                        (Res::ToolMod, ty::Visibility::Public, ident.span, ExpnId::root())
                            .to_name_binding(self.arenas);
                    return Some(LexicalScopeBinding::Item(binding));
                }
            }
            if let Some(prelude) = self.prelude {
                if let Ok(binding) = self.resolve_ident_in_module_unadjusted(
                    ModuleOrUniformRoot::Module(prelude),
                    ident,
                    ns,
                    parent_scope,
                    false,
                    path_span,
                ) {
                    return Some(LexicalScopeBinding::Item(binding));
                }
            }
        }

        if ns == TypeNS {
            if let Some(prim_ty) = self.primitive_type_table.primitive_types.get(&ident.name) {
                let binding =
                    (Res::PrimTy(*prim_ty), ty::Visibility::Public, DUMMY_SP, ExpnId::root())
                        .to_name_binding(self.arenas);
                return Some(LexicalScopeBinding::Item(binding));
            }
        }

        None
    }

    fn hygienic_lexical_parent(
        &mut self,
        module: Module<'a>,
        span: &mut Span,
    ) -> Option<Module<'a>> {
        if !module.expansion.outer_expn_is_descendant_of(span.ctxt()) {
            return Some(self.macro_def_scope(span.remove_mark()));
        }

        if let ModuleKind::Block(..) = module.kind {
            return Some(module.parent.unwrap().nearest_item_scope());
        }

        None
    }

    fn hygienic_lexical_parent_with_compatibility_fallback(
        &mut self,
        module: Module<'a>,
        span: &mut Span,
        node_id: NodeId,
        poisoned: &mut Option<NodeId>,
    ) -> Option<Module<'a>> {
        if let module @ Some(..) = self.hygienic_lexical_parent(module, span) {
            return module;
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
        if let Some(parent) = module.parent {
            // Inner module is inside the macro, parent module is outside of the macro.
            if module.expansion != parent.expansion
                && module.expansion.is_descendant_of(parent.expansion)
            {
                // The macro is a proc macro derive
                if let Some(def_id) = module.expansion.expn_data().macro_def_id {
                    if let Some(ext) = self.get_macro_by_def_id(def_id) {
                        if !ext.is_builtin
                            && ext.macro_kind() == MacroKind::Derive
                            && parent.expansion.outer_expn_is_descendant_of(span.ctxt())
                        {
                            *poisoned = Some(node_id);
                            return module.parent;
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
                        ParentScope { module: self.macro_def_scope(def), ..*parent_scope };
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
            Some(def) => self.macro_def_scope(def),
            None => {
                debug!(
                    "resolve_crate_root({:?}): found no mark (ident.span = {:?})",
                    ident, ident.span
                );
                return self.graph_root;
            }
        };
        let module = self.get_module(DefId { index: CRATE_DEF_INDEX, ..module.normal_ancestor_id });
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
        let mut module = self.get_module(module.normal_ancestor_id);
        while module.span.ctxt().normalize_to_macros_2_0() != *ctxt {
            let parent = module.parent.unwrap_or_else(|| self.macro_def_scope(ctxt.remove_mark()));
            module = self.get_module(parent.normal_ancestor_id);
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
                        } else {
                            (format!("maybe a missing crate `{}`?", ident), None)
                        }
                    } else if i == 0 {
                        if ident
                            .name
                            .as_str()
                            .chars()
                            .next()
                            .map_or(false, |c| c.is_ascii_uppercase())
                        {
                            (format!("use of undeclared type `{}`", ident), None)
                        } else {
                            (format!("use of undeclared crate or module `{}`", ident), None)
                        }
                    } else {
                        let mut msg =
                            format!("could not find `{}` in `{}`", ident, path[i - 1].ident);
                        if ns == TypeNS || ns == ValueNS {
                            let ns_to_try = if ns == TypeNS { ValueNS } else { TypeNS };
                            if let FindBindingResult::Binding(Ok(binding)) =
                                find_binding_in_ns(self, ns_to_try)
                            {
                                let mut found = |what| {
                                    msg = format!(
                                        "expected {}, found {} `{}` in `{}`",
                                        ns.descr(),
                                        what,
                                        ident,
                                        path[i - 1].ident
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
        all_ribs: &[Rib<'a>],
    ) -> Res {
        const CG_BUG_STR: &str = "min_const_generics resolve check didn't stop compilation";
        debug!("validate_res_from_ribs({:?})", res);
        let ribs = &all_ribs[rib_index + 1..];

        // An invalid forward use of a type parameter from a previous default.
        if let ForwardTyParamBanRibKind = all_ribs[rib_index].kind {
            if record_used {
                let res_error = if rib_ident.name == kw::SelfUpper {
                    ResolutionError::SelfInTyParamDefault
                } else {
                    ResolutionError::ForwardDeclaredTyParam
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
                        | ForwardTyParamBanRibKind => {
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
                        ConstantItemRibKind(_) => {
                            // Still doesn't deal with upvars
                            if record_used {
                                self.report_error(span, AttemptToUseNonConstantValueInConstant);
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
            Res::Def(DefKind::TyParam, _) | Res::SelfTy(..) => {
                let mut in_ty_param_default = false;
                for rib in ribs {
                    let has_generic_params = match rib.kind {
                        NormalRibKind
                        | ClosureOrAsyncRibKind
                        | AssocItemRibKind
                        | ModuleRibKind(..)
                        | MacroDefinition(..) => {
                            // Nothing to do. Continue.
                            continue;
                        }

                        // We only forbid constant items if we are inside of type defaults,
                        // for example `struct Foo<T, U = [u8; std::mem::size_of::<T>()]>`
                        ForwardTyParamBanRibKind => {
                            in_ty_param_default = true;
                            continue;
                        }
                        ConstantItemRibKind(trivial) => {
                            let features = self.session.features_untracked();
                            // HACK(min_const_generics): We currently only allow `N` or `{ N }`.
                            if !(trivial
                                || features.const_generics
                                || features.lazy_normalization_consts)
                            {
                                // HACK(min_const_generics): If we encounter `Self` in an anonymous constant
                                // we can't easily tell if it's generic at this stage, so we instead remember
                                // this and then enforce the self type to be concrete later on.
                                if let Res::SelfTy(trait_def, Some((impl_def, _))) = res {
                                    res = Res::SelfTy(trait_def, Some((impl_def, true)));
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

                            if in_ty_param_default {
                                if record_used {
                                    self.report_error(
                                        span,
                                        ResolutionError::ParamInAnonConstInTyDefault(
                                            rib_ident.name,
                                        ),
                                    );
                                }
                                return Res::Err;
                            } else {
                                continue;
                            }
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

                let mut in_ty_param_default = false;
                for rib in ribs {
                    let has_generic_params = match rib.kind {
                        NormalRibKind
                        | ClosureOrAsyncRibKind
                        | AssocItemRibKind
                        | ModuleRibKind(..)
                        | MacroDefinition(..) => continue,

                        // We only forbid constant items if we are inside of type defaults,
                        // for example `struct Foo<T, U = [u8; std::mem::size_of::<T>()]>`
                        ForwardTyParamBanRibKind => {
                            in_ty_param_default = true;
                            continue;
                        }
                        ConstantItemRibKind(trivial) => {
                            let features = self.session.features_untracked();
                            // HACK(min_const_generics): We currently only allow `N` or `{ N }`.
                            if !(trivial
                                || features.const_generics
                                || features.lazy_normalization_consts)
                            {
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

                            if in_ty_param_default {
                                if record_used {
                                    self.report_error(
                                        span,
                                        ResolutionError::ParamInAnonConstInTyDefault(
                                            rib_ident.name,
                                        ),
                                    );
                                }
                                return Res::Err;
                            } else {
                                continue;
                            }
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

    fn is_accessible_from(&self, vis: ty::Visibility, module: Module<'a>) -> bool {
        vis.is_accessible_from(module.normal_ancestor_id, self)
    }

    fn set_binding_parent_module(&mut self, binding: &'a NameBinding<'a>, module: Module<'a>) {
        if let Some(old_module) = self.binding_parent_modules.insert(PtrKey(binding), module) {
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
            self.binding_parent_modules.get(&PtrKey(macro_rules)),
            self.binding_parent_modules.get(&PtrKey(modularized)),
        ) {
            (Some(macro_rules), Some(modularized)) => {
                macro_rules.normal_ancestor_id == modularized.normal_ancestor_id
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
                diagnostics::show_candidates(&mut err, span, &candidates, instead, found_use);
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
            ModuleKind::Def(kind, _, _) => kind.descr(parent.def_id().unwrap()),
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
            self.extern_prelude.get(&ident).map(|entry| entry.introduced_by_item).unwrap_or(true);
        // Only suggest removing an import if both bindings are to the same def, if both spans
        // aren't dummy spans. Further, if both bindings are imports, then the ident must have
        // been introduced by a item.
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

    /// This function adds a suggestion to remove a unnecessary binding from an import that is
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
                // Remove the entire line if we cannot extend the span back, this indicates a
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
                    self.record_use(ident, TypeNS, binding, false);
                }
                Some(binding)
            } else {
                let crate_id = if !speculative {
                    self.crate_loader.process_path_extern(ident.name, ident.span)
                } else {
                    self.crate_loader.maybe_process_path_extern(ident.name)?
                };
                let crate_root = self.get_module(DefId { krate: crate_id, index: CRATE_DEF_INDEX });
                Some(
                    (crate_root, ty::Visibility::Public, DUMMY_SP, ExpnId::root())
                        .to_name_binding(self.arenas),
                )
            }
        })
    }

    /// This is equivalent to `get_traits_in_module_containing_item`, but without filtering by the associated item.
    ///
    /// This is used by rustdoc for intra-doc links.
    pub fn traits_in_scope(&mut self, module_id: DefId) -> Vec<TraitCandidate> {
        let module = self.get_module(module_id);
        module.ensure_traits(self);
        let traits = module.traits.borrow();
        let to_candidate =
            |this: &mut Self, &(trait_name, binding): &(Ident, &NameBinding<'_>)| TraitCandidate {
                def_id: binding.res().def_id(),
                import_ids: this.find_transitive_imports(&binding.kind, trait_name),
            };

        let mut candidates: Vec<_> =
            traits.as_ref().unwrap().iter().map(|x| to_candidate(self, x)).collect();

        if let Some(prelude) = self.prelude {
            if !module.no_implicit_prelude {
                prelude.ensure_traits(self);
                candidates.extend(
                    prelude.traits.borrow().as_ref().unwrap().iter().map(|x| to_candidate(self, x)),
                );
            }
        }

        candidates
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
        let module = self.get_module(module_id);
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

    /// Retrieves the span of the given `DefId` if `DefId` is in the local crate.
    #[inline]
    pub fn opt_span(&self, def_id: DefId) -> Option<Span> {
        if let Some(def_id) = def_id.as_local() { Some(self.def_id_to_span[def_id]) } else { None }
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
        result.push_str(&name.as_str());
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
