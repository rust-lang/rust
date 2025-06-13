//! This crate is responsible for the part of name resolution that doesn't require type checker.
//!
//! Module structure of the crate is built here.
//! Paths in macros, imports, expressions, types, patterns are resolved here.
//! Label and lifetime names are resolved here as well.
//!
//! Type-relative name resolution (methods, fields, associated items) happens in `rustc_hir_analysis`.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use diagnostics::{ImportSuggestion, LabelSuggestion, Suggestion};
use effective_visibilities::EffectiveVisibilitiesVisitor;
use errors::{ParamKindInEnumDiscriminant, ParamKindInNonTrivialAnonConst};
use imports::{Import, ImportData, ImportKind, NameResolution};
use late::{
    ForwardGenericParamBanReason, HasGenericParams, PathSource, PatternSource,
    UnnecessaryQualification,
};
use macros::{MacroRulesBinding, MacroRulesScope, MacroRulesScopeRef};
use rustc_arena::{DroplessArena, TypedArena};
use rustc_ast::expand::StrippedCfgItem;
use rustc_ast::node_id::NodeMap;
use rustc_ast::{
    self as ast, AngleBracketedArg, CRATE_NODE_ID, Crate, Expr, ExprKind, GenericArg, GenericArgs,
    LitKind, NodeId, Path, attr,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::FreezeReadGuard;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{Applicability, Diag, ErrCode, ErrorGuaranteed};
use rustc_expand::base::{DeriveResolution, SyntaxExtension, SyntaxExtensionKind};
use rustc_feature::BUILTIN_ATTRIBUTES;
use rustc_hir::def::Namespace::{self, *};
use rustc_hir::def::{
    self, CtorOf, DefKind, DocLinkResMap, LifetimeRes, NonMacroAttrKind, PartialRes, PerNS,
};
use rustc_hir::def_id::{CRATE_DEF_ID, CrateNum, DefId, LOCAL_CRATE, LocalDefId, LocalDefIdMap};
use rustc_hir::definitions::DisambiguatorState;
use rustc_hir::{PrimTy, TraitCandidate};
use rustc_metadata::creader::{CStore, CrateLoader};
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::privacy::EffectiveVisibilities;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{
    self, DelegationFnSig, Feed, MainDefinition, RegisteredTools, ResolverGlobalCtxt,
    ResolverOutputs, TyCtxt, TyCtxtFeed,
};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::lint::builtin::PRIVATE_MACRO_USE;
use rustc_session::lint::{BuiltinLintDiag, LintBuffer};
use rustc_span::hygiene::{ExpnId, LocalExpnId, MacroKind, SyntaxContext, Transparency};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};
use smallvec::{SmallVec, smallvec};
use tracing::debug;

type Res = def::Res<NodeId>;

mod build_reduced_graph;
mod check_unused;
mod def_collector;
mod diagnostics;
mod effective_visibilities;
mod errors;
mod ident;
mod imports;
mod late;
mod macros;
pub mod rustdoc;

pub use macros::registered_tools_ast;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

#[derive(Debug)]
enum Weak {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum Determinacy {
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
#[derive(Clone, Copy, Debug)]
enum Scope<'ra> {
    DeriveHelpers(LocalExpnId),
    DeriveHelpersCompat,
    MacroRules(MacroRulesScopeRef<'ra>),
    CrateRoot,
    // The node ID is for reporting the `PROC_MACRO_DERIVE_RESOLUTION_FALLBACK`
    // lint if it should be reported.
    Module(Module<'ra>, Option<NodeId>),
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
#[derive(Clone, Copy, Debug)]
enum ScopeSet<'ra> {
    /// All scopes with the given namespace.
    All(Namespace),
    /// Crate root, then extern prelude (used for mixed 2015-2018 mode in macros).
    AbsolutePath(Namespace),
    /// All scopes with macro namespace and the given macro kind restriction.
    Macro(MacroKind),
    /// All scopes with the given namespace, used for partially performing late resolution.
    /// The node id enables lints and is used for reporting them.
    Late(Namespace, Module<'ra>, Option<NodeId>),
}

/// Everything you need to know about a name's location to resolve it.
/// Serves as a starting point for the scope visitor.
/// This struct is currently used only for early resolution (imports and macros),
/// but not for late resolution yet.
#[derive(Clone, Copy, Debug)]
struct ParentScope<'ra> {
    module: Module<'ra>,
    expansion: LocalExpnId,
    macro_rules: MacroRulesScopeRef<'ra>,
    derives: &'ra [ast::Path],
}

impl<'ra> ParentScope<'ra> {
    /// Creates a parent scope with the passed argument used as the module scope component,
    /// and other scope components set to default empty values.
    fn module(module: Module<'ra>, resolver: &Resolver<'ra, '_>) -> ParentScope<'ra> {
        ParentScope {
            module,
            expansion: LocalExpnId::ROOT,
            macro_rules: resolver.arenas.alloc_macro_rules_scope(MacroRulesScope::Empty),
            derives: &[],
        }
    }
}

#[derive(Copy, Debug, Clone)]
struct InvocationParent {
    parent_def: LocalDefId,
    impl_trait_context: ImplTraitContext,
    in_attr: bool,
}

impl InvocationParent {
    const ROOT: Self = Self {
        parent_def: CRATE_DEF_ID,
        impl_trait_context: ImplTraitContext::Existential,
        in_attr: false,
    };
}

#[derive(Copy, Debug, Clone)]
enum ImplTraitContext {
    Existential,
    Universal,
    InBinding,
}

/// Used for tracking import use types which will be used for redundant import checking.
///
/// ### Used::Scope Example
///
/// ```rust,compile_fail
/// #![deny(redundant_imports)]
/// use std::mem::drop;
/// fn main() {
///     let s = Box::new(32);
///     drop(s);
/// }
/// ```
///
/// Used::Other is for other situations like module-relative uses.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
enum Used {
    Scope,
    Other,
}

#[derive(Debug)]
struct BindingError {
    name: Ident,
    origin: BTreeSet<Span>,
    target: BTreeSet<Span>,
    could_be_path: bool,
}

#[derive(Debug)]
enum ResolutionError<'ra> {
    /// Error E0401: can't use type or const parameters from outer item.
    GenericParamsFromOuterItem(Res, HasGenericParams, DefKind),
    /// Error E0403: the name is already used for a type or const parameter in this generic
    /// parameter list.
    NameAlreadyUsedInParameterList(Ident, Span),
    /// Error E0407: method is not a member of trait.
    MethodNotMemberOfTrait(Ident, String, Option<Symbol>),
    /// Error E0437: type is not a member of trait.
    TypeNotMemberOfTrait(Ident, String, Option<Symbol>),
    /// Error E0438: const is not a member of trait.
    ConstNotMemberOfTrait(Ident, String, Option<Symbol>),
    /// Error E0408: variable `{}` is not bound in all patterns.
    VariableNotBoundInPattern(BindingError, ParentScope<'ra>),
    /// Error E0409: variable `{}` is bound in inconsistent ways within the same match arm.
    VariableBoundWithDifferentMode(Ident, Span),
    /// Error E0415: identifier is bound more than once in this parameter list.
    IdentifierBoundMoreThanOnceInParameterList(Ident),
    /// Error E0416: identifier is bound more than once in the same pattern.
    IdentifierBoundMoreThanOnceInSamePattern(Ident),
    /// Error E0426: use of undeclared label.
    UndeclaredLabel { name: Symbol, suggestion: Option<LabelSuggestion> },
    /// Error E0429: `self` imports are only allowed within a `{ }` list.
    SelfImportsOnlyAllowedWithin { root: bool, span_with_rename: Span },
    /// Error E0430: `self` import can only appear once in the list.
    SelfImportCanOnlyAppearOnceInTheList,
    /// Error E0431: `self` import can only appear in an import list with a non-empty prefix.
    SelfImportOnlyInImportListWithNonEmptyPrefix,
    /// Error E0433: failed to resolve.
    FailedToResolve {
        segment: Option<Symbol>,
        label: String,
        suggestion: Option<Suggestion>,
        module: Option<ModuleOrUniformRoot<'ra>>,
    },
    /// Error E0434: can't capture dynamic environment in a fn item.
    CannotCaptureDynamicEnvironmentInFnItem,
    /// Error E0435: attempt to use a non-constant value in a constant.
    AttemptToUseNonConstantValueInConstant {
        ident: Ident,
        suggestion: &'static str,
        current: &'static str,
        type_span: Option<Span>,
    },
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
    ForwardDeclaredGenericParam(Symbol, ForwardGenericParamBanReason),
    // FIXME(generic_const_parameter_types): This should give custom output specifying it's only
    // problematic to use *forward declared* parameters when the feature is enabled.
    /// ERROR E0770: the type of const parameters must not depend on other generic parameters.
    ParamInTyOfConstParam { name: Symbol },
    /// generic parameters must not be used inside const evaluations.
    ///
    /// This error is only emitted when using `min_const_generics`.
    ParamInNonTrivialAnonConst { name: Symbol, param_kind: ParamKindInNonTrivialAnonConst },
    /// generic parameters must not be used inside enum discriminants.
    ///
    /// This error is emitted even with `generic_const_exprs`.
    ParamInEnumDiscriminant { name: Symbol, param_kind: ParamKindInEnumDiscriminant },
    /// Error E0735: generic parameters with a default cannot use `Self`
    ForwardDeclaredSelf(ForwardGenericParamBanReason),
    /// Error E0767: use of unreachable label
    UnreachableLabel { name: Symbol, definition_span: Span, suggestion: Option<LabelSuggestion> },
    /// Error E0323, E0324, E0325: mismatch between trait item and impl item.
    TraitImplMismatch {
        name: Ident,
        kind: &'static str,
        trait_path: String,
        trait_item_span: Span,
        code: ErrCode,
    },
    /// Error E0201: multiple impl items for the same trait item.
    TraitImplDuplicate { name: Ident, trait_item_span: Span, old_span: Span },
    /// Inline asm `sym` operand must refer to a `fn` or `static`.
    InvalidAsmSym,
    /// `self` used instead of `Self` in a generic parameter
    LowercaseSelf,
    /// A never pattern has a binding.
    BindingInNeverPattern,
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
struct Segment {
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
        names_to_string(segments.iter().map(|seg| seg.ident.name))
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
                GenericArgs::ParenthesizedElided(span) => (*span, true),
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
#[derive(Debug, Copy, Clone)]
enum LexicalScopeBinding<'ra> {
    Item(NameBinding<'ra>),
    Res(Res),
}

impl<'ra> LexicalScopeBinding<'ra> {
    fn res(self) -> Res {
        match self {
            LexicalScopeBinding::Item(binding) => binding.res(),
            LexicalScopeBinding::Res(res) => res,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum ModuleOrUniformRoot<'ra> {
    /// Regular module.
    Module(Module<'ra>),

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

#[derive(Debug)]
enum PathResult<'ra> {
    Module(ModuleOrUniformRoot<'ra>),
    NonModule(PartialRes),
    Indeterminate,
    Failed {
        span: Span,
        label: String,
        suggestion: Option<Suggestion>,
        is_error_from_last_segment: bool,
        /// The final module being resolved, for instance:
        ///
        /// ```compile_fail
        /// mod a {
        ///     mod b {
        ///         mod c {}
        ///     }
        /// }
        ///
        /// use a::not_exist::c;
        /// ```
        ///
        /// In this case, `module` will point to `a`.
        module: Option<ModuleOrUniformRoot<'ra>>,
        /// The segment name of target
        segment_name: Symbol,
        error_implied_by_parse_error: bool,
    },
}

impl<'ra> PathResult<'ra> {
    fn failed(
        ident: Ident,
        is_error_from_last_segment: bool,
        finalize: bool,
        error_implied_by_parse_error: bool,
        module: Option<ModuleOrUniformRoot<'ra>>,
        label_and_suggestion: impl FnOnce() -> (String, Option<Suggestion>),
    ) -> PathResult<'ra> {
        let (label, suggestion) =
            if finalize { label_and_suggestion() } else { (String::new(), None) };
        PathResult::Failed {
            span: ident.span,
            segment_name: ident.name,
            label,
            suggestion,
            is_error_from_last_segment,
            module,
            error_implied_by_parse_error,
        }
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
    ///   The crate root will have `None` for the symbol.
    /// * A trait or an enum (it implicitly contains associated types, methods and variant
    ///   constructors).
    Def(DefKind, DefId, Option<Symbol>),
}

impl ModuleKind {
    /// Get name of the module.
    fn name(&self) -> Option<Symbol> {
        match *self {
            ModuleKind::Block => None,
            ModuleKind::Def(.., name) => name,
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

impl BindingKey {
    fn new(ident: Ident, ns: Namespace) -> Self {
        let ident = ident.normalize_to_macros_2_0();
        BindingKey { ident, ns, disambiguator: 0 }
    }
}

type Resolutions<'ra> = RefCell<FxIndexMap<BindingKey, &'ra RefCell<NameResolution<'ra>>>>;

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
struct ModuleData<'ra> {
    /// The direct parent module (it may not be a `mod`, however).
    parent: Option<Module<'ra>>,
    /// What kind of module this is, because this may not be a `mod`.
    kind: ModuleKind,

    /// Mapping between names and their (possibly in-progress) resolutions in this module.
    /// Resolutions in modules from other crates are not populated until accessed.
    lazy_resolutions: Resolutions<'ra>,
    /// True if this is a module from other crate that needs to be populated on access.
    populate_on_access: Cell<bool>,

    /// Macro invocations that can expand into items in this module.
    unexpanded_invocations: RefCell<FxHashSet<LocalExpnId>>,

    /// Whether `#[no_implicit_prelude]` is active.
    no_implicit_prelude: bool,

    glob_importers: RefCell<Vec<Import<'ra>>>,
    globs: RefCell<Vec<Import<'ra>>>,

    /// Used to memoize the traits in this module for faster searches through all traits in scope.
    traits: RefCell<Option<Box<[(Ident, NameBinding<'ra>)]>>>,

    /// Span of the module itself. Used for error reporting.
    span: Span,

    expansion: ExpnId,
}

/// All modules are unique and allocated on a same arena,
/// so we can use referential equality to compare them.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[rustc_pass_by_value]
struct Module<'ra>(Interned<'ra, ModuleData<'ra>>);

// Allows us to use Interned without actually enforcing (via Hash/PartialEq/...) uniqueness of the
// contained data.
// FIXME: We may wish to actually have at least debug-level assertions that Interned's guarantees
// are upheld.
impl std::hash::Hash for ModuleData<'_> {
    fn hash<H>(&self, _: &mut H)
    where
        H: std::hash::Hasher,
    {
        unreachable!()
    }
}

impl<'ra> ModuleData<'ra> {
    fn new(
        parent: Option<Module<'ra>>,
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
}

impl<'ra> Module<'ra> {
    fn for_each_child<'tcx, R, F>(self, resolver: &mut R, mut f: F)
    where
        R: AsMut<Resolver<'ra, 'tcx>>,
        F: FnMut(&mut R, Ident, Namespace, NameBinding<'ra>),
    {
        for (key, name_resolution) in resolver.as_mut().resolutions(self).borrow().iter() {
            if let Some(binding) = name_resolution.borrow().binding {
                f(resolver, key.ident, key.ns, binding);
            }
        }
    }

    /// This modifies `self` in place. The traits will be stored in `self.traits`.
    fn ensure_traits<'tcx, R>(self, resolver: &mut R)
    where
        R: AsMut<Resolver<'ra, 'tcx>>,
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

    fn res(self) -> Option<Res> {
        match self.kind {
            ModuleKind::Def(kind, def_id, _) => Some(Res::Def(kind, def_id)),
            _ => None,
        }
    }

    // Public for rustdoc.
    fn def_id(self) -> DefId {
        self.opt_def_id().expect("`ModuleData::def_id` is called on a block module")
    }

    fn opt_def_id(self) -> Option<DefId> {
        match self.kind {
            ModuleKind::Def(_, def_id, _) => Some(def_id),
            _ => None,
        }
    }

    // `self` resolves to the first module ancestor that `is_normal`.
    fn is_normal(self) -> bool {
        matches!(self.kind, ModuleKind::Def(DefKind::Mod, _, _))
    }

    fn is_trait(self) -> bool {
        matches!(self.kind, ModuleKind::Def(DefKind::Trait, _, _))
    }

    fn nearest_item_scope(self) -> Module<'ra> {
        match self.kind {
            ModuleKind::Def(DefKind::Enum | DefKind::Trait, ..) => {
                self.parent.expect("enum or trait module without a parent")
            }
            _ => self,
        }
    }

    /// The [`DefId`] of the nearest `mod` item ancestor (which may be this module).
    /// This may be the crate root.
    fn nearest_parent_mod(self) -> DefId {
        match self.kind {
            ModuleKind::Def(DefKind::Mod, def_id, _) => def_id,
            _ => self.parent.expect("non-root module without parent").nearest_parent_mod(),
        }
    }

    fn is_ancestor_of(self, mut other: Self) -> bool {
        while self != other {
            if let Some(parent) = other.parent {
                other = parent;
            } else {
                return false;
            }
        }
        true
    }
}

impl<'ra> std::ops::Deref for Module<'ra> {
    type Target = ModuleData<'ra>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'ra> fmt::Debug for Module<'ra> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.res())
    }
}

/// Records a possibly-private value, type, or module definition.
#[derive(Clone, Copy, Debug)]
struct NameBindingData<'ra> {
    kind: NameBindingKind<'ra>,
    ambiguity: Option<(NameBinding<'ra>, AmbiguityKind)>,
    /// Produce a warning instead of an error when reporting ambiguities inside this binding.
    /// May apply to indirect ambiguities under imports, so `ambiguity.is_some()` is not required.
    warn_ambiguity: bool,
    expansion: LocalExpnId,
    span: Span,
    vis: ty::Visibility<DefId>,
}

/// All name bindings are unique and allocated on a same arena,
/// so we can use referential equality to compare them.
type NameBinding<'ra> = Interned<'ra, NameBindingData<'ra>>;

// Allows us to use Interned without actually enforcing (via Hash/PartialEq/...) uniqueness of the
// contained data.
// FIXME: We may wish to actually have at least debug-level assertions that Interned's guarantees
// are upheld.
impl std::hash::Hash for NameBindingData<'_> {
    fn hash<H>(&self, _: &mut H)
    where
        H: std::hash::Hasher,
    {
        unreachable!()
    }
}

trait ToNameBinding<'ra> {
    fn to_name_binding(self, arenas: &'ra ResolverArenas<'ra>) -> NameBinding<'ra>;
}

impl<'ra> ToNameBinding<'ra> for NameBinding<'ra> {
    fn to_name_binding(self, _: &'ra ResolverArenas<'ra>) -> NameBinding<'ra> {
        self
    }
}

#[derive(Clone, Copy, Debug)]
enum NameBindingKind<'ra> {
    Res(Res),
    Module(Module<'ra>),
    Import { binding: NameBinding<'ra>, import: Import<'ra> },
}

impl<'ra> NameBindingKind<'ra> {
    /// Is this a name binding of an import?
    fn is_import(&self) -> bool {
        matches!(*self, NameBindingKind::Import { .. })
    }
}

#[derive(Debug)]
struct PrivacyError<'ra> {
    ident: Ident,
    binding: NameBinding<'ra>,
    dedup_span: Span,
    outermost_res: Option<(Res, Ident)>,
    parent_scope: ParentScope<'ra>,
    /// Is the format `use a::{b,c}`?
    single_nested: bool,
}

#[derive(Debug)]
struct UseError<'a> {
    err: Diag<'a>,
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
    /// Whether the expected source is a call
    is_call: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum AmbiguityKind {
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

struct AmbiguityError<'ra> {
    kind: AmbiguityKind,
    ident: Ident,
    b1: NameBinding<'ra>,
    b2: NameBinding<'ra>,
    misc1: AmbiguityErrorMisc,
    misc2: AmbiguityErrorMisc,
    warning: bool,
}

impl<'ra> NameBindingData<'ra> {
    fn module(&self) -> Option<Module<'ra>> {
        match self.kind {
            NameBindingKind::Module(module) => Some(module),
            NameBindingKind::Import { binding, .. } => binding.module(),
            _ => None,
        }
    }

    fn res(&self) -> Res {
        match self.kind {
            NameBindingKind::Res(res) => res,
            NameBindingKind::Module(module) => module.res().unwrap(),
            NameBindingKind::Import { binding, .. } => binding.res(),
        }
    }

    fn is_ambiguity_recursive(&self) -> bool {
        self.ambiguity.is_some()
            || match self.kind {
                NameBindingKind::Import { binding, .. } => binding.is_ambiguity_recursive(),
                _ => false,
            }
    }

    fn warn_ambiguity_recursive(&self) -> bool {
        self.warn_ambiguity
            || match self.kind {
                NameBindingKind::Import { binding, .. } => binding.warn_ambiguity_recursive(),
                _ => false,
            }
    }

    fn is_possibly_imported_variant(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { binding, .. } => binding.is_possibly_imported_variant(),
            NameBindingKind::Res(Res::Def(
                DefKind::Variant | DefKind::Ctor(CtorOf::Variant, ..),
                _,
            )) => true,
            NameBindingKind::Res(..) | NameBindingKind::Module(..) => false,
        }
    }

    fn is_extern_crate(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { import, .. } => {
                matches!(import.kind, ImportKind::ExternCrate { .. })
            }
            NameBindingKind::Module(module)
                if let ModuleKind::Def(DefKind::Mod, def_id, _) = module.kind =>
            {
                def_id.is_crate_root()
            }
            _ => false,
        }
    }

    fn is_import(&self) -> bool {
        matches!(self.kind, NameBindingKind::Import { .. })
    }

    /// The binding introduced by `#[macro_export] macro_rules` is a public import, but it might
    /// not be perceived as such by users, so treat it as a non-import in some diagnostics.
    fn is_import_user_facing(&self) -> bool {
        matches!(self.kind, NameBindingKind::Import { import, .. }
            if !matches!(import.kind, ImportKind::MacroExport))
    }

    fn is_glob_import(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { import, .. } => import.is_glob(),
            _ => false,
        }
    }

    fn is_assoc_item(&self) -> bool {
        matches!(self.res(), Res::Def(DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy, _))
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
        binding: NameBinding<'_>,
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

    // Its purpose is to postpone the determination of a single binding because
    // we can't predict whether it will be overwritten by recently expanded macros.
    // FIXME: How can we integrate it with the `update_resolution`?
    fn determined(&self) -> bool {
        match &self.kind {
            NameBindingKind::Import { binding, import, .. } if import.is_glob() => {
                import.parent_scope.module.unexpanded_invocations.borrow().is_empty()
                    && binding.determined()
            }
            _ => true,
        }
    }
}

#[derive(Default, Clone)]
struct ExternPreludeEntry<'ra> {
    binding: Option<NameBinding<'ra>>,
    introduced_by_item: bool,
}

impl ExternPreludeEntry<'_> {
    fn is_import(&self) -> bool {
        self.binding.is_some_and(|binding| binding.is_import())
    }
}

struct DeriveData {
    resolutions: Vec<DeriveResolution>,
    helper_attrs: Vec<(usize, Ident)>,
    has_derive_copy: bool,
}

struct MacroData {
    ext: Arc<SyntaxExtension>,
    rule_spans: Vec<(usize, Span)>,
    macro_rules: bool,
}

impl MacroData {
    fn new(ext: Arc<SyntaxExtension>) -> MacroData {
        MacroData { ext, rule_spans: Vec::new(), macro_rules: false }
    }
}

/// The main resolver class.
///
/// This is the visitor that walks the whole crate.
pub struct Resolver<'ra, 'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Item with a given `LocalDefId` was defined during macro expansion with ID `ExpnId`.
    expn_that_defined: FxHashMap<LocalDefId, ExpnId>,

    graph_root: Module<'ra>,

    prelude: Option<Module<'ra>>,
    extern_prelude: FxIndexMap<Ident, ExternPreludeEntry<'ra>>,

    /// N.B., this is used only for better diagnostics, not name resolution itself.
    field_names: LocalDefIdMap<Vec<Ident>>,

    /// Span of the privacy modifier in fields of an item `DefId` accessible with dot syntax.
    /// Used for hints during error reporting.
    field_visibility_spans: FxHashMap<DefId, Vec<Span>>,

    /// All imports known to succeed or fail.
    determined_imports: Vec<Import<'ra>>,

    /// All non-determined imports.
    indeterminate_imports: Vec<Import<'ra>>,

    // Spans for local variables found during pattern resolution.
    // Used for suggestions during error reporting.
    pat_span_map: NodeMap<Span>,

    /// Resolutions for nodes that have a single resolution.
    partial_res_map: NodeMap<PartialRes>,
    /// Resolutions for import nodes, which have multiple resolutions in different namespaces.
    import_res_map: NodeMap<PerNS<Option<Res>>>,
    /// An import will be inserted into this map if it has been used.
    import_use_map: FxHashMap<Import<'ra>, Used>,
    /// Resolutions for labels (node IDs of their corresponding blocks or loops).
    label_res_map: NodeMap<NodeId>,
    /// Resolutions for lifetimes.
    lifetimes_res_map: NodeMap<LifetimeRes>,
    /// Lifetime parameters that lowering will have to introduce.
    extra_lifetime_params_map: NodeMap<Vec<(Ident, NodeId, LifetimeRes)>>,

    /// `CrateNum` resolutions of `extern crate` items.
    extern_crate_map: UnordMap<LocalDefId, CrateNum>,
    module_children: LocalDefIdMap<Vec<ModChild>>,
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
    block_map: NodeMap<Module<'ra>>,
    /// A fake module that contains no definition and no prelude. Used so that
    /// some AST passes can generate identifiers that only resolve to local or
    /// lang items.
    empty_module: Module<'ra>,
    module_map: FxIndexMap<DefId, Module<'ra>>,
    binding_parent_modules: FxHashMap<NameBinding<'ra>, Module<'ra>>,

    underscore_disambiguator: u32,

    /// Maps glob imports to the names of items actually imported.
    glob_map: FxIndexMap<LocalDefId, FxIndexSet<Symbol>>,
    glob_error: Option<ErrorGuaranteed>,
    visibilities_for_hashing: Vec<(LocalDefId, ty::Visibility)>,
    used_imports: FxHashSet<NodeId>,
    maybe_unused_trait_imports: FxIndexSet<LocalDefId>,

    /// Privacy errors are delayed until the end in order to deduplicate them.
    privacy_errors: Vec<PrivacyError<'ra>>,
    /// Ambiguity errors are delayed for deduplication.
    ambiguity_errors: Vec<AmbiguityError<'ra>>,
    /// `use` injections are delayed for better placement and deduplication.
    use_injections: Vec<UseError<'tcx>>,
    /// Crate-local macro expanded `macro_export` referred to by a module-relative path.
    macro_expanded_macro_export_errors: BTreeSet<(Span, Span)>,

    arenas: &'ra ResolverArenas<'ra>,
    dummy_binding: NameBinding<'ra>,
    builtin_types_bindings: FxHashMap<Symbol, NameBinding<'ra>>,
    builtin_attrs_bindings: FxHashMap<Symbol, NameBinding<'ra>>,
    registered_tool_bindings: FxHashMap<Ident, NameBinding<'ra>>,
    /// Binding for implicitly declared names that come with a module,
    /// like `self` (not yet used), or `crate`/`$crate` (for root modules).
    module_self_bindings: FxHashMap<Module<'ra>, NameBinding<'ra>>,

    used_extern_options: FxHashSet<Symbol>,
    macro_names: FxHashSet<Ident>,
    builtin_macros: FxHashMap<Symbol, SyntaxExtensionKind>,
    registered_tools: &'tcx RegisteredTools,
    macro_use_prelude: FxIndexMap<Symbol, NameBinding<'ra>>,
    macro_map: FxHashMap<DefId, MacroData>,
    dummy_ext_bang: Arc<SyntaxExtension>,
    dummy_ext_derive: Arc<SyntaxExtension>,
    non_macro_attr: MacroData,
    local_macro_def_scopes: FxHashMap<LocalDefId, Module<'ra>>,
    ast_transform_scopes: FxHashMap<LocalExpnId, Module<'ra>>,
    unused_macros: FxIndexMap<LocalDefId, (NodeId, Ident)>,
    /// A map from the macro to all its potentially unused arms.
    unused_macro_rules: FxIndexMap<NodeId, UnordMap<usize, (Ident, Span)>>,
    proc_macro_stubs: FxHashSet<LocalDefId>,
    /// Traces collected during macro resolution and validated when it's complete.
    single_segment_macro_resolutions:
        Vec<(Ident, MacroKind, ParentScope<'ra>, Option<NameBinding<'ra>>, Option<Span>)>,
    multi_segment_macro_resolutions:
        Vec<(Vec<Segment>, Span, MacroKind, ParentScope<'ra>, Option<Res>, Namespace)>,
    builtin_attrs: Vec<(Ident, ParentScope<'ra>)>,
    /// `derive(Copy)` marks items they are applied to so they are treated specially later.
    /// Derive macros cannot modify the item themselves and have to store the markers in the global
    /// context, so they attach the markers to derive container IDs using this resolver table.
    containers_deriving_copy: FxHashSet<LocalExpnId>,
    /// Parent scopes in which the macros were invoked.
    /// FIXME: `derives` are missing in these parent scopes and need to be taken from elsewhere.
    invocation_parent_scopes: FxHashMap<LocalExpnId, ParentScope<'ra>>,
    /// `macro_rules` scopes *produced* by expanding the macro invocations,
    /// include all the `macro_rules` items and other invocations generated by them.
    output_macro_rules_scopes: FxHashMap<LocalExpnId, MacroRulesScopeRef<'ra>>,
    /// `macro_rules` scopes produced by `macro_rules` item definitions.
    macro_rules_scopes: FxHashMap<LocalDefId, MacroRulesScopeRef<'ra>>,
    /// Helper attributes that are in scope for the given expansion.
    helper_attrs: FxHashMap<LocalExpnId, Vec<(Ident, NameBinding<'ra>)>>,
    /// Ready or in-progress results of resolving paths inside the `#[derive(...)]` attribute
    /// with the given `ExpnId`.
    derive_data: FxHashMap<LocalExpnId, DeriveData>,

    /// Avoid duplicated errors for "name already defined".
    name_already_seen: FxHashMap<Symbol, Span>,

    potentially_unused_imports: Vec<Import<'ra>>,

    potentially_unnecessary_qualifications: Vec<UnnecessaryQualification<'ra>>,

    /// Table for mapping struct IDs into struct constructor IDs,
    /// it's not used during normal resolution, only for better error reporting.
    /// Also includes of list of each fields visibility
    struct_constructors: LocalDefIdMap<(Res, ty::Visibility<DefId>, Vec<ty::Visibility<DefId>>)>,

    lint_buffer: LintBuffer,

    next_node_id: NodeId,

    node_id_to_def_id: NodeMap<Feed<'tcx, LocalDefId>>,

    disambiguator: DisambiguatorState,

    /// Indices of unnamed struct or variant fields with unresolved attributes.
    placeholder_field_indices: FxHashMap<NodeId, usize>,
    /// When collecting definitions from an AST fragment produced by a macro invocation `ExpnId`
    /// we know what parent node that fragment should be attached to thanks to this table,
    /// and how the `impl Trait` fragments were introduced.
    invocation_parents: FxHashMap<LocalExpnId, InvocationParent>,

    legacy_const_generic_args: FxHashMap<DefId, Option<Vec<usize>>>,
    /// Amount of lifetime parameters for each item in the crate.
    item_generics_num_lifetimes: FxHashMap<LocalDefId, usize>,
    delegation_fn_sigs: LocalDefIdMap<DelegationFnSig>,

    main_def: Option<MainDefinition>,
    trait_impls: FxIndexMap<DefId, Vec<LocalDefId>>,
    /// A list of proc macro LocalDefIds, written out in the order in which
    /// they are declared in the static array generated by proc_macro_harness.
    proc_macros: Vec<LocalDefId>,
    confused_type_with_std_module: FxIndexMap<Span, Span>,
    /// Whether lifetime elision was successful.
    lifetime_elision_allowed: FxHashSet<NodeId>,

    /// Names of items that were stripped out via cfg with their corresponding cfg meta item.
    stripped_cfg_items: Vec<StrippedCfgItem<NodeId>>,

    effective_visibilities: EffectiveVisibilities,
    doc_link_resolutions: FxIndexMap<LocalDefId, DocLinkResMap>,
    doc_link_traits_in_scope: FxIndexMap<LocalDefId, Vec<DefId>>,
    all_macro_rules: FxHashSet<Symbol>,

    /// Invocation ids of all glob delegations.
    glob_delegation_invoc_ids: FxHashSet<LocalExpnId>,
    /// Analogue of module `unexpanded_invocations` but in trait impls, excluding glob delegations.
    /// Needed because glob delegations wait for all other neighboring macros to expand.
    impl_unexpanded_invocations: FxHashMap<LocalDefId, FxHashSet<LocalExpnId>>,
    /// Simplified analogue of module `resolutions` but in trait impls, excluding glob delegations.
    /// Needed because glob delegations exclude explicitly defined names.
    impl_binding_keys: FxHashMap<LocalDefId, FxHashSet<BindingKey>>,

    /// This is the `Span` where an `extern crate foo;` suggestion would be inserted, if `foo`
    /// could be a crate that wasn't imported. For diagnostics use only.
    current_crate_outer_attr_insert_span: Span,

    mods_with_parse_errors: FxHashSet<DefId>,
}

/// This provides memory for the rest of the crate. The `'ra` lifetime that is
/// used by many types in this crate is an abbreviation of `ResolverArenas`.
#[derive(Default)]
pub struct ResolverArenas<'ra> {
    modules: TypedArena<ModuleData<'ra>>,
    local_modules: RefCell<Vec<Module<'ra>>>,
    imports: TypedArena<ImportData<'ra>>,
    name_resolutions: TypedArena<RefCell<NameResolution<'ra>>>,
    ast_paths: TypedArena<ast::Path>,
    dropless: DroplessArena,
}

impl<'ra> ResolverArenas<'ra> {
    fn new_module(
        &'ra self,
        parent: Option<Module<'ra>>,
        kind: ModuleKind,
        expn_id: ExpnId,
        span: Span,
        no_implicit_prelude: bool,
        module_map: &mut FxIndexMap<DefId, Module<'ra>>,
        module_self_bindings: &mut FxHashMap<Module<'ra>, NameBinding<'ra>>,
    ) -> Module<'ra> {
        let module = Module(Interned::new_unchecked(self.modules.alloc(ModuleData::new(
            parent,
            kind,
            expn_id,
            span,
            no_implicit_prelude,
        ))));
        let def_id = module.opt_def_id();
        if def_id.is_none_or(|def_id| def_id.is_local()) {
            self.local_modules.borrow_mut().push(module);
        }
        if let Some(def_id) = def_id {
            module_map.insert(def_id, module);
            let vis = ty::Visibility::<DefId>::Public;
            let binding = (module, vis, module.span, LocalExpnId::ROOT).to_name_binding(self);
            module_self_bindings.insert(module, binding);
        }
        module
    }
    fn local_modules(&'ra self) -> std::cell::Ref<'ra, Vec<Module<'ra>>> {
        self.local_modules.borrow()
    }
    fn alloc_name_binding(&'ra self, name_binding: NameBindingData<'ra>) -> NameBinding<'ra> {
        Interned::new_unchecked(self.dropless.alloc(name_binding))
    }
    fn alloc_import(&'ra self, import: ImportData<'ra>) -> Import<'ra> {
        Interned::new_unchecked(self.imports.alloc(import))
    }
    fn alloc_name_resolution(&'ra self) -> &'ra RefCell<NameResolution<'ra>> {
        self.name_resolutions.alloc(Default::default())
    }
    fn alloc_macro_rules_scope(&'ra self, scope: MacroRulesScope<'ra>) -> MacroRulesScopeRef<'ra> {
        Interned::new_unchecked(self.dropless.alloc(Cell::new(scope)))
    }
    fn alloc_macro_rules_binding(
        &'ra self,
        binding: MacroRulesBinding<'ra>,
    ) -> &'ra MacroRulesBinding<'ra> {
        self.dropless.alloc(binding)
    }
    fn alloc_ast_paths(&'ra self, paths: &[ast::Path]) -> &'ra [ast::Path] {
        self.ast_paths.alloc_from_iter(paths.iter().cloned())
    }
    fn alloc_pattern_spans(&'ra self, spans: impl Iterator<Item = Span>) -> &'ra [Span] {
        self.dropless.alloc_from_iter(spans)
    }
}

impl<'ra, 'tcx> AsMut<Resolver<'ra, 'tcx>> for Resolver<'ra, 'tcx> {
    fn as_mut(&mut self) -> &mut Resolver<'ra, 'tcx> {
        self
    }
}

impl<'tcx> Resolver<'_, 'tcx> {
    fn opt_local_def_id(&self, node: NodeId) -> Option<LocalDefId> {
        self.opt_feed(node).map(|f| f.key())
    }

    fn local_def_id(&self, node: NodeId) -> LocalDefId {
        self.feed(node).key()
    }

    fn opt_feed(&self, node: NodeId) -> Option<Feed<'tcx, LocalDefId>> {
        self.node_id_to_def_id.get(&node).copied()
    }

    fn feed(&self, node: NodeId) -> Feed<'tcx, LocalDefId> {
        self.opt_feed(node).unwrap_or_else(|| panic!("no entry for node id: `{node:?}`"))
    }

    fn local_def_kind(&self, node: NodeId) -> DefKind {
        self.tcx.def_kind(self.local_def_id(node))
    }

    /// Adds a definition with a parent definition.
    fn create_def(
        &mut self,
        parent: LocalDefId,
        node_id: ast::NodeId,
        name: Option<Symbol>,
        def_kind: DefKind,
        expn_id: ExpnId,
        span: Span,
    ) -> TyCtxtFeed<'tcx, LocalDefId> {
        assert!(
            !self.node_id_to_def_id.contains_key(&node_id),
            "adding a def for node-id {:?}, name {:?}, data {:?} but a previous def exists: {:?}",
            node_id,
            name,
            def_kind,
            self.tcx.definitions_untracked().def_key(self.node_id_to_def_id[&node_id].key()),
        );

        // FIXME: remove `def_span` body, pass in the right spans here and call `tcx.at().create_def()`
        let feed = self.tcx.create_def(parent, name, def_kind, None, &mut self.disambiguator);
        let def_id = feed.def_id();

        // Create the definition.
        if expn_id != ExpnId::root() {
            self.expn_that_defined.insert(def_id, expn_id);
        }

        // A relative span's parent must be an absolute span.
        debug_assert_eq!(span.data_untracked().parent, None);
        let _id = self.tcx.untracked().source_span.push(span);
        debug_assert_eq!(_id, def_id);

        // Some things for which we allocate `LocalDefId`s don't correspond to
        // anything in the AST, so they don't have a `NodeId`. For these cases
        // we don't need a mapping from `NodeId` to `LocalDefId`.
        if node_id != ast::DUMMY_NODE_ID {
            debug!("create_def: def_id_to_node_id[{:?}] <-> {:?}", def_id, node_id);
            self.node_id_to_def_id.insert(node_id, feed.downgrade());
        }

        feed
    }

    fn item_generics_num_lifetimes(&self, def_id: DefId) -> usize {
        if let Some(def_id) = def_id.as_local() {
            self.item_generics_num_lifetimes[&def_id]
        } else {
            self.tcx.generics_of(def_id).own_counts().lifetimes
        }
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    /// This function is very slow, as it iterates over the entire
    /// [Resolver::node_id_to_def_id] map just to find the [NodeId]
    /// that corresponds to the given [LocalDefId]. Only use this in
    /// diagnostics code paths.
    fn def_id_to_node_id(&self, def_id: LocalDefId) -> NodeId {
        self.node_id_to_def_id
            .items()
            .filter(|(_, v)| v.key() == def_id)
            .map(|(k, _)| *k)
            .get_only()
            .unwrap()
    }
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        attrs: &[ast::Attribute],
        crate_span: Span,
        current_crate_outer_attr_insert_span: Span,
        arenas: &'ra ResolverArenas<'ra>,
    ) -> Resolver<'ra, 'tcx> {
        let root_def_id = CRATE_DEF_ID.to_def_id();
        let mut module_map = FxIndexMap::default();
        let mut module_self_bindings = FxHashMap::default();
        let graph_root = arenas.new_module(
            None,
            ModuleKind::Def(DefKind::Mod, root_def_id, None),
            ExpnId::root(),
            crate_span,
            attr::contains_name(attrs, sym::no_implicit_prelude),
            &mut module_map,
            &mut module_self_bindings,
        );
        let empty_module = arenas.new_module(
            None,
            ModuleKind::Def(DefKind::Mod, root_def_id, None),
            ExpnId::root(),
            DUMMY_SP,
            true,
            &mut Default::default(),
            &mut Default::default(),
        );

        let mut node_id_to_def_id = NodeMap::default();
        let crate_feed = tcx.create_local_crate_def_id(crate_span);

        crate_feed.def_kind(DefKind::Mod);
        let crate_feed = crate_feed.downgrade();
        node_id_to_def_id.insert(CRATE_NODE_ID, crate_feed);

        let mut invocation_parents = FxHashMap::default();
        invocation_parents.insert(LocalExpnId::ROOT, InvocationParent::ROOT);

        let mut extern_prelude: FxIndexMap<Ident, ExternPreludeEntry<'_>> = tcx
            .sess
            .opts
            .externs
            .iter()
            .filter(|(_, entry)| entry.add_prelude)
            .map(|(name, _)| (Ident::from_str(name), Default::default()))
            .collect();

        if !attr::contains_name(attrs, sym::no_core) {
            extern_prelude.insert(Ident::with_dummy_span(sym::core), Default::default());
            if !attr::contains_name(attrs, sym::no_std) {
                extern_prelude.insert(Ident::with_dummy_span(sym::std), Default::default());
            }
        }

        let registered_tools = tcx.registered_tools(());

        let pub_vis = ty::Visibility::<DefId>::Public;
        let edition = tcx.sess.edition();

        let mut resolver = Resolver {
            tcx,

            expn_that_defined: Default::default(),

            // The outermost module has def ID 0; this is not reflected in the
            // AST.
            graph_root,
            prelude: None,
            extern_prelude,

            field_names: Default::default(),
            field_visibility_spans: FxHashMap::default(),

            determined_imports: Vec::new(),
            indeterminate_imports: Vec::new(),

            pat_span_map: Default::default(),
            partial_res_map: Default::default(),
            import_res_map: Default::default(),
            import_use_map: Default::default(),
            label_res_map: Default::default(),
            lifetimes_res_map: Default::default(),
            extra_lifetime_params_map: Default::default(),
            extern_crate_map: Default::default(),
            module_children: Default::default(),
            trait_map: NodeMap::default(),
            underscore_disambiguator: 0,
            empty_module,
            module_map,
            block_map: Default::default(),
            binding_parent_modules: FxHashMap::default(),
            ast_transform_scopes: FxHashMap::default(),

            glob_map: Default::default(),
            glob_error: None,
            visibilities_for_hashing: Default::default(),
            used_imports: FxHashSet::default(),
            maybe_unused_trait_imports: Default::default(),

            privacy_errors: Vec::new(),
            ambiguity_errors: Vec::new(),
            use_injections: Vec::new(),
            macro_expanded_macro_export_errors: BTreeSet::new(),

            arenas,
            dummy_binding: (Res::Err, pub_vis, DUMMY_SP, LocalExpnId::ROOT).to_name_binding(arenas),
            builtin_types_bindings: PrimTy::ALL
                .iter()
                .map(|prim_ty| {
                    let binding = (Res::PrimTy(*prim_ty), pub_vis, DUMMY_SP, LocalExpnId::ROOT)
                        .to_name_binding(arenas);
                    (prim_ty.name(), binding)
                })
                .collect(),
            builtin_attrs_bindings: BUILTIN_ATTRIBUTES
                .iter()
                .map(|builtin_attr| {
                    let res = Res::NonMacroAttr(NonMacroAttrKind::Builtin(builtin_attr.name));
                    let binding =
                        (res, pub_vis, DUMMY_SP, LocalExpnId::ROOT).to_name_binding(arenas);
                    (builtin_attr.name, binding)
                })
                .collect(),
            registered_tool_bindings: registered_tools
                .iter()
                .map(|ident| {
                    let binding = (Res::ToolMod, pub_vis, ident.span, LocalExpnId::ROOT)
                        .to_name_binding(arenas);
                    (*ident, binding)
                })
                .collect(),
            module_self_bindings,

            used_extern_options: Default::default(),
            macro_names: FxHashSet::default(),
            builtin_macros: Default::default(),
            registered_tools,
            macro_use_prelude: Default::default(),
            macro_map: FxHashMap::default(),
            dummy_ext_bang: Arc::new(SyntaxExtension::dummy_bang(edition)),
            dummy_ext_derive: Arc::new(SyntaxExtension::dummy_derive(edition)),
            non_macro_attr: MacroData::new(Arc::new(SyntaxExtension::non_macro_attr(edition))),
            invocation_parent_scopes: Default::default(),
            output_macro_rules_scopes: Default::default(),
            macro_rules_scopes: Default::default(),
            helper_attrs: Default::default(),
            derive_data: Default::default(),
            local_macro_def_scopes: FxHashMap::default(),
            name_already_seen: FxHashMap::default(),
            potentially_unused_imports: Vec::new(),
            potentially_unnecessary_qualifications: Default::default(),
            struct_constructors: Default::default(),
            unused_macros: Default::default(),
            unused_macro_rules: Default::default(),
            proc_macro_stubs: Default::default(),
            single_segment_macro_resolutions: Default::default(),
            multi_segment_macro_resolutions: Default::default(),
            builtin_attrs: Default::default(),
            containers_deriving_copy: Default::default(),
            lint_buffer: LintBuffer::default(),
            next_node_id: CRATE_NODE_ID,
            node_id_to_def_id,
            disambiguator: DisambiguatorState::new(),
            placeholder_field_indices: Default::default(),
            invocation_parents,
            legacy_const_generic_args: Default::default(),
            item_generics_num_lifetimes: Default::default(),
            main_def: Default::default(),
            trait_impls: Default::default(),
            proc_macros: Default::default(),
            confused_type_with_std_module: Default::default(),
            lifetime_elision_allowed: Default::default(),
            stripped_cfg_items: Default::default(),
            effective_visibilities: Default::default(),
            doc_link_resolutions: Default::default(),
            doc_link_traits_in_scope: Default::default(),
            all_macro_rules: Default::default(),
            delegation_fn_sigs: Default::default(),
            glob_delegation_invoc_ids: Default::default(),
            impl_unexpanded_invocations: Default::default(),
            impl_binding_keys: Default::default(),
            current_crate_outer_attr_insert_span,
            mods_with_parse_errors: Default::default(),
        };

        let root_parent_scope = ParentScope::module(graph_root, &resolver);
        resolver.invocation_parent_scopes.insert(LocalExpnId::ROOT, root_parent_scope);
        resolver.feed_visibility(crate_feed, ty::Visibility::Public);

        resolver
    }

    fn new_module(
        &mut self,
        parent: Option<Module<'ra>>,
        kind: ModuleKind,
        expn_id: ExpnId,
        span: Span,
        no_implicit_prelude: bool,
    ) -> Module<'ra> {
        let module_map = &mut self.module_map;
        let module_self_bindings = &mut self.module_self_bindings;
        self.arenas.new_module(
            parent,
            kind,
            expn_id,
            span,
            no_implicit_prelude,
            module_map,
            module_self_bindings,
        )
    }

    fn next_node_id(&mut self) -> NodeId {
        let start = self.next_node_id;
        let next = start.as_u32().checked_add(1).expect("input too large; ran out of NodeIds");
        self.next_node_id = ast::NodeId::from_u32(next);
        start
    }

    fn next_node_ids(&mut self, count: usize) -> std::ops::Range<NodeId> {
        let start = self.next_node_id;
        let end = start.as_usize().checked_add(count).expect("input too large; ran out of NodeIds");
        self.next_node_id = ast::NodeId::from_usize(end);
        start..self.next_node_id
    }

    pub fn lint_buffer(&mut self) -> &mut LintBuffer {
        &mut self.lint_buffer
    }

    pub fn arenas() -> ResolverArenas<'ra> {
        Default::default()
    }

    fn feed_visibility(&mut self, feed: Feed<'tcx, LocalDefId>, vis: ty::Visibility) {
        let feed = feed.upgrade(self.tcx);
        feed.visibility(vis.to_def_id());
        self.visibilities_for_hashing.push((feed.def_id(), vis));
    }

    pub fn into_outputs(self) -> ResolverOutputs {
        let proc_macros = self.proc_macros;
        let expn_that_defined = self.expn_that_defined;
        let extern_crate_map = self.extern_crate_map;
        let maybe_unused_trait_imports = self.maybe_unused_trait_imports;
        let glob_map = self.glob_map;
        let main_def = self.main_def;
        let confused_type_with_std_module = self.confused_type_with_std_module;
        let effective_visibilities = self.effective_visibilities;

        let stripped_cfg_items = Steal::new(
            self.stripped_cfg_items
                .into_iter()
                .filter_map(|item| {
                    let parent_module =
                        self.node_id_to_def_id.get(&item.parent_module)?.key().to_def_id();
                    Some(StrippedCfgItem { parent_module, ident: item.ident, cfg: item.cfg })
                })
                .collect(),
        );

        let global_ctxt = ResolverGlobalCtxt {
            expn_that_defined,
            visibilities_for_hashing: self.visibilities_for_hashing,
            effective_visibilities,
            extern_crate_map,
            module_children: self.module_children,
            glob_map,
            maybe_unused_trait_imports,
            main_def,
            trait_impls: self.trait_impls,
            proc_macros,
            confused_type_with_std_module,
            doc_link_resolutions: self.doc_link_resolutions,
            doc_link_traits_in_scope: self.doc_link_traits_in_scope,
            all_macro_rules: self.all_macro_rules,
            stripped_cfg_items,
        };
        let ast_lowering = ty::ResolverAstLowering {
            legacy_const_generic_args: self.legacy_const_generic_args,
            partial_res_map: self.partial_res_map,
            import_res_map: self.import_res_map,
            label_res_map: self.label_res_map,
            lifetimes_res_map: self.lifetimes_res_map,
            extra_lifetime_params_map: self.extra_lifetime_params_map,
            next_node_id: self.next_node_id,
            node_id_to_def_id: self
                .node_id_to_def_id
                .into_items()
                .map(|(k, f)| (k, f.key()))
                .collect(),
            disambiguator: self.disambiguator,
            trait_map: self.trait_map,
            lifetime_elision_allowed: self.lifetime_elision_allowed,
            lint_buffer: Steal::new(self.lint_buffer),
            delegation_fn_sigs: self.delegation_fn_sigs,
        };
        ResolverOutputs { global_ctxt, ast_lowering }
    }

    fn create_stable_hashing_context(&self) -> StableHashingContext<'_> {
        StableHashingContext::new(self.tcx.sess, self.tcx.untracked())
    }

    fn crate_loader<T>(&mut self, f: impl FnOnce(&mut CrateLoader<'_, '_>) -> T) -> T {
        f(&mut CrateLoader::new(
            self.tcx,
            &mut CStore::from_tcx_mut(self.tcx),
            &mut self.used_extern_options,
        ))
    }

    fn cstore(&self) -> FreezeReadGuard<'_, CStore> {
        CStore::from_tcx(self.tcx)
    }

    fn dummy_ext(&self, macro_kind: MacroKind) -> Arc<SyntaxExtension> {
        match macro_kind {
            MacroKind::Bang => Arc::clone(&self.dummy_ext_bang),
            MacroKind::Derive => Arc::clone(&self.dummy_ext_derive),
            MacroKind::Attr => Arc::clone(&self.non_macro_attr.ext),
        }
    }

    /// Runs the function on each namespace.
    fn per_ns<F: FnMut(&mut Self, Namespace)>(&mut self, mut f: F) {
        f(self, TypeNS);
        f(self, ValueNS);
        f(self, MacroNS);
    }

    fn is_builtin_macro(&mut self, res: Res) -> bool {
        self.get_macro(res).is_some_and(|macro_data| macro_data.ext.builtin_name.is_some())
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
        self.tcx.sess.time("resolve_crate", || {
            self.tcx.sess.time("finalize_imports", || self.finalize_imports());
            let exported_ambiguities = self.tcx.sess.time("compute_effective_visibilities", || {
                EffectiveVisibilitiesVisitor::compute_effective_visibilities(self, krate)
            });
            self.tcx.sess.time("check_hidden_glob_reexports", || {
                self.check_hidden_glob_reexports(exported_ambiguities)
            });
            self.tcx
                .sess
                .time("finalize_macro_resolutions", || self.finalize_macro_resolutions(krate));
            self.tcx.sess.time("late_resolve_crate", || self.late_resolve_crate(krate));
            self.tcx.sess.time("resolve_main", || self.resolve_main());
            self.tcx.sess.time("resolve_check_unused", || self.check_unused(krate));
            self.tcx.sess.time("resolve_report_errors", || self.report_errors(krate));
            self.tcx
                .sess
                .time("resolve_postprocess", || self.crate_loader(|c| c.postprocess(krate)));
        });

        // Make sure we don't mutate the cstore from here on.
        self.tcx.untracked().cstore.freeze();
    }

    fn traits_in_scope(
        &mut self,
        current_trait: Option<Module<'ra>>,
        parent_scope: &ParentScope<'ra>,
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

        self.visit_scopes(ScopeSet::All(TypeNS), parent_scope, ctxt, |this, scope, _, _| {
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
        module: Module<'ra>,
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
        trait_module: Option<Module<'ra>>,
        assoc_item: Option<(Symbol, Namespace)>,
    ) -> bool {
        match (trait_module, assoc_item) {
            (Some(trait_module), Some((name, ns))) => self
                .resolutions(trait_module)
                .borrow()
                .iter()
                .any(|(key, _name_resolution)| key.ns == ns && key.ident.name == name),
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
            if let Some(node_id) = import.id() {
                let def_id = self.local_def_id(node_id);
                self.maybe_unused_trait_imports.insert(def_id);
                import_ids.push(def_id);
            }
            self.add_to_glob_map(*import, trait_name);
            kind = &binding.kind;
        }
        import_ids
    }

    fn new_disambiguated_key(&mut self, ident: Ident, ns: Namespace) -> BindingKey {
        let ident = ident.normalize_to_macros_2_0();
        let disambiguator = if ident.name == kw::Underscore {
            self.underscore_disambiguator += 1;
            self.underscore_disambiguator
        } else {
            0
        };
        BindingKey { ident, ns, disambiguator }
    }

    fn resolutions(&mut self, module: Module<'ra>) -> &'ra Resolutions<'ra> {
        if module.populate_on_access.get() {
            module.populate_on_access.set(false);
            self.build_reduced_graph_external(module);
        }
        &module.0.0.lazy_resolutions
    }

    fn resolution(
        &mut self,
        module: Module<'ra>,
        key: BindingKey,
    ) -> &'ra RefCell<NameResolution<'ra>> {
        *self
            .resolutions(module)
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| self.arenas.alloc_name_resolution())
    }

    /// Test if AmbiguityError ambi is any identical to any one inside ambiguity_errors
    fn matches_previous_ambiguity_error(&self, ambi: &AmbiguityError<'_>) -> bool {
        for ambiguity_error in &self.ambiguity_errors {
            // if the span location and ident as well as its span are the same
            if ambiguity_error.kind == ambi.kind
                && ambiguity_error.ident == ambi.ident
                && ambiguity_error.ident.span == ambi.ident.span
                && ambiguity_error.b1.span == ambi.b1.span
                && ambiguity_error.b2.span == ambi.b2.span
                && ambiguity_error.misc1 == ambi.misc1
                && ambiguity_error.misc2 == ambi.misc2
            {
                return true;
            }
        }
        false
    }

    fn record_use(&mut self, ident: Ident, used_binding: NameBinding<'ra>, used: Used) {
        self.record_use_inner(ident, used_binding, used, used_binding.warn_ambiguity);
    }

    fn record_use_inner(
        &mut self,
        ident: Ident,
        used_binding: NameBinding<'ra>,
        used: Used,
        warn_ambiguity: bool,
    ) {
        if let Some((b2, kind)) = used_binding.ambiguity {
            let ambiguity_error = AmbiguityError {
                kind,
                ident,
                b1: used_binding,
                b2,
                misc1: AmbiguityErrorMisc::None,
                misc2: AmbiguityErrorMisc::None,
                warning: warn_ambiguity,
            };
            if !self.matches_previous_ambiguity_error(&ambiguity_error) {
                // avoid duplicated span information to be emit out
                self.ambiguity_errors.push(ambiguity_error);
            }
        }
        if let NameBindingKind::Import { import, binding } = used_binding.kind {
            if let ImportKind::MacroUse { warn_private: true } = import.kind {
                // Do not report the lint if the macro name resolves in stdlib prelude
                // even without the problematic `macro_use` import.
                let found_in_stdlib_prelude = self.prelude.is_some_and(|prelude| {
                    self.maybe_resolve_ident_in_module(
                        ModuleOrUniformRoot::Module(prelude),
                        ident,
                        MacroNS,
                        &ParentScope::module(self.empty_module, self),
                        None,
                    )
                    .is_ok()
                });
                if !found_in_stdlib_prelude {
                    self.lint_buffer().buffer_lint(
                        PRIVATE_MACRO_USE,
                        import.root_id,
                        ident.span,
                        BuiltinLintDiag::MacroIsPrivate(ident),
                    );
                }
            }
            // Avoid marking `extern crate` items that refer to a name from extern prelude,
            // but not introduce it, as used if they are accessed from lexical scope.
            if used == Used::Scope {
                if let Some(entry) = self.extern_prelude.get(&ident.normalize_to_macros_2_0()) {
                    if !entry.introduced_by_item && entry.binding == Some(used_binding) {
                        return;
                    }
                }
            }
            let old_used = self.import_use_map.entry(import).or_insert(used);
            if *old_used < used {
                *old_used = used;
            }
            if let Some(id) = import.id() {
                self.used_imports.insert(id);
            }
            self.add_to_glob_map(import, ident);
            self.record_use_inner(
                ident,
                binding,
                Used::Other,
                warn_ambiguity || binding.warn_ambiguity,
            );
        }
    }

    #[inline]
    fn add_to_glob_map(&mut self, import: Import<'_>, ident: Ident) {
        if let ImportKind::Glob { id, .. } = import.kind {
            let def_id = self.local_def_id(id);
            self.glob_map.entry(def_id).or_default().insert(ident.name);
        }
    }

    fn resolve_crate_root(&mut self, ident: Ident) -> Module<'ra> {
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
            // Then find the last semi-opaque mark from the end if it exists.
            for (mark, transparency) in iter {
                if transparency == Transparency::SemiOpaque {
                    result = Some(mark);
                } else {
                    break;
                }
            }
            debug!(
                "resolve_crate_root: found semi-opaque mark {:?} {:?}",
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

    fn resolve_self(&mut self, ctxt: &mut SyntaxContext, module: Module<'ra>) -> Module<'ra> {
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
            panic!("path resolved multiple times ({prev_res:?} before, {resolution:?} now)");
        }
    }

    fn record_pat_span(&mut self, node: NodeId, span: Span) {
        debug!("(recording pat) recording {:?} for {:?}", node, span);
        self.pat_span_map.insert(node, span);
    }

    fn is_accessible_from(
        &self,
        vis: ty::Visibility<impl Into<DefId>>,
        module: Module<'ra>,
    ) -> bool {
        vis.is_accessible_from(module.nearest_parent_mod(), self.tcx)
    }

    fn set_binding_parent_module(&mut self, binding: NameBinding<'ra>, module: Module<'ra>) {
        if let Some(old_module) = self.binding_parent_modules.insert(binding, module) {
            if module != old_module {
                span_bug!(binding.span, "parent module is reset for binding");
            }
        }
    }

    fn disambiguate_macro_rules_vs_modularized(
        &self,
        macro_rules: NameBinding<'ra>,
        modularized: NameBinding<'ra>,
    ) -> bool {
        // Some non-controversial subset of ambiguities "modularized macro name" vs "macro_rules"
        // is disambiguated to mitigate regressions from macro modularization.
        // Scoping for `macro_rules` behaves like scoping for `let` at module level, in general.
        match (
            self.binding_parent_modules.get(&macro_rules),
            self.binding_parent_modules.get(&modularized),
        ) {
            (Some(macro_rules), Some(modularized)) => {
                macro_rules.nearest_parent_mod() == modularized.nearest_parent_mod()
                    && modularized.is_ancestor_of(*macro_rules)
            }
            _ => false,
        }
    }

    fn extern_prelude_get(&mut self, ident: Ident, finalize: bool) -> Option<NameBinding<'ra>> {
        if ident.is_path_segment_keyword() {
            // Make sure `self`, `super` etc produce an error when passed to here.
            return None;
        }

        let norm_ident = ident.normalize_to_macros_2_0();
        let binding = self.extern_prelude.get(&norm_ident).cloned().and_then(|entry| {
            Some(if let Some(binding) = entry.binding {
                if finalize {
                    if !entry.is_import() {
                        self.crate_loader(|c| c.process_path_extern(ident.name, ident.span));
                    } else if entry.introduced_by_item {
                        self.record_use(ident, binding, Used::Other);
                    }
                }
                binding
            } else {
                let crate_id = if finalize {
                    let Some(crate_id) =
                        self.crate_loader(|c| c.process_path_extern(ident.name, ident.span))
                    else {
                        return Some(self.dummy_binding);
                    };
                    crate_id
                } else {
                    self.crate_loader(|c| c.maybe_process_path_extern(ident.name))?
                };
                let crate_root = self.expect_module(crate_id.as_def_id());
                let vis = ty::Visibility::<DefId>::Public;
                (crate_root, vis, DUMMY_SP, LocalExpnId::ROOT).to_name_binding(self.arenas)
            })
        });

        if let Some(entry) = self.extern_prelude.get_mut(&norm_ident) {
            entry.binding = binding;
        }

        binding
    }

    /// Rustdoc uses this to resolve doc link paths in a recoverable way. `PathResult<'a>`
    /// isn't something that can be returned because it can't be made to live that long,
    /// and also it's a private type. Fortunately rustdoc doesn't need to know the error,
    /// just that an error occurred.
    fn resolve_rustdoc_path(
        &mut self,
        path_str: &str,
        ns: Namespace,
        parent_scope: ParentScope<'ra>,
    ) -> Option<Res> {
        let segments: Result<Vec<_>, ()> = path_str
            .split("::")
            .enumerate()
            .map(|(i, s)| {
                let sym = if s.is_empty() {
                    if i == 0 {
                        // For a path like `::a::b`, use `kw::PathRoot` as the leading segment.
                        kw::PathRoot
                    } else {
                        return Err(()); // occurs in cases like `String::`
                    }
                } else {
                    Symbol::intern(s)
                };
                Ok(Segment::from_ident(Ident::with_dummy_span(sym)))
            })
            .collect();
        let Ok(segments) = segments else { return None };

        match self.maybe_resolve_path(&segments, Some(ns), &parent_scope, None) {
            PathResult::Module(ModuleOrUniformRoot::Module(module)) => Some(module.res().unwrap()),
            PathResult::NonModule(path_res) => {
                path_res.full_res().filter(|res| !matches!(res, Res::Def(DefKind::Ctor(..), _)))
            }
            PathResult::Module(ModuleOrUniformRoot::ExternPrelude) | PathResult::Failed { .. } => {
                None
            }
            PathResult::Module(..) | PathResult::Indeterminate => unreachable!(),
        }
    }

    /// Retrieves definition span of the given `DefId`.
    fn def_span(&self, def_id: DefId) -> Span {
        match def_id.as_local() {
            Some(def_id) => self.tcx.source_span(def_id),
            // Query `def_span` is not used because hashing its result span is expensive.
            None => self.cstore().def_span_untracked(def_id, self.tcx.sess),
        }
    }

    fn field_idents(&self, def_id: DefId) -> Option<Vec<Ident>> {
        match def_id.as_local() {
            Some(def_id) => self.field_names.get(&def_id).cloned(),
            None => Some(
                self.tcx
                    .associated_item_def_ids(def_id)
                    .iter()
                    .map(|&def_id| {
                        Ident::new(self.tcx.item_name(def_id), self.tcx.def_span(def_id))
                    })
                    .collect(),
            ),
        }
    }

    /// Checks if an expression refers to a function marked with
    /// `#[rustc_legacy_const_generics]` and returns the argument index list
    /// from the attribute.
    fn legacy_const_generic_args(&mut self, expr: &Expr) -> Option<Vec<usize>> {
        if let ExprKind::Path(None, path) = &expr.kind {
            // Don't perform legacy const generics rewriting if the path already
            // has generic arguments.
            if path.segments.last().unwrap().args.is_some() {
                return None;
            }

            let res = self.partial_res_map.get(&expr.id)?.full_res()?;
            if let Res::Def(def::DefKind::Fn, def_id) = res {
                // We only support cross-crate argument rewriting. Uses
                // within the same crate should be updated to use the new
                // const generics style.
                if def_id.is_local() {
                    return None;
                }

                if let Some(v) = self.legacy_const_generic_args.get(&def_id) {
                    return v.clone();
                }

                let attr = self.tcx.get_attr(def_id, sym::rustc_legacy_const_generics)?;
                let mut ret = Vec::new();
                for meta in attr.meta_item_list()? {
                    match meta.lit()?.kind {
                        LitKind::Int(a, _) => ret.push(a.get() as usize),
                        _ => panic!("invalid arg index"),
                    }
                }
                // Cache the lookup to avoid parsing attributes for an item multiple times.
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
            None,
        ) else {
            return;
        };

        let res = name_binding.res();
        let is_import = name_binding.is_import();
        let span = name_binding.span;
        if let Res::Def(DefKind::Fn, _) = res {
            self.record_use(ident, name_binding, Used::Other);
        }
        self.main_def = Some(MainDefinition { res, is_import, span });
    }
}

fn names_to_string(names: impl Iterator<Item = Symbol>) -> String {
    let mut result = String::new();
    for (i, name) in names.filter(|name| *name != kw::PathRoot).enumerate() {
        if i > 0 {
            result.push_str("::");
        }
        if Ident::with_dummy_span(name).is_raw_guess() {
            result.push_str("r#");
        }
        result.push_str(name.as_str());
    }
    result
}

fn path_names_to_string(path: &Path) -> String {
    names_to_string(path.segments.iter().map(|seg| seg.ident.name))
}

/// A somewhat inefficient routine to obtain the name of a module.
fn module_to_string(mut module: Module<'_>) -> Option<String> {
    let mut names = Vec::new();
    loop {
        if let ModuleKind::Def(.., name) = module.kind {
            if let Some(parent) = module.parent {
                // `unwrap` is safe: the presence of a parent means it's not the crate root.
                names.push(name.unwrap());
                module = parent
            } else {
                break;
            }
        } else {
            names.push(sym::opaque_module_name_placeholder);
            let Some(parent) = module.parent else {
                return None;
            };
            module = parent;
        }
    }
    if names.is_empty() {
        return None;
    }
    Some(names_to_string(names.iter().rev().copied()))
}

#[derive(Copy, Clone, Debug)]
struct Finalize {
    /// Node ID for linting.
    node_id: NodeId,
    /// Span of the whole path or some its characteristic fragment.
    /// E.g. span of `b` in `foo::{a, b, c}`, or full span for regular paths.
    path_span: Span,
    /// Span of the path start, suitable for prepending something to it.
    /// E.g. span of `foo` in `foo::{a, b, c}`, or full span for regular paths.
    root_span: Span,
    /// Whether to report privacy errors or silently return "no resolution" for them,
    /// similarly to speculative resolution.
    report_private: bool,
    /// Tracks whether an item is used in scope or used relatively to a module.
    used: Used,
}

impl Finalize {
    fn new(node_id: NodeId, path_span: Span) -> Finalize {
        Finalize::with_root_span(node_id, path_span, path_span)
    }

    fn with_root_span(node_id: NodeId, path_span: Span, root_span: Span) -> Finalize {
        Finalize { node_id, path_span, root_span, report_private: true, used: Used::Other }
    }
}

pub fn provide(providers: &mut Providers) {
    providers.registered_tools = macros::registered_tools;
}
