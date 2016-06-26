// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_resolve"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(associated_consts)]
#![feature(borrow_state)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;
extern crate arena;
#[macro_use]
extern crate rustc;

use self::Namespace::*;
use self::ResolveResult::*;
use self::FallbackSuggestion::*;
use self::TypeParameters::*;
use self::RibKind::*;
use self::UseLexicalScopeFlag::*;
use self::ModulePrefixResult::*;
use self::ParentLink::*;

use rustc::hir::map::Definitions;
use rustc::hir::{self, PrimTy, TyBool, TyChar, TyFloat, TyInt, TyUint, TyStr};
use rustc::session::Session;
use rustc::lint;
use rustc::hir::def::*;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::ty::subst::{ParamSpace, FnSpace, TypeSpace};
use rustc::hir::{Freevar, FreevarMap, TraitCandidate, TraitMap, GlobMap};
use rustc::util::nodemap::{NodeMap, NodeSet, FnvHashMap, FnvHashSet};

use syntax::ext::mtwt;
use syntax::ast::{self, FloatTy};
use syntax::ast::{CRATE_NODE_ID, Name, NodeId, CrateNum, IntTy, UintTy};
use syntax::parse::token::{self, keywords};
use syntax::util::lev_distance::find_best_match_for_name;

use syntax::visit::{self, FnKind, Visitor};
use syntax::ast::{Arm, BindingMode, Block, Crate, Expr, ExprKind};
use syntax::ast::{FnDecl, ForeignItem, ForeignItemKind, Generics};
use syntax::ast::{Item, ItemKind, ImplItem, ImplItemKind};
use syntax::ast::{Local, Mutability, Pat, PatKind, Path};
use syntax::ast::{PathSegment, PathParameters, QSelf, TraitItemKind, TraitRef, Ty, TyKind};

use syntax_pos::Span;
use errors::DiagnosticBuilder;

use std::collections::{HashMap, HashSet};
use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem::replace;

use resolve_imports::{ImportDirective, NameResolution};

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
mod diagnostics;

mod check_unused;
mod build_reduced_graph;
mod resolve_imports;

enum SuggestionType {
    Macro(String),
    Function(token::InternedString),
    NotFound,
}

/// Candidates for a name resolution failure
struct SuggestedCandidates {
    name: String,
    candidates: Vec<Path>,
}

enum ResolutionError<'a> {
    /// error E0401: can't use type parameters from outer function
    TypeParametersFromOuterFunction,
    /// error E0402: cannot use an outer type parameter in this context
    OuterTypeParameterContext,
    /// error E0403: the name is already used for a type parameter in this type parameter list
    NameAlreadyUsedInTypeParameterList(Name),
    /// error E0404: is not a trait
    IsNotATrait(&'a str),
    /// error E0405: use of undeclared trait name
    UndeclaredTraitName(&'a str, SuggestedCandidates),
    /// error E0407: method is not a member of trait
    MethodNotMemberOfTrait(Name, &'a str),
    /// error E0437: type is not a member of trait
    TypeNotMemberOfTrait(Name, &'a str),
    /// error E0438: const is not a member of trait
    ConstNotMemberOfTrait(Name, &'a str),
    /// error E0408: variable `{}` from pattern #{} is not bound in pattern #{}
    VariableNotBoundInPattern(Name, usize, usize),
    /// error E0409: variable is bound with different mode in pattern #{} than in pattern #1
    VariableBoundWithDifferentMode(Name, usize),
    /// error E0411: use of `Self` outside of an impl or trait
    SelfUsedOutsideImplOrTrait,
    /// error E0412: use of undeclared
    UseOfUndeclared(&'a str, &'a str, SuggestedCandidates),
    /// error E0415: identifier is bound more than once in this parameter list
    IdentifierBoundMoreThanOnceInParameterList(&'a str),
    /// error E0416: identifier is bound more than once in the same pattern
    IdentifierBoundMoreThanOnceInSamePattern(&'a str),
    /// error E0422: does not name a struct
    DoesNotNameAStruct(&'a str),
    /// error E0423: is a struct variant name, but this expression uses it like a function name
    StructVariantUsedAsFunction(&'a str),
    /// error E0424: `self` is not available in a static method
    SelfNotAvailableInStaticMethod,
    /// error E0425: unresolved name
    UnresolvedName {
        path: &'a str,
        message: &'a str,
        context: UnresolvedNameContext<'a>,
        is_static_method: bool,
        is_field: bool,
        def: Def,
    },
    /// error E0426: use of undeclared label
    UndeclaredLabel(&'a str),
    /// error E0429: `self` imports are only allowed within a { } list
    SelfImportsOnlyAllowedWithin,
    /// error E0430: `self` import can only appear once in the list
    SelfImportCanOnlyAppearOnceInTheList,
    /// error E0431: `self` import can only appear in an import list with a non-empty prefix
    SelfImportOnlyInImportListWithNonEmptyPrefix,
    /// error E0432: unresolved import
    UnresolvedImport(Option<(&'a str, &'a str)>),
    /// error E0433: failed to resolve
    FailedToResolve(&'a str),
    /// error E0434: can't capture dynamic environment in a fn item
    CannotCaptureDynamicEnvironmentInFnItem,
    /// error E0435: attempt to use a non-constant value in a constant
    AttemptToUseNonConstantValueInConstant,
    /// error E0530: X bindings cannot shadow Ys
    BindingShadowsSomethingUnacceptable(&'a str, &'a str, Name),
    /// error E0531: unresolved pattern path kind `name`
    PatPathUnresolved(&'a str, &'a Path),
    /// error E0532: expected pattern path kind, found another pattern path kind
    PatPathUnexpected(&'a str, &'a str, &'a Path),
}

/// Context of where `ResolutionError::UnresolvedName` arose.
#[derive(Clone, PartialEq, Eq, Debug)]
enum UnresolvedNameContext<'a> {
    /// `PathIsMod(parent)` indicates that a given path, used in
    /// expression context, actually resolved to a module rather than
    /// a value. The optional expression attached to the variant is the
    /// the parent of the erroneous path expression.
    PathIsMod(Option<&'a Expr>),

    /// `Other` means we have no extra information about the context
    /// of the unresolved name error. (Maybe we could eliminate all
    /// such cases; but for now, this is an information-free default.)
    Other,
}

fn resolve_error<'b, 'a: 'b, 'c>(resolver: &'b Resolver<'a>,
                                 span: syntax_pos::Span,
                                 resolution_error: ResolutionError<'c>) {
    resolve_struct_error(resolver, span, resolution_error).emit();
}

fn resolve_struct_error<'b, 'a: 'b, 'c>(resolver: &'b Resolver<'a>,
                                        span: syntax_pos::Span,
                                        resolution_error: ResolutionError<'c>)
                                        -> DiagnosticBuilder<'a> {
    if !resolver.emit_errors {
        return resolver.session.diagnostic().struct_dummy();
    }

    match resolution_error {
        ResolutionError::TypeParametersFromOuterFunction => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0401,
                                           "can't use type parameters from outer function; \
                                           try using a local type parameter instead");
            err.span_label(span, &format!("use of type variable from outer function"));
            err
        }
        ResolutionError::OuterTypeParameterContext => {
            struct_span_err!(resolver.session,
                             span,
                             E0402,
                             "cannot use an outer type parameter in this context")
        }
        ResolutionError::NameAlreadyUsedInTypeParameterList(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0403,
                             "the name `{}` is already used for a type parameter in this type \
                              parameter list",
                             name)
        }
        ResolutionError::IsNotATrait(name) => {
            struct_span_err!(resolver.session, span, E0404, "`{}` is not a trait", name)
        }
        ResolutionError::UndeclaredTraitName(name, candidates) => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0405,
                                           "trait `{}` is not in scope",
                                           name);
            show_candidates(&mut err, &candidates);
            err.span_label(span, &format!("`{}` is not in scope", name));
            err
        }
        ResolutionError::MethodNotMemberOfTrait(method, trait_) => {
            struct_span_err!(resolver.session,
                             span,
                             E0407,
                             "method `{}` is not a member of trait `{}`",
                             method,
                             trait_)
        }
        ResolutionError::TypeNotMemberOfTrait(type_, trait_) => {
            struct_span_err!(resolver.session,
                             span,
                             E0437,
                             "type `{}` is not a member of trait `{}`",
                             type_,
                             trait_)
        }
        ResolutionError::ConstNotMemberOfTrait(const_, trait_) => {
            struct_span_err!(resolver.session,
                             span,
                             E0438,
                             "const `{}` is not a member of trait `{}`",
                             const_,
                             trait_)
        }
        ResolutionError::VariableNotBoundInPattern(variable_name, from, to) => {
            struct_span_err!(resolver.session,
                             span,
                             E0408,
                             "variable `{}` from pattern #{} is not bound in pattern #{}",
                             variable_name,
                             from,
                             to)
        }
        ResolutionError::VariableBoundWithDifferentMode(variable_name, pattern_number) => {
            struct_span_err!(resolver.session,
                             span,
                             E0409,
                             "variable `{}` is bound with different mode in pattern #{} than in \
                              pattern #1",
                             variable_name,
                             pattern_number)
        }
        ResolutionError::SelfUsedOutsideImplOrTrait => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0411,
                                           "use of `Self` outside of an impl or trait");
            err.span_label(span, &format!("used outside of impl or trait"));
            err
        }
        ResolutionError::UseOfUndeclared(kind, name, candidates) => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0412,
                                           "{} `{}` is undefined or not in scope",
                                           kind,
                                           name);
            show_candidates(&mut err, &candidates);
            err.span_label(span, &format!("undefined or not in scope"));
            err
        }
        ResolutionError::IdentifierBoundMoreThanOnceInParameterList(identifier) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0415,
                             "identifier `{}` is bound more than once in this parameter list",
                             identifier);
            err.span_label(span, &format!("used as parameter more than once"));
            err
        }
        ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(identifier) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0416,
                             "identifier `{}` is bound more than once in the same pattern",
                             identifier);
            err.span_label(span, &format!("used in a pattern more than once"));
            err
        }
        ResolutionError::DoesNotNameAStruct(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0422,
                             "`{}` does not name a structure",
                             name)
        }
        ResolutionError::StructVariantUsedAsFunction(path_name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0423,
                             "`{}` is the name of a struct or struct variant, but this expression \
                             uses it like a function name",
                             path_name)
        }
        ResolutionError::SelfNotAvailableInStaticMethod => {
            struct_span_err!(resolver.session,
                             span,
                             E0424,
                             "`self` is not available in a static method. Maybe a `self` \
                             argument is missing?")
        }
        ResolutionError::UnresolvedName { path, message: msg, context, is_static_method,
                                          is_field, def } => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0425,
                                           "unresolved name `{}`{}",
                                           path,
                                           msg);
            match context {
                UnresolvedNameContext::Other => {
                    if msg.is_empty() && is_static_method && is_field {
                        err.help("this is an associated function, you don't have access to \
                                  this type's fields or methods");
                    }
                }
                UnresolvedNameContext::PathIsMod(parent) => {
                    err.help(&match parent.map(|parent| &parent.node) {
                        Some(&ExprKind::Field(_, ident)) => {
                            format!("to reference an item from the `{module}` module, \
                                     use `{module}::{ident}`",
                                    module = path,
                                    ident = ident.node)
                        }
                        Some(&ExprKind::MethodCall(ident, _, _)) => {
                            format!("to call a function from the `{module}` module, \
                                     use `{module}::{ident}(..)`",
                                    module = path,
                                    ident = ident.node)
                        }
                        _ => {
                            format!("{def} `{module}` cannot be used as an expression",
                                    def = def.kind_name(),
                                    module = path)
                        }
                    });
                }
            }
            err
        }
        ResolutionError::UndeclaredLabel(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0426,
                             "use of undeclared label `{}`",
                             name)
        }
        ResolutionError::SelfImportsOnlyAllowedWithin => {
            struct_span_err!(resolver.session,
                             span,
                             E0429,
                             "{}",
                             "`self` imports are only allowed within a { } list")
        }
        ResolutionError::SelfImportCanOnlyAppearOnceInTheList => {
            struct_span_err!(resolver.session,
                             span,
                             E0430,
                             "`self` import can only appear once in the list")
        }
        ResolutionError::SelfImportOnlyInImportListWithNonEmptyPrefix => {
            struct_span_err!(resolver.session,
                             span,
                             E0431,
                             "`self` import can only appear in an import list with a \
                              non-empty prefix")
        }
        ResolutionError::UnresolvedImport(name) => {
            let msg = match name {
                Some((n, p)) => format!("unresolved import `{}`{}", n, p),
                None => "unresolved import".to_owned(),
            };
            struct_span_err!(resolver.session, span, E0432, "{}", msg)
        }
        ResolutionError::FailedToResolve(msg) => {
            struct_span_err!(resolver.session, span, E0433, "failed to resolve. {}", msg)
        }
        ResolutionError::CannotCaptureDynamicEnvironmentInFnItem => {
            struct_span_err!(resolver.session,
                             span,
                             E0434,
                             "{}",
                             "can't capture dynamic environment in a fn item; use the || { ... } \
                              closure form instead")
        }
        ResolutionError::AttemptToUseNonConstantValueInConstant => {
            struct_span_err!(resolver.session,
                             span,
                             E0435,
                             "attempt to use a non-constant value in a constant")
        }
        ResolutionError::BindingShadowsSomethingUnacceptable(what_binding, shadows_what, name) => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0530,
                                           "{}s cannot shadow {}s", what_binding, shadows_what);
            err.span_label(span, &format!("cannot be named the same as a {}", shadows_what));
            if let Success(binding) = resolver.current_module.resolve_name(name, ValueNS, true) {
                let participle = if binding.is_import() { "imported" } else { "defined" };
                err.span_label(binding.span, &format!("a {} `{}` is {} here",
                                                      shadows_what, name, participle));
            }
            err
        }
        ResolutionError::PatPathUnresolved(expected_what, path) => {
            struct_span_err!(resolver.session,
                             span,
                             E0531,
                             "unresolved {} `{}`",
                             expected_what,
                             path.segments.last().unwrap().identifier)
        }
        ResolutionError::PatPathUnexpected(expected_what, found_what, path) => {
            struct_span_err!(resolver.session,
                             span,
                             E0532,
                             "expected {}, found {} `{}`",
                             expected_what,
                             found_what,
                             path.segments.last().unwrap().identifier)
        }
    }
}

#[derive(Copy, Clone)]
struct BindingInfo {
    span: Span,
    binding_mode: BindingMode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = HashMap<Name, BindingInfo>;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum PatternSource {
    Match,
    IfLet,
    WhileLet,
    Let,
    For,
    FnParam,
}

impl PatternSource {
    fn is_refutable(self) -> bool {
        match self {
            PatternSource::Match | PatternSource::IfLet | PatternSource::WhileLet => true,
            PatternSource::Let | PatternSource::For | PatternSource::FnParam  => false,
        }
    }
    fn descr(self) -> &'static str {
        match self {
            PatternSource::Match => "match binding",
            PatternSource::IfLet => "if let binding",
            PatternSource::WhileLet => "while let binding",
            PatternSource::Let => "let binding",
            PatternSource::For => "for binding",
            PatternSource::FnParam => "function parameter",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Namespace {
    TypeNS,
    ValueNS,
}

impl<'a> Visitor for Resolver<'a> {
    fn visit_item(&mut self, item: &Item) {
        self.resolve_item(item);
    }
    fn visit_arm(&mut self, arm: &Arm) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &Block) {
        self.resolve_block(block);
    }
    fn visit_expr(&mut self, expr: &Expr) {
        self.resolve_expr(expr, None);
    }
    fn visit_local(&mut self, local: &Local) {
        self.resolve_local(local);
    }
    fn visit_ty(&mut self, ty: &Ty) {
        self.resolve_type(ty);
    }
    fn visit_poly_trait_ref(&mut self, tref: &ast::PolyTraitRef, m: &ast::TraitBoundModifier) {
        match self.resolve_trait_reference(tref.trait_ref.ref_id, &tref.trait_ref.path, 0) {
            Ok(def) => self.record_def(tref.trait_ref.ref_id, def),
            Err(_) => {
                // error already reported
                self.record_def(tref.trait_ref.ref_id, err_path_resolution())
            }
        }
        visit::walk_poly_trait_ref(self, tref, m);
    }
    fn visit_variant(&mut self,
                     variant: &ast::Variant,
                     generics: &Generics,
                     item_id: ast::NodeId) {
        if let Some(ref dis_expr) = variant.node.disr_expr {
            // resolve the discriminator expr as a constant
            self.with_constant_rib(|this| {
                this.visit_expr(dis_expr);
            });
        }

        // `visit::walk_variant` without the discriminant expression.
        self.visit_variant_data(&variant.node.data,
                                variant.node.name,
                                generics,
                                item_id,
                                variant.span);
    }
    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        let type_parameters = match foreign_item.node {
            ForeignItemKind::Fn(_, ref generics) => {
                HasTypeParameters(generics, FnSpace, ItemRibKind)
            }
            ForeignItemKind::Static(..) => NoTypeParameters,
        };
        self.with_type_parameter_rib(type_parameters, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }
    fn visit_fn(&mut self,
                function_kind: FnKind,
                declaration: &FnDecl,
                block: &Block,
                _: Span,
                node_id: NodeId) {
        let rib_kind = match function_kind {
            FnKind::ItemFn(_, generics, _, _, _, _) => {
                self.visit_generics(generics);
                ItemRibKind
            }
            FnKind::Method(_, sig, _) => {
                self.visit_generics(&sig.generics);
                MethodRibKind(!sig.decl.has_self())
            }
            FnKind::Closure => ClosureRibKind(node_id),
        };
        self.resolve_function(rib_kind, declaration, block);
    }
}

pub type ErrorMessage = Option<(Span, String)>;

#[derive(Clone, PartialEq, Eq)]
pub enum ResolveResult<T> {
    Failed(ErrorMessage), // Failed to resolve the name, optional helpful error message.
    Indeterminate, // Couldn't determine due to unresolved globs.
    Success(T), // Successfully resolved the import.
}

impl<T> ResolveResult<T> {
    fn and_then<U, F: FnOnce(T) -> ResolveResult<U>>(self, f: F) -> ResolveResult<U> {
        match self {
            Failed(msg) => Failed(msg),
            Indeterminate => Indeterminate,
            Success(t) => f(t),
        }
    }

    fn success(self) -> Option<T> {
        match self {
            Success(t) => Some(t),
            _ => None,
        }
    }
}

enum FallbackSuggestion {
    NoSuggestion,
    Field,
    TraitItem,
    TraitMethod(String),
}

#[derive(Copy, Clone)]
enum TypeParameters<'a, 'b> {
    NoTypeParameters,
    HasTypeParameters(// Type parameters.
                      &'b Generics,

                      // Identifies the things that these parameters
                      // were declared on (type, fn, etc)
                      ParamSpace,

                      // The kind of the rib used for type parameters.
                      RibKind<'a>),
}

// The rib kind controls the translation of local
// definitions (`Def::Local`) to upvars (`Def::Upvar`).
#[derive(Copy, Clone, Debug)]
enum RibKind<'a> {
    // No translation needs to be applied.
    NormalRibKind,

    // We passed through a closure scope at the given node ID.
    // Translate upvars as appropriate.
    ClosureRibKind(NodeId /* func id */),

    // We passed through an impl or trait and are now in one of its
    // methods. Allow references to ty params that impl or trait
    // binds. Disallow any other upvars (including other ty params that are
    // upvars).
    //
    // The boolean value represents the fact that this method is static or not.
    MethodRibKind(bool),

    // We passed through an item scope. Disallow upvars.
    ItemRibKind,

    // We're in a constant item. Can't refer to dynamic stuff.
    ConstantItemRibKind,

    // We passed through a module.
    ModuleRibKind(Module<'a>),
}

#[derive(Copy, Clone)]
enum UseLexicalScopeFlag {
    DontUseLexicalScope,
    UseLexicalScope,
}

enum ModulePrefixResult<'a> {
    NoPrefixFound,
    PrefixFound(Module<'a>, usize),
}

/// One local scope.
#[derive(Debug)]
struct Rib<'a> {
    bindings: HashMap<Name, Def>,
    kind: RibKind<'a>,
}

impl<'a> Rib<'a> {
    fn new(kind: RibKind<'a>) -> Rib<'a> {
        Rib {
            bindings: HashMap::new(),
            kind: kind,
        }
    }
}

/// A definition along with the index of the rib it was found on
struct LocalDef {
    ribs: Option<(Namespace, usize)>,
    def: Def,
}

impl LocalDef {
    fn from_def(def: Def) -> Self {
        LocalDef {
            ribs: None,
            def: def,
        }
    }
}

enum LexicalScopeBinding<'a> {
    Item(&'a NameBinding<'a>),
    LocalDef(LocalDef),
}

impl<'a> LexicalScopeBinding<'a> {
    fn local_def(self) -> LocalDef {
        match self {
            LexicalScopeBinding::LocalDef(local_def) => local_def,
            LexicalScopeBinding::Item(binding) => LocalDef::from_def(binding.def().unwrap()),
        }
    }

    fn module(self) -> Option<Module<'a>> {
        match self {
            LexicalScopeBinding::Item(binding) => binding.module(),
            _ => None,
        }
    }
}

/// The link from a module up to its nearest parent node.
#[derive(Clone,Debug)]
enum ParentLink<'a> {
    NoParentLink,
    ModuleParentLink(Module<'a>, Name),
    BlockParentLink(Module<'a>, NodeId),
}

/// One node in the tree of modules.
pub struct ModuleS<'a> {
    parent_link: ParentLink<'a>,
    def: Option<Def>,

    // If the module is an extern crate, `def` is root of the external crate and `extern_crate_id`
    // is the NodeId of the local `extern crate` item (otherwise, `extern_crate_id` is None).
    extern_crate_id: Option<NodeId>,

    resolutions: RefCell<HashMap<(Name, Namespace), &'a RefCell<NameResolution<'a>>>>,
    unresolved_imports: RefCell<Vec<&'a ImportDirective<'a>>>,

    no_implicit_prelude: Cell<bool>,

    glob_importers: RefCell<Vec<(Module<'a>, &'a ImportDirective<'a>)>>,
    globs: RefCell<Vec<&'a ImportDirective<'a>>>,

    // Used to memoize the traits in this module for faster searches through all traits in scope.
    traits: RefCell<Option<Box<[(Name, &'a NameBinding<'a>)]>>>,

    // Whether this module is populated. If not populated, any attempt to
    // access the children must be preceded with a
    // `populate_module_if_necessary` call.
    populated: Cell<bool>,

    arenas: &'a ResolverArenas<'a>,
}

pub type Module<'a> = &'a ModuleS<'a>;

impl<'a> ModuleS<'a> {
    fn new(parent_link: ParentLink<'a>,
           def: Option<Def>,
           external: bool,
           arenas: &'a ResolverArenas<'a>) -> Self {
        ModuleS {
            parent_link: parent_link,
            def: def,
            extern_crate_id: None,
            resolutions: RefCell::new(HashMap::new()),
            unresolved_imports: RefCell::new(Vec::new()),
            no_implicit_prelude: Cell::new(false),
            glob_importers: RefCell::new(Vec::new()),
            globs: RefCell::new((Vec::new())),
            traits: RefCell::new(None),
            populated: Cell::new(!external),
            arenas: arenas
        }
    }

    fn for_each_child<F: FnMut(Name, Namespace, &'a NameBinding<'a>)>(&self, mut f: F) {
        for (&(name, ns), name_resolution) in self.resolutions.borrow().iter() {
            name_resolution.borrow().binding.map(|binding| f(name, ns, binding));
        }
    }

    fn def_id(&self) -> Option<DefId> {
        self.def.as_ref().map(Def::def_id)
    }

    // `self` resolves to the first module ancestor that `is_normal`.
    fn is_normal(&self) -> bool {
        match self.def {
            Some(Def::Mod(_)) => true,
            _ => false,
        }
    }

    fn is_trait(&self) -> bool {
        match self.def {
            Some(Def::Trait(_)) => true,
            _ => false,
        }
    }
}

impl<'a> fmt::Debug for ModuleS<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.def)
    }
}

// Records a possibly-private value, type, or module definition.
#[derive(Clone, Debug)]
pub struct NameBinding<'a> {
    kind: NameBindingKind<'a>,
    span: Span,
    vis: ty::Visibility,
}

#[derive(Clone, Debug)]
enum NameBindingKind<'a> {
    Def(Def),
    Module(Module<'a>),
    Import {
        binding: &'a NameBinding<'a>,
        directive: &'a ImportDirective<'a>,
        // Some(error) if using this imported name causes the import to be a privacy error
        privacy_error: Option<Box<PrivacyError<'a>>>,
    },
}

#[derive(Clone, Debug)]
struct PrivacyError<'a>(Span, Name, &'a NameBinding<'a>);

impl<'a> NameBinding<'a> {
    fn module(&self) -> Option<Module<'a>> {
        match self.kind {
            NameBindingKind::Module(module) => Some(module),
            NameBindingKind::Def(_) => None,
            NameBindingKind::Import { binding, .. } => binding.module(),
        }
    }

    fn def(&self) -> Option<Def> {
        match self.kind {
            NameBindingKind::Def(def) => Some(def),
            NameBindingKind::Module(module) => module.def,
            NameBindingKind::Import { binding, .. } => binding.def(),
        }
    }

    fn is_pseudo_public(&self) -> bool {
        self.pseudo_vis() == ty::Visibility::Public
    }

    // We sometimes need to treat variants as `pub` for backwards compatibility
    fn pseudo_vis(&self) -> ty::Visibility {
        if self.is_variant() { ty::Visibility::Public } else { self.vis }
    }

    fn is_variant(&self) -> bool {
        match self.kind {
            NameBindingKind::Def(Def::Variant(..)) => true,
            _ => false,
        }
    }

    fn is_extern_crate(&self) -> bool {
        self.module().and_then(|module| module.extern_crate_id).is_some()
    }

    fn is_import(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { .. } => true,
            _ => false,
        }
    }

    fn is_glob_import(&self) -> bool {
        match self.kind {
            NameBindingKind::Import { directive, .. } => directive.is_glob(),
            _ => false,
        }
    }

    fn is_importable(&self) -> bool {
        match self.def().unwrap() {
            Def::AssociatedConst(..) | Def::Method(..) | Def::AssociatedTy(..) => false,
            _ => true,
        }
    }
}

/// Interns the names of the primitive types.
struct PrimitiveTypeTable {
    primitive_types: HashMap<Name, PrimTy>,
}

impl PrimitiveTypeTable {
    fn new() -> PrimitiveTypeTable {
        let mut table = PrimitiveTypeTable { primitive_types: HashMap::new() };

        table.intern("bool", TyBool);
        table.intern("char", TyChar);
        table.intern("f32", TyFloat(FloatTy::F32));
        table.intern("f64", TyFloat(FloatTy::F64));
        table.intern("isize", TyInt(IntTy::Is));
        table.intern("i8", TyInt(IntTy::I8));
        table.intern("i16", TyInt(IntTy::I16));
        table.intern("i32", TyInt(IntTy::I32));
        table.intern("i64", TyInt(IntTy::I64));
        table.intern("str", TyStr);
        table.intern("usize", TyUint(UintTy::Us));
        table.intern("u8", TyUint(UintTy::U8));
        table.intern("u16", TyUint(UintTy::U16));
        table.intern("u32", TyUint(UintTy::U32));
        table.intern("u64", TyUint(UintTy::U64));

        table
    }

    fn intern(&mut self, string: &str, primitive_type: PrimTy) {
        self.primitive_types.insert(token::intern(string), primitive_type);
    }
}

/// The main resolver class.
pub struct Resolver<'a> {
    session: &'a Session,

    definitions: &'a mut Definitions,

    graph_root: Module<'a>,

    prelude: Option<Module<'a>>,

    trait_item_map: FnvHashMap<(Name, DefId), bool /* is static method? */>,

    structs: FnvHashMap<DefId, Vec<Name>>,

    // The number of imports that are currently unresolved.
    unresolved_imports: usize,

    // The module that represents the current item scope.
    current_module: Module<'a>,

    // The current set of local scopes, for values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    value_ribs: Vec<Rib<'a>>,

    // The current set of local scopes, for types.
    type_ribs: Vec<Rib<'a>>,

    // The current set of local scopes, for labels.
    label_ribs: Vec<Rib<'a>>,

    // The trait that the current context can refer to.
    current_trait_ref: Option<(DefId, TraitRef)>,

    // The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    // The idents for the primitive types.
    primitive_type_table: PrimitiveTypeTable,

    pub def_map: DefMap,
    pub freevars: FreevarMap,
    freevars_seen: NodeMap<NodeMap<usize>>,
    pub export_map: ExportMap,
    pub trait_map: TraitMap,

    // A map from nodes to modules, both normal (`mod`) modules and anonymous modules.
    // Anonymous modules are pseudo-modules that are implicitly created around items
    // contained within blocks.
    //
    // For example, if we have this:
    //
    //  fn f() {
    //      fn g() {
    //          ...
    //      }
    //  }
    //
    // There will be an anonymous module created around `g` with the ID of the
    // entry block for `f`.
    module_map: NodeMap<Module<'a>>,

    // Whether or not to print error messages. Can be set to true
    // when getting additional info for error message suggestions,
    // so as to avoid printing duplicate errors
    emit_errors: bool,

    pub make_glob_map: bool,
    // Maps imports to the names of items actually imported (this actually maps
    // all imports, but only glob imports are actually interesting).
    pub glob_map: GlobMap,

    used_imports: HashSet<(NodeId, Namespace)>,
    used_crates: HashSet<CrateNum>,
    pub maybe_unused_trait_imports: NodeSet,

    privacy_errors: Vec<PrivacyError<'a>>,

    arenas: &'a ResolverArenas<'a>,
}

struct ResolverArenas<'a> {
    modules: arena::TypedArena<ModuleS<'a>>,
    local_modules: RefCell<Vec<Module<'a>>>,
    name_bindings: arena::TypedArena<NameBinding<'a>>,
    import_directives: arena::TypedArena<ImportDirective<'a>>,
    name_resolutions: arena::TypedArena<RefCell<NameResolution<'a>>>,
}

impl<'a> ResolverArenas<'a> {
    fn alloc_module(&'a self, module: ModuleS<'a>) -> Module<'a> {
        let module = self.modules.alloc(module);
        if module.def_id().map(|def_id| def_id.is_local()).unwrap_or(true) {
            self.local_modules.borrow_mut().push(module);
        }
        module
    }
    fn local_modules(&'a self) -> ::std::cell::Ref<'a, Vec<Module<'a>>> {
        self.local_modules.borrow()
    }
    fn alloc_name_binding(&'a self, name_binding: NameBinding<'a>) -> &'a NameBinding<'a> {
        self.name_bindings.alloc(name_binding)
    }
    fn alloc_import_directive(&'a self, import_directive: ImportDirective<'a>)
                              -> &'a ImportDirective {
        self.import_directives.alloc(import_directive)
    }
    fn alloc_name_resolution(&'a self) -> &'a RefCell<NameResolution<'a>> {
        self.name_resolutions.alloc(Default::default())
    }
}

impl<'a> ty::NodeIdTree for Resolver<'a> {
    fn is_descendant_of(&self, node: NodeId, ancestor: NodeId) -> bool {
        let ancestor = self.definitions.local_def_id(ancestor);
        let mut module = *self.module_map.get(&node).unwrap();
        while module.def_id() != Some(ancestor) {
            let module_parent = match self.get_nearest_normal_module_parent(module) {
                Some(parent) => parent,
                None => return false,
            };
            module = module_parent;
        }
        true
    }
}

impl<'a> hir::lowering::Resolver for Resolver<'a> {
    fn resolve_generated_global_path(&mut self, path: &hir::Path, is_value: bool) -> Def {
        let namespace = if is_value { ValueNS } else { TypeNS };
        match self.resolve_crate_relative_path(path.span, &path.segments, namespace) {
            Ok(binding) => binding.def().unwrap(),
            Err(true) => Def::Err,
            Err(false) => {
                let path_name = &format!("{}", path);
                let error =
                    ResolutionError::UnresolvedName {
                        path: path_name,
                        message: "",
                        context: UnresolvedNameContext::Other,
                        is_static_method: false,
                        is_field: false,
                        def: Def::Err,
                    };
                resolve_error(self, path.span, error);
                Def::Err
            }
        }
    }

    fn get_resolution(&mut self, id: NodeId) -> Option<PathResolution> {
        self.def_map.get(&id).cloned()
    }

    fn record_resolution(&mut self, id: NodeId, def: Def) {
        self.def_map.insert(id, PathResolution::new(def));
    }

    fn definitions(&mut self) -> Option<&mut Definitions> {
        Some(self.definitions)
    }
}

trait Named {
    fn name(&self) -> Name;
}

impl Named for ast::PathSegment {
    fn name(&self) -> Name {
        self.identifier.name
    }
}

impl Named for hir::PathSegment {
    fn name(&self) -> Name {
        self.name
    }
}

impl<'a> Resolver<'a> {
    fn new(session: &'a Session,
           definitions: &'a mut Definitions,
           make_glob_map: MakeGlobMap,
           arenas: &'a ResolverArenas<'a>)
           -> Resolver<'a> {
        let root_def_id = definitions.local_def_id(CRATE_NODE_ID);
        let graph_root =
            ModuleS::new(NoParentLink, Some(Def::Mod(root_def_id)), false, arenas);
        let graph_root = arenas.alloc_module(graph_root);
        let mut module_map = NodeMap();
        module_map.insert(CRATE_NODE_ID, graph_root);

        Resolver {
            session: session,

            definitions: definitions,

            // The outermost module has def ID 0; this is not reflected in the
            // AST.
            graph_root: graph_root,
            prelude: None,

            trait_item_map: FnvHashMap(),
            structs: FnvHashMap(),

            unresolved_imports: 0,

            current_module: graph_root,
            value_ribs: vec![Rib::new(ModuleRibKind(graph_root))],
            type_ribs: vec![Rib::new(ModuleRibKind(graph_root))],
            label_ribs: Vec::new(),

            current_trait_ref: None,
            current_self_type: None,

            primitive_type_table: PrimitiveTypeTable::new(),

            def_map: NodeMap(),
            freevars: NodeMap(),
            freevars_seen: NodeMap(),
            export_map: NodeMap(),
            trait_map: NodeMap(),
            module_map: module_map,

            emit_errors: true,
            make_glob_map: make_glob_map == MakeGlobMap::Yes,
            glob_map: NodeMap(),

            used_imports: HashSet::new(),
            used_crates: HashSet::new(),
            maybe_unused_trait_imports: NodeSet(),

            privacy_errors: Vec::new(),

            arenas: arenas,
        }
    }

    fn arenas() -> ResolverArenas<'a> {
        ResolverArenas {
            modules: arena::TypedArena::new(),
            local_modules: RefCell::new(Vec::new()),
            name_bindings: arena::TypedArena::new(),
            import_directives: arena::TypedArena::new(),
            name_resolutions: arena::TypedArena::new(),
        }
    }

    fn new_module(&self, parent_link: ParentLink<'a>, def: Option<Def>, external: bool)
                  -> Module<'a> {
        self.arenas.alloc_module(ModuleS::new(parent_link, def, external, self.arenas))
    }

    fn new_extern_crate_module(&self, parent_link: ParentLink<'a>, def: Def, local_node_id: NodeId)
                               -> Module<'a> {
        let mut module = ModuleS::new(parent_link, Some(def), false, self.arenas);
        module.extern_crate_id = Some(local_node_id);
        self.arenas.modules.alloc(module)
    }

    fn get_ribs<'b>(&'b mut self, ns: Namespace) -> &'b mut Vec<Rib<'a>> {
        match ns { ValueNS => &mut self.value_ribs, TypeNS => &mut self.type_ribs }
    }

    #[inline]
    fn record_use(&mut self, name: Name, binding: &'a NameBinding<'a>) {
        // track extern crates for unused_extern_crate lint
        if let Some(DefId { krate, .. }) = binding.module().and_then(ModuleS::def_id) {
            self.used_crates.insert(krate);
        }

        let (directive, privacy_error) = match binding.kind {
            NameBindingKind::Import { directive, ref privacy_error, .. } =>
                (directive, privacy_error),
            _ => return,
        };

        if let Some(error) = privacy_error.as_ref() {
            self.privacy_errors.push((**error).clone());
        }

        if !self.make_glob_map {
            return;
        }
        if self.glob_map.contains_key(&directive.id) {
            self.glob_map.get_mut(&directive.id).unwrap().insert(name);
            return;
        }

        let mut new_set = FnvHashSet();
        new_set.insert(name);
        self.glob_map.insert(directive.id, new_set);
    }

    /// Resolves the given module path from the given root `module_`.
    fn resolve_module_path_from_root(&mut self,
                                     module_: Module<'a>,
                                     module_path: &[Name],
                                     index: usize,
                                     span: Span)
                                     -> ResolveResult<Module<'a>> {
        fn search_parent_externals(needle: Name, module: Module) -> Option<Module> {
            match module.resolve_name(needle, TypeNS, false) {
                Success(binding) if binding.is_extern_crate() => Some(module),
                _ => match module.parent_link {
                    ModuleParentLink(ref parent, _) => {
                        search_parent_externals(needle, parent)
                    }
                    _ => None,
                },
            }
        }

        let mut search_module = module_;
        let mut index = index;
        let module_path_len = module_path.len();

        // Resolve the module part of the path. This does not involve looking
        // upward though scope chains; we simply resolve names directly in
        // modules as we go.
        while index < module_path_len {
            let name = module_path[index];
            match self.resolve_name_in_module(search_module, name, TypeNS, false, true) {
                Failed(None) => {
                    let segment_name = name.as_str();
                    let module_name = module_to_string(search_module);
                    let msg = if "???" == &module_name {
                        match search_parent_externals(name, &self.current_module) {
                            Some(module) => {
                                let path_str = names_to_string(module_path);
                                let target_mod_str = module_to_string(&module);
                                let current_mod_str = module_to_string(&self.current_module);

                                let prefix = if target_mod_str == current_mod_str {
                                    "self::".to_string()
                                } else {
                                    format!("{}::", target_mod_str)
                                };

                                format!("Did you mean `{}{}`?", prefix, path_str)
                            }
                            None => format!("Maybe a missing `extern crate {}`?", segment_name),
                        }
                    } else {
                        format!("Could not find `{}` in `{}`", segment_name, module_name)
                    };

                    return Failed(Some((span, msg)));
                }
                Failed(err) => return Failed(err),
                Indeterminate => {
                    debug!("(resolving module path for import) module resolution is \
                            indeterminate: {}",
                           name);
                    return Indeterminate;
                }
                Success(binding) => {
                    // Check to see whether there are type bindings, and, if
                    // so, whether there is a module within.
                    if let Some(module_def) = binding.module() {
                        self.check_privacy(name, binding, span);
                        search_module = module_def;
                    } else {
                        let msg = format!("Not a module `{}`", name);
                        return Failed(Some((span, msg)));
                    }
                }
            }

            index += 1;
        }

        return Success(search_module);
    }

    /// Attempts to resolve the module part of an import directive or path
    /// rooted at the given module.
    fn resolve_module_path(&mut self,
                           module_path: &[Name],
                           use_lexical_scope: UseLexicalScopeFlag,
                           span: Span)
                           -> ResolveResult<Module<'a>> {
        if module_path.len() == 0 {
            return Success(self.graph_root) // Use the crate root
        }

        debug!("(resolving module path for import) processing `{}` rooted at `{}`",
               names_to_string(module_path),
               module_to_string(self.current_module));

        // Resolve the module prefix, if any.
        let module_prefix_result = self.resolve_module_prefix(module_path, span);

        let search_module;
        let start_index;
        match module_prefix_result {
            Failed(err) => return Failed(err),
            Indeterminate => {
                debug!("(resolving module path for import) indeterminate; bailing");
                return Indeterminate;
            }
            Success(NoPrefixFound) => {
                // There was no prefix, so we're considering the first element
                // of the path. How we handle this depends on whether we were
                // instructed to use lexical scope or not.
                match use_lexical_scope {
                    DontUseLexicalScope => {
                        // This is a crate-relative path. We will start the
                        // resolution process at index zero.
                        search_module = self.graph_root;
                        start_index = 0;
                    }
                    UseLexicalScope => {
                        // This is not a crate-relative path. We resolve the
                        // first component of the path in the current lexical
                        // scope and then proceed to resolve below that.
                        let ident = ast::Ident::with_empty_ctxt(module_path[0]);
                        match self.resolve_ident_in_lexical_scope(ident, TypeNS, true)
                                  .and_then(LexicalScopeBinding::module) {
                            None => return Failed(None),
                            Some(containing_module) => {
                                search_module = containing_module;
                                start_index = 1;
                            }
                        }
                    }
                }
            }
            Success(PrefixFound(ref containing_module, index)) => {
                search_module = containing_module;
                start_index = index;
            }
        }

        self.resolve_module_path_from_root(search_module,
                                           module_path,
                                           start_index,
                                           span)
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
    fn resolve_ident_in_lexical_scope(&mut self,
                                      ident: ast::Ident,
                                      ns: Namespace,
                                      record_used: bool)
                                      -> Option<LexicalScopeBinding<'a>> {
        let name = match ns { ValueNS => mtwt::resolve(ident), TypeNS => ident.name };

        // Walk backwards up the ribs in scope.
        for i in (0 .. self.get_ribs(ns).len()).rev() {
            if let Some(def) = self.get_ribs(ns)[i].bindings.get(&name).cloned() {
                // The ident resolves to a type parameter or local variable.
                return Some(LexicalScopeBinding::LocalDef(LocalDef {
                    ribs: Some((ns, i)),
                    def: def,
                }));
            }

            if let ModuleRibKind(module) = self.get_ribs(ns)[i].kind {
                let name = ident.name;
                let item = self.resolve_name_in_module(module, name, ns, true, record_used);
                if let Success(binding) = item {
                    // The ident resolves to an item.
                    return Some(LexicalScopeBinding::Item(binding));
                }

                // We can only see through anonymous modules
                if module.def.is_some() {
                    return match self.prelude {
                        Some(prelude) if !module.no_implicit_prelude.get() => {
                            prelude.resolve_name(name, ns, false).success()
                                   .map(LexicalScopeBinding::Item)
                        }
                        _ => None,
                    };
                }
            }
        }

        None
    }

    /// Returns the nearest normal module parent of the given module.
    fn get_nearest_normal_module_parent(&self, module_: Module<'a>) -> Option<Module<'a>> {
        let mut module_ = module_;
        loop {
            match module_.parent_link {
                NoParentLink => return None,
                ModuleParentLink(new_module, _) |
                BlockParentLink(new_module, _) => {
                    let new_module = new_module;
                    if new_module.is_normal() {
                        return Some(new_module);
                    }
                    module_ = new_module;
                }
            }
        }
    }

    /// Returns the nearest normal module parent of the given module, or the
    /// module itself if it is a normal module.
    fn get_nearest_normal_module_parent_or_self(&self, module_: Module<'a>) -> Module<'a> {
        if module_.is_normal() {
            return module_;
        }
        match self.get_nearest_normal_module_parent(module_) {
            None => module_,
            Some(new_module) => new_module,
        }
    }

    /// Resolves a "module prefix". A module prefix is one or both of (a) `self::`;
    /// (b) some chain of `super::`.
    /// grammar: (SELF MOD_SEP ) ? (SUPER MOD_SEP) *
    fn resolve_module_prefix(&mut self, module_path: &[Name], span: Span)
                             -> ResolveResult<ModulePrefixResult<'a>> {
        // Start at the current module if we see `self` or `super`, or at the
        // top of the crate otherwise.
        let mut i = match &*module_path[0].as_str() {
            "self" => 1,
            "super" => 0,
            _ => return Success(NoPrefixFound),
        };
        let module_ = self.current_module;
        let mut containing_module = self.get_nearest_normal_module_parent_or_self(module_);

        // Now loop through all the `super`s we find.
        while i < module_path.len() && "super" == module_path[i].as_str() {
            debug!("(resolving module prefix) resolving `super` at {}",
                   module_to_string(&containing_module));
            match self.get_nearest_normal_module_parent(containing_module) {
                None => {
                    let msg = "There are too many initial `super`s.".into();
                    return Failed(Some((span, msg)));
                }
                Some(new_module) => {
                    containing_module = new_module;
                    i += 1;
                }
            }
        }

        debug!("(resolving module prefix) finished resolving prefix at {}",
               module_to_string(&containing_module));

        return Success(PrefixFound(containing_module, i));
    }

    /// Attempts to resolve the supplied name in the given module for the
    /// given namespace. If successful, returns the binding corresponding to
    /// the name.
    fn resolve_name_in_module(&mut self,
                              module: Module<'a>,
                              name: Name,
                              namespace: Namespace,
                              use_lexical_scope: bool,
                              record_used: bool)
                              -> ResolveResult<&'a NameBinding<'a>> {
        debug!("(resolving name in module) resolving `{}` in `{}`", name, module_to_string(module));

        self.populate_module_if_necessary(module);
        module.resolve_name(name, namespace, use_lexical_scope).and_then(|binding| {
            if record_used {
                if let NameBindingKind::Import { directive, .. } = binding.kind {
                    self.used_imports.insert((directive.id, namespace));
                }
                self.record_use(name, binding);
            }
            Success(binding)
        })
    }

    // AST resolution
    //
    // We maintain a list of value ribs and type ribs.
    //
    // Simultaneously, we keep track of the current position in the module
    // graph in the `current_module` pointer. When we go to resolve a name in
    // the value or type namespaces, we first look through all the ribs and
    // then query the module graph. When we resolve a name in the module
    // namespace, we can skip all the ribs (since nested modules are not
    // allowed within blocks in Rust) and jump straight to the current module
    // graph node.
    //
    // Named implementations are handled separately. When we find a method
    // call, we consult the module node to find all of the implementations in
    // scope. This information is lazily cached in the module node. We then
    // generate a fake "implementation scope" containing all the
    // implementations thus found, for compatibility with old resolve pass.

    fn with_scope<F>(&mut self, id: NodeId, f: F)
        where F: FnOnce(&mut Resolver)
    {
        let module = self.module_map.get(&id).cloned(); // clones a reference
        if let Some(module) = module {
            // Move down in the graph.
            let orig_module = ::std::mem::replace(&mut self.current_module, module);
            self.value_ribs.push(Rib::new(ModuleRibKind(module)));
            self.type_ribs.push(Rib::new(ModuleRibKind(module)));

            f(self);

            self.current_module = orig_module;
            self.value_ribs.pop();
            self.type_ribs.pop();
        } else {
            f(self);
        }
    }

    /// Searches the current set of local scopes for labels.
    /// Stops after meeting a closure.
    fn search_label(&self, name: Name) -> Option<Def> {
        for rib in self.label_ribs.iter().rev() {
            match rib.kind {
                NormalRibKind => {
                    // Continue
                }
                _ => {
                    // Do not resolve labels across function boundary
                    return None;
                }
            }
            let result = rib.bindings.get(&name).cloned();
            if result.is_some() {
                return result;
            }
        }
        None
    }

    fn resolve_crate(&mut self, krate: &Crate) {
        debug!("(resolving crate) starting");
        self.current_module = self.graph_root;
        visit::walk_crate(self, krate);
    }

    fn resolve_item(&mut self, item: &Item) {
        let name = item.ident.name;

        debug!("(resolving item) resolving {}", name);

        match item.node {
            ItemKind::Enum(_, ref generics) |
            ItemKind::Ty(_, ref generics) |
            ItemKind::Struct(_, ref generics) => {
                self.with_type_parameter_rib(HasTypeParameters(generics, TypeSpace, ItemRibKind),
                                             |this| visit::walk_item(this, item));
            }
            ItemKind::Fn(_, _, _, _, ref generics, _) => {
                self.with_type_parameter_rib(HasTypeParameters(generics, FnSpace, ItemRibKind),
                                             |this| visit::walk_item(this, item));
            }

            ItemKind::DefaultImpl(_, ref trait_ref) => {
                self.with_optional_trait_ref(Some(trait_ref), |_, _| {});
            }
            ItemKind::Impl(_, _, ref generics, ref opt_trait_ref, ref self_type, ref impl_items) =>
                self.resolve_implementation(generics,
                                            opt_trait_ref,
                                            &self_type,
                                            item.id,
                                            impl_items),

            ItemKind::Trait(_, ref generics, ref bounds, ref trait_items) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               TypeSpace,
                                                               ItemRibKind),
                                             |this| {
                    let local_def_id = this.definitions.local_def_id(item.id);
                    this.with_self_rib(Def::SelfTy(Some(local_def_id), None), |this| {
                        this.visit_generics(generics);
                        walk_list!(this, visit_ty_param_bound, bounds);

                        for trait_item in trait_items {
                            match trait_item.node {
                                TraitItemKind::Const(_, ref default) => {
                                    // Only impose the restrictions of
                                    // ConstRibKind if there's an actual constant
                                    // expression in a provided default.
                                    if default.is_some() {
                                        this.with_constant_rib(|this| {
                                            visit::walk_trait_item(this, trait_item)
                                        });
                                    } else {
                                        visit::walk_trait_item(this, trait_item)
                                    }
                                }
                                TraitItemKind::Method(ref sig, _) => {
                                    let type_parameters =
                                        HasTypeParameters(&sig.generics,
                                                          FnSpace,
                                                          MethodRibKind(!sig.decl.has_self()));
                                    this.with_type_parameter_rib(type_parameters, |this| {
                                        visit::walk_trait_item(this, trait_item)
                                    });
                                }
                                TraitItemKind::Type(..) => {
                                    this.with_type_parameter_rib(NoTypeParameters, |this| {
                                        visit::walk_trait_item(this, trait_item)
                                    });
                                }
                                TraitItemKind::Macro(_) => panic!("unexpanded macro in resolve!"),
                            };
                        }
                    });
                });
            }

            ItemKind::Mod(_) | ItemKind::ForeignMod(_) => {
                self.with_scope(item.id, |this| {
                    visit::walk_item(this, item);
                });
            }

            ItemKind::Const(..) | ItemKind::Static(..) => {
                self.with_constant_rib(|this| {
                    visit::walk_item(this, item);
                });
            }

            ItemKind::Use(ref view_path) => {
                match view_path.node {
                    ast::ViewPathList(ref prefix, ref items) => {
                        // Resolve prefix of an import with empty braces (issue #28388)
                        if items.is_empty() && !prefix.segments.is_empty() {
                            match self.resolve_crate_relative_path(prefix.span,
                                                                   &prefix.segments,
                                                                   TypeNS) {
                                Ok(binding) => {
                                    let def = binding.def().unwrap();
                                    self.record_def(item.id, PathResolution::new(def));
                                }
                                Err(true) => self.record_def(item.id, err_path_resolution()),
                                Err(false) => {
                                    resolve_error(self,
                                                  prefix.span,
                                                  ResolutionError::FailedToResolve(
                                                      &path_names_to_string(prefix, 0)));
                                    self.record_def(item.id, err_path_resolution());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            ItemKind::ExternCrate(_) => {
                // do nothing, these are just around to be encoded
            }

            ItemKind::Mac(_) => panic!("unexpanded macro in resolve!"),
        }
    }

    fn with_type_parameter_rib<'b, F>(&'b mut self, type_parameters: TypeParameters<'a, 'b>, f: F)
        where F: FnOnce(&mut Resolver)
    {
        match type_parameters {
            HasTypeParameters(generics, space, rib_kind) => {
                let mut function_type_rib = Rib::new(rib_kind);
                let mut seen_bindings = HashSet::new();
                for (index, type_parameter) in generics.ty_params.iter().enumerate() {
                    let name = type_parameter.ident.name;
                    debug!("with_type_parameter_rib: {}", type_parameter.id);

                    if seen_bindings.contains(&name) {
                        resolve_error(self,
                                      type_parameter.span,
                                      ResolutionError::NameAlreadyUsedInTypeParameterList(name));
                    }
                    seen_bindings.insert(name);

                    // plain insert (no renaming)
                    let def_id = self.definitions.local_def_id(type_parameter.id);
                    let def = Def::TyParam(space, index as u32, def_id, name);
                    function_type_rib.bindings.insert(name, def);
                }
                self.type_ribs.push(function_type_rib);
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f(self);

        if let HasTypeParameters(..) = type_parameters {
            self.type_ribs.pop();
        }
    }

    fn with_label_rib<F>(&mut self, f: F)
        where F: FnOnce(&mut Resolver)
    {
        self.label_ribs.push(Rib::new(NormalRibKind));
        f(self);
        self.label_ribs.pop();
    }

    fn with_constant_rib<F>(&mut self, f: F)
        where F: FnOnce(&mut Resolver)
    {
        self.value_ribs.push(Rib::new(ConstantItemRibKind));
        self.type_ribs.push(Rib::new(ConstantItemRibKind));
        f(self);
        self.type_ribs.pop();
        self.value_ribs.pop();
    }

    fn resolve_function(&mut self,
                        rib_kind: RibKind<'a>,
                        declaration: &FnDecl,
                        block: &Block) {
        // Create a value rib for the function.
        self.value_ribs.push(Rib::new(rib_kind));

        // Create a label rib for the function.
        self.label_ribs.push(Rib::new(rib_kind));

        // Add each argument to the rib.
        let mut bindings_list = HashMap::new();
        for argument in &declaration.inputs {
            self.resolve_pattern(&argument.pat, PatternSource::FnParam, &mut bindings_list);

            self.visit_ty(&argument.ty);

            debug!("(resolving function) recorded argument");
        }
        visit::walk_fn_ret_ty(self, &declaration.output);

        // Resolve the function body.
        self.visit_block(block);

        debug!("(resolving function) leaving function");

        self.label_ribs.pop();
        self.value_ribs.pop();
    }

    fn resolve_trait_reference(&mut self,
                               id: NodeId,
                               trait_path: &Path,
                               path_depth: usize)
                               -> Result<PathResolution, ()> {
        self.resolve_path(id, trait_path, path_depth, TypeNS).and_then(|path_res| {
            if let Def::Trait(_) = path_res.base_def {
                debug!("(resolving trait) found trait def: {:?}", path_res);
                Ok(path_res)
            } else {
                let mut err =
                    resolve_struct_error(self,
                                  trait_path.span,
                                  ResolutionError::IsNotATrait(&path_names_to_string(trait_path,
                                                                                      path_depth)));

                // If it's a typedef, give a note
                if let Def::TyAlias(..) = path_res.base_def {
                    let trait_name = trait_path.segments.last().unwrap().identifier.name;
                    err.span_label(trait_path.span,
                                   &format!("`{}` is not a trait", trait_name));

                    let definition_site = {
                        let segments = &trait_path.segments;
                        if trait_path.global {
                            self.resolve_crate_relative_path(trait_path.span, segments, TypeNS)
                        } else {
                            self.resolve_module_relative_path(trait_path.span, segments, TypeNS)
                        }.map(|binding| binding.span).unwrap_or(syntax_pos::DUMMY_SP)
                    };

                    if definition_site != syntax_pos::DUMMY_SP {
                        err.span_label(definition_site,
                                       &format!("type aliases cannot be used for traits"));
                    }
                }
                err.emit();
                Err(true)
            }
        }).map_err(|error_reported| {
            if error_reported { return }

            // find possible candidates
            let trait_name = trait_path.segments.last().unwrap().identifier.name;
            let candidates =
                self.lookup_candidates(
                    trait_name,
                    TypeNS,
                    |def| match def {
                        Def::Trait(_) => true,
                        _             => false,
                    },
                );

            // create error object
            let name = &path_names_to_string(trait_path, path_depth);
            let error =
                ResolutionError::UndeclaredTraitName(
                    name,
                    candidates,
                );

            resolve_error(self, trait_path.span, error);
        })
    }

    fn with_current_self_type<T, F>(&mut self, self_type: &Ty, f: F) -> T
        where F: FnOnce(&mut Resolver) -> T
    {
        // Handle nested impls (inside fn bodies)
        let previous_value = replace(&mut self.current_self_type, Some(self_type.clone()));
        let result = f(self);
        self.current_self_type = previous_value;
        result
    }

    fn with_optional_trait_ref<T, F>(&mut self, opt_trait_ref: Option<&TraitRef>, f: F) -> T
        where F: FnOnce(&mut Resolver, Option<DefId>) -> T
    {
        let mut new_val = None;
        let mut new_id = None;
        if let Some(trait_ref) = opt_trait_ref {
            if let Ok(path_res) = self.resolve_trait_reference(trait_ref.ref_id,
                                                               &trait_ref.path,
                                                               0) {
                assert!(path_res.depth == 0);
                self.record_def(trait_ref.ref_id, path_res);
                new_val = Some((path_res.base_def.def_id(), trait_ref.clone()));
                new_id = Some(path_res.base_def.def_id());
            } else {
                self.record_def(trait_ref.ref_id, err_path_resolution());
            }
            visit::walk_trait_ref(self, trait_ref);
        }
        let original_trait_ref = replace(&mut self.current_trait_ref, new_val);
        let result = f(self, new_id);
        self.current_trait_ref = original_trait_ref;
        result
    }

    fn with_self_rib<F>(&mut self, self_def: Def, f: F)
        where F: FnOnce(&mut Resolver)
    {
        let mut self_type_rib = Rib::new(NormalRibKind);

        // plain insert (no renaming, types are not currently hygienic....)
        self_type_rib.bindings.insert(keywords::SelfType.name(), self_def);
        self.type_ribs.push(self_type_rib);
        f(self);
        self.type_ribs.pop();
    }

    fn resolve_implementation(&mut self,
                              generics: &Generics,
                              opt_trait_reference: &Option<TraitRef>,
                              self_type: &Ty,
                              item_id: NodeId,
                              impl_items: &[ImplItem]) {
        // If applicable, create a rib for the type parameters.
        self.with_type_parameter_rib(HasTypeParameters(generics,
                                                       TypeSpace,
                                                       ItemRibKind),
                                     |this| {
            // Resolve the type parameters.
            this.visit_generics(generics);

            // Resolve the trait reference, if necessary.
            this.with_optional_trait_ref(opt_trait_reference.as_ref(), |this, trait_id| {
                // Resolve the self type.
                this.visit_ty(self_type);

                this.with_self_rib(Def::SelfTy(trait_id, Some(item_id)), |this| {
                    this.with_current_self_type(self_type, |this| {
                        for impl_item in impl_items {
                            this.resolve_visibility(&impl_item.vis);
                            match impl_item.node {
                                ImplItemKind::Const(..) => {
                                    // If this is a trait impl, ensure the const
                                    // exists in trait
                                    this.check_trait_item(impl_item.ident.name,
                                                          impl_item.span,
                                        |n, s| ResolutionError::ConstNotMemberOfTrait(n, s));
                                    visit::walk_impl_item(this, impl_item);
                                }
                                ImplItemKind::Method(ref sig, _) => {
                                    // If this is a trait impl, ensure the method
                                    // exists in trait
                                    this.check_trait_item(impl_item.ident.name,
                                                          impl_item.span,
                                        |n, s| ResolutionError::MethodNotMemberOfTrait(n, s));

                                    // We also need a new scope for the method-
                                    // specific type parameters.
                                    let type_parameters =
                                        HasTypeParameters(&sig.generics,
                                                          FnSpace,
                                                          MethodRibKind(!sig.decl.has_self()));
                                    this.with_type_parameter_rib(type_parameters, |this| {
                                        visit::walk_impl_item(this, impl_item);
                                    });
                                }
                                ImplItemKind::Type(ref ty) => {
                                    // If this is a trait impl, ensure the type
                                    // exists in trait
                                    this.check_trait_item(impl_item.ident.name,
                                                          impl_item.span,
                                        |n, s| ResolutionError::TypeNotMemberOfTrait(n, s));

                                    this.visit_ty(ty);
                                }
                                ImplItemKind::Macro(_) => panic!("unexpanded macro in resolve!"),
                            }
                        }
                    });
                });
            });
        });
    }

    fn check_trait_item<F>(&self, name: Name, span: Span, err: F)
        where F: FnOnce(Name, &str) -> ResolutionError
    {
        // If there is a TraitRef in scope for an impl, then the method must be in the
        // trait.
        if let Some((did, ref trait_ref)) = self.current_trait_ref {
            if !self.trait_item_map.contains_key(&(name, did)) {
                let path_str = path_names_to_string(&trait_ref.path, 0);
                resolve_error(self, span, err(name, &path_str));
            }
        }
    }

    fn resolve_local(&mut self, local: &Local) {
        // Resolve the type.
        walk_list!(self, visit_ty, &local.ty);

        // Resolve the initializer.
        walk_list!(self, visit_expr, &local.init);

        // Resolve the pattern.
        self.resolve_pattern(&local.pat, PatternSource::Let, &mut HashMap::new());
    }

    // build a map from pattern identifiers to binding-info's.
    // this is done hygienically. This could arise for a macro
    // that expands into an or-pattern where one 'x' was from the
    // user and one 'x' came from the macro.
    fn binding_mode_map(&mut self, pat: &Pat) -> BindingMap {
        let mut binding_map = HashMap::new();

        pat.walk(&mut |pat| {
            if let PatKind::Ident(binding_mode, ident, ref sub_pat) = pat.node {
                if sub_pat.is_some() || match self.def_map.get(&pat.id) {
                    Some(&PathResolution { base_def: Def::Local(..), .. }) => true,
                    _ => false,
                } {
                    let binding_info = BindingInfo { span: ident.span, binding_mode: binding_mode };
                    binding_map.insert(mtwt::resolve(ident.node), binding_info);
                }
            }
            true
        });

        binding_map
    }

    // check that all of the arms in an or-pattern have exactly the
    // same set of bindings, with the same binding modes for each.
    fn check_consistent_bindings(&mut self, arm: &Arm) {
        if arm.pats.is_empty() {
            return;
        }
        let map_0 = self.binding_mode_map(&arm.pats[0]);
        for (i, p) in arm.pats.iter().enumerate() {
            let map_i = self.binding_mode_map(&p);

            for (&key, &binding_0) in &map_0 {
                match map_i.get(&key) {
                    None => {
                        resolve_error(self,
                                      p.span,
                                      ResolutionError::VariableNotBoundInPattern(key, 1, i + 1));
                    }
                    Some(binding_i) => {
                        if binding_0.binding_mode != binding_i.binding_mode {
                            resolve_error(self,
                                          binding_i.span,
                                          ResolutionError::VariableBoundWithDifferentMode(key,
                                                                                          i + 1));
                        }
                    }
                }
            }

            for (&key, &binding) in &map_i {
                if !map_0.contains_key(&key) {
                    resolve_error(self,
                                  binding.span,
                                  ResolutionError::VariableNotBoundInPattern(key, i + 1, 1));
                }
            }
        }
    }

    fn resolve_arm(&mut self, arm: &Arm) {
        self.value_ribs.push(Rib::new(NormalRibKind));

        let mut bindings_list = HashMap::new();
        for pattern in &arm.pats {
            self.resolve_pattern(&pattern, PatternSource::Match, &mut bindings_list);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        walk_list!(self, visit_expr, &arm.guard);
        self.visit_expr(&arm.body);

        self.value_ribs.pop();
    }

    fn resolve_block(&mut self, block: &Block) {
        debug!("(resolving block) entering block");
        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        let anonymous_module = self.module_map.get(&block.id).cloned(); // clones a reference

        if let Some(anonymous_module) = anonymous_module {
            debug!("(resolving block) found anonymous module, moving down");
            self.value_ribs.push(Rib::new(ModuleRibKind(anonymous_module)));
            self.type_ribs.push(Rib::new(ModuleRibKind(anonymous_module)));
            self.current_module = anonymous_module;
        } else {
            self.value_ribs.push(Rib::new(NormalRibKind));
        }

        // Descend into the block.
        visit::walk_block(self, block);

        // Move back up.
        self.current_module = orig_module;
        self.value_ribs.pop();
        if let Some(_) = anonymous_module {
            self.type_ribs.pop();
        }
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(&mut self, ty: &Ty) {
        match ty.node {
            TyKind::Path(ref maybe_qself, ref path) => {
                // This is a path in the type namespace. Walk through scopes
                // looking for it.
                if let Some(def) = self.resolve_possibly_assoc_item(ty.id, maybe_qself.as_ref(),
                                                                    path, TypeNS) {
                    match def.base_def {
                        Def::Mod(..) if def.depth == 0 => {
                            self.session.span_err(path.span, "expected type, found module");
                            self.record_def(ty.id, err_path_resolution());
                        }
                        _ => {
                            // Write the result into the def map.
                            debug!("(resolving type) writing resolution for `{}` (id {}) = {:?}",
                                   path_names_to_string(path, 0), ty.id, def);
                            self.record_def(ty.id, def);
                        }
                    }
                } else {
                    self.record_def(ty.id, err_path_resolution());

                    // Keep reporting some errors even if they're ignored above.
                    if let Err(true) = self.resolve_path(ty.id, path, 0, TypeNS) {
                        // `resolve_path` already reported the error
                    } else {
                        let kind = if maybe_qself.is_some() {
                            "associated type"
                        } else {
                            "type name"
                        };

                        let is_invalid_self_type_name = path.segments.len() > 0 &&
                                                        maybe_qself.is_none() &&
                                                        path.segments[0].identifier.name ==
                                                        keywords::SelfType.name();
                        if is_invalid_self_type_name {
                            resolve_error(self,
                                          ty.span,
                                          ResolutionError::SelfUsedOutsideImplOrTrait);
                        } else {
                            let segment = path.segments.last();
                            let segment = segment.expect("missing name in path");
                            let type_name = segment.identifier.name;

                            let candidates =
                                self.lookup_candidates(
                                    type_name,
                                    TypeNS,
                                    |def| match def {
                                        Def::Trait(_) |
                                        Def::Enum(_) |
                                        Def::Struct(_) |
                                        Def::TyAlias(_) => true,
                                        _               => false,
                                    },
                                );

                            // create error object
                            let name = &path_names_to_string(path, 0);
                            let error =
                                ResolutionError::UseOfUndeclared(
                                    kind,
                                    name,
                                    candidates,
                                );

                            resolve_error(self, ty.span, error);
                        }
                    }
                }
            }
            _ => {}
        }
        // Resolve embedded types.
        visit::walk_ty(self, ty);
    }

    fn fresh_binding(&mut self,
                     ident: &ast::SpannedIdent,
                     pat_id: NodeId,
                     outer_pat_id: NodeId,
                     pat_src: PatternSource,
                     bindings: &mut HashMap<Name, NodeId>)
                     -> PathResolution {
        // Add the binding to the local ribs, if it
        // doesn't already exist in the bindings map. (We
        // must not add it if it's in the bindings map
        // because that breaks the assumptions later
        // passes make about or-patterns.)
        let renamed = mtwt::resolve(ident.node);
        let def = match bindings.get(&renamed).cloned() {
            Some(id) if id == outer_pat_id => {
                // `Variant(a, a)`, error
                resolve_error(
                    self,
                    ident.span,
                    ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(
                        &ident.node.name.as_str())
                );
                Def::Err
            }
            Some(..) if pat_src == PatternSource::FnParam => {
                // `fn f(a: u8, a: u8)`, error
                resolve_error(
                    self,
                    ident.span,
                    ResolutionError::IdentifierBoundMoreThanOnceInParameterList(
                        &ident.node.name.as_str())
                );
                Def::Err
            }
            Some(..) if pat_src == PatternSource::Match => {
                // `Variant1(a) | Variant2(a)`, ok
                // Reuse definition from the first `a`.
                self.value_ribs.last_mut().unwrap().bindings[&renamed]
            }
            Some(..) => {
                span_bug!(ident.span, "two bindings with the same name from \
                                       unexpected pattern source {:?}", pat_src);
            }
            None => {
                // A completely fresh binding, add to the lists.
                // FIXME: Later stages are not ready to deal with `Def::Err` here yet, so
                // define `Invalid` bindings as `Def::Local`, just don't add them to the lists.
                let def = Def::Local(self.definitions.local_def_id(pat_id), pat_id);
                if ident.node.name != keywords::Invalid.name() {
                    bindings.insert(renamed, outer_pat_id);
                    self.value_ribs.last_mut().unwrap().bindings.insert(renamed, def);
                }
                def
            }
        };

        PathResolution::new(def)
    }

    fn resolve_pattern_path<ExpectedFn>(&mut self,
                                        pat_id: NodeId,
                                        qself: Option<&QSelf>,
                                        path: &Path,
                                        namespace: Namespace,
                                        expected_fn: ExpectedFn,
                                        expected_what: &str)
        where ExpectedFn: FnOnce(Def) -> bool
    {
        let resolution = if let Some(resolution) = self.resolve_possibly_assoc_item(pat_id,
                                                                        qself, path, namespace) {
            if resolution.depth == 0 {
                if expected_fn(resolution.base_def) {
                    resolution
                } else {
                    resolve_error(
                        self,
                        path.span,
                        ResolutionError::PatPathUnexpected(expected_what,
                                                           resolution.kind_name(), path)
                    );
                    err_path_resolution()
                }
            } else {
                // Not fully resolved associated item `T::A::B` or `<T as Tr>::A::B`
                // or `<T>::A::B`. If `B` should be resolved in value namespace then
                // it needs to be added to the trait map.
                if namespace == ValueNS {
                    let item_name = path.segments.last().unwrap().identifier.name;
                    let traits = self.get_traits_containing_item(item_name);
                    self.trait_map.insert(pat_id, traits);
                }
                resolution
            }
        } else {
            if let Err(false) = self.resolve_path(pat_id, path, 0, namespace) {
                resolve_error(
                    self,
                    path.span,
                    ResolutionError::PatPathUnresolved(expected_what, path)
                );
            }
            err_path_resolution()
        };

        self.record_def(pat_id, resolution);
    }

    fn resolve_pattern(&mut self,
                       pat: &Pat,
                       pat_src: PatternSource,
                       // Maps idents to the node ID for the
                       // outermost pattern that binds them.
                       bindings: &mut HashMap<Name, NodeId>) {
        // Visit all direct subpatterns of this pattern.
        let outer_pat_id = pat.id;
        pat.walk(&mut |pat| {
            match pat.node {
                PatKind::Ident(bmode, ref ident, ref opt_pat) => {
                    // First try to resolve the identifier as some existing
                    // entity, then fall back to a fresh binding.
                    let resolution = if let Ok(resolution) = self.resolve_path(pat.id,
                                &Path::from_ident(ident.span, ident.node), 0, ValueNS) {
                        let always_binding = !pat_src.is_refutable() || opt_pat.is_some() ||
                                             bmode != BindingMode::ByValue(Mutability::Immutable);
                        match resolution.base_def {
                            Def::Struct(..) | Def::Variant(..) |
                            Def::Const(..) | Def::AssociatedConst(..) if !always_binding => {
                                // A constant, unit variant, etc pattern.
                                resolution
                            }
                            Def::Struct(..) | Def::Variant(..) |
                            Def::Const(..) | Def::AssociatedConst(..) | Def::Static(..) => {
                                // A fresh binding that shadows something unacceptable.
                                resolve_error(
                                    self,
                                    ident.span,
                                    ResolutionError::BindingShadowsSomethingUnacceptable(
                                        pat_src.descr(), resolution.kind_name(), ident.node.name)
                                );
                                err_path_resolution()
                            }
                            Def::Local(..) | Def::Upvar(..) | Def::Fn(..) | Def::Err => {
                                // These entities are explicitly allowed
                                // to be shadowed by fresh bindings.
                                self.fresh_binding(ident, pat.id, outer_pat_id,
                                                   pat_src, bindings)
                            }
                            def => {
                                span_bug!(ident.span, "unexpected definition for an \
                                                       identifier in pattern {:?}", def);
                            }
                        }
                    } else {
                        // Fall back to a fresh binding.
                        self.fresh_binding(ident, pat.id, outer_pat_id, pat_src, bindings)
                    };

                    self.record_def(pat.id, resolution);
                }

                PatKind::TupleStruct(ref path, _, _) => {
                    self.resolve_pattern_path(pat.id, None, path, ValueNS, |def| {
                        match def {
                            Def::Struct(..) | Def::Variant(..) | Def::Err => true,
                            _ => false,
                        }
                    }, "variant or struct");
                }

                PatKind::Path(ref qself, ref path) => {
                    self.resolve_pattern_path(pat.id, qself.as_ref(), path, ValueNS, |def| {
                        match def {
                            Def::Struct(..) | Def::Variant(..) |
                            Def::Const(..) | Def::AssociatedConst(..) | Def::Err => true,
                            _ => false,
                        }
                    }, "variant, struct or constant");
                }

                PatKind::Struct(ref path, _, _) => {
                    self.resolve_pattern_path(pat.id, None, path, TypeNS, |def| {
                        match def {
                            Def::Struct(..) | Def::Variant(..) |
                            Def::TyAlias(..) | Def::AssociatedTy(..) | Def::Err => true,
                            _ => false,
                        }
                    }, "variant, struct or type alias");
                }

                _ => {}
            }
            true
        });

        visit::walk_pat(self, pat);
    }

    /// Handles paths that may refer to associated items
    fn resolve_possibly_assoc_item(&mut self,
                                   id: NodeId,
                                   maybe_qself: Option<&QSelf>,
                                   path: &Path,
                                   namespace: Namespace)
                                   -> Option<PathResolution> {
        let max_assoc_types;

        match maybe_qself {
            Some(qself) => {
                if qself.position == 0 {
                    // FIXME: Create some fake resolution that can't possibly be a type.
                    return Some(PathResolution {
                        base_def: Def::Mod(self.definitions.local_def_id(ast::CRATE_NODE_ID)),
                        depth: path.segments.len(),
                    });
                }
                max_assoc_types = path.segments.len() - qself.position;
                // Make sure the trait is valid.
                let _ = self.resolve_trait_reference(id, path, max_assoc_types);
            }
            None => {
                max_assoc_types = path.segments.len();
            }
        }

        let mut resolution = self.with_no_errors(|this| {
            this.resolve_path(id, path, 0, namespace).ok()
        });
        for depth in 1..max_assoc_types {
            if resolution.is_some() {
                break;
            }
            self.with_no_errors(|this| {
                let partial_resolution = this.resolve_path(id, path, depth, TypeNS).ok();
                if let Some(Def::Mod(..)) = partial_resolution.map(|r| r.base_def) {
                    // Modules cannot have associated items
                } else {
                    resolution = partial_resolution;
                }
            });
        }
        resolution
    }

    /// Skips `path_depth` trailing segments, which is also reflected in the
    /// returned value. See `hir::def::PathResolution` for more info.
    fn resolve_path(&mut self, id: NodeId, path: &Path, path_depth: usize, namespace: Namespace)
                    -> Result<PathResolution, bool /* true if an error was reported */ > {
        debug!("resolve_path(id={:?} path={:?}, path_depth={:?})", id, path, path_depth);

        let span = path.span;
        let segments = &path.segments[..path.segments.len() - path_depth];

        let mk_res = |def| PathResolution { base_def: def, depth: path_depth };

        if path.global {
            let binding = self.resolve_crate_relative_path(span, segments, namespace);
            return binding.map(|binding| mk_res(binding.def().unwrap()));
        }

        // Try to find a path to an item in a module.
        let last_ident = segments.last().unwrap().identifier;
        // Resolve a single identifier with fallback to primitive types
        let resolve_identifier_with_fallback = |this: &mut Self, record_used| {
            let def = this.resolve_identifier(last_ident, namespace, record_used);
            match def {
                None | Some(LocalDef{def: Def::Mod(..), ..}) if namespace == TypeNS =>
                    this.primitive_type_table
                        .primitive_types
                        .get(&last_ident.name)
                        .map_or(def, |prim_ty| Some(LocalDef::from_def(Def::PrimTy(*prim_ty)))),
                _ => def
            }
        };

        if segments.len() == 1 {
            // In `a(::assoc_item)*` `a` cannot be a module. If `a` does resolve to a module we
            // don't report an error right away, but try to fallback to a primitive type.
            // So, we are still able to successfully resolve something like
            //
            // use std::u8; // bring module u8 in scope
            // fn f() -> u8 { // OK, resolves to primitive u8, not to std::u8
            //     u8::max_value() // OK, resolves to associated function <u8>::max_value,
            //                     // not to non-existent std::u8::max_value
            // }
            //
            // Such behavior is required for backward compatibility.
            // The same fallback is used when `a` resolves to nothing.
            let def = resolve_identifier_with_fallback(self, true).ok_or(false);
            return def.and_then(|def| self.adjust_local_def(def, span).ok_or(true)).map(mk_res);
        }

        let unqualified_def = resolve_identifier_with_fallback(self, false);
        let qualified_binding = self.resolve_module_relative_path(span, segments, namespace);
        match (qualified_binding, unqualified_def) {
            (Ok(binding), Some(ref ud)) if binding.def().unwrap() == ud.def => {
                self.session
                    .add_lint(lint::builtin::UNUSED_QUALIFICATIONS,
                              id,
                              span,
                              "unnecessary qualification".to_string());
            }
            _ => {}
        }

        qualified_binding.map(|binding| mk_res(binding.def().unwrap()))
    }

    // Resolve a single identifier
    fn resolve_identifier(&mut self,
                          identifier: ast::Ident,
                          namespace: Namespace,
                          record_used: bool)
                          -> Option<LocalDef> {
        if identifier.name == keywords::Invalid.name() {
            return Some(LocalDef::from_def(Def::Err));
        }

        self.resolve_ident_in_lexical_scope(identifier, namespace, record_used)
            .map(LexicalScopeBinding::local_def)
    }

    // Resolve a local definition, potentially adjusting for closures.
    fn adjust_local_def(&mut self, local_def: LocalDef, span: Span) -> Option<Def> {
        let ribs = match local_def.ribs {
            Some((TypeNS, i)) => &self.type_ribs[i + 1..],
            Some((ValueNS, i)) => &self.value_ribs[i + 1..],
            _ => &[] as &[_],
        };
        let mut def = local_def.def;
        match def {
            Def::Upvar(..) => {
                span_bug!(span, "unexpected {:?} in bindings", def)
            }
            Def::Local(_, node_id) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind | ModuleRibKind(..) => {
                            // Nothing to do. Continue.
                        }
                        ClosureRibKind(function_id) => {
                            let prev_def = def;
                            let node_def_id = self.definitions.local_def_id(node_id);

                            let seen = self.freevars_seen
                                           .entry(function_id)
                                           .or_insert_with(|| NodeMap());
                            if let Some(&index) = seen.get(&node_id) {
                                def = Def::Upvar(node_def_id, node_id, index, function_id);
                                continue;
                            }
                            let vec = self.freevars
                                          .entry(function_id)
                                          .or_insert_with(|| vec![]);
                            let depth = vec.len();
                            vec.push(Freevar {
                                def: prev_def,
                                span: span,
                            });

                            def = Def::Upvar(node_def_id, node_id, depth, function_id);
                            seen.insert(node_id, depth);
                        }
                        ItemRibKind | MethodRibKind(_) => {
                            // This was an attempt to access an upvar inside a
                            // named function item. This is not allowed, so we
                            // report an error.
                            resolve_error(self,
                                          span,
                                          ResolutionError::CannotCaptureDynamicEnvironmentInFnItem);
                            return None;
                        }
                        ConstantItemRibKind => {
                            // Still doesn't deal with upvars
                            resolve_error(self,
                                          span,
                                          ResolutionError::AttemptToUseNonConstantValueInConstant);
                            return None;
                        }
                    }
                }
            }
            Def::TyParam(..) | Def::SelfTy(..) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind | MethodRibKind(_) | ClosureRibKind(..) |
                        ModuleRibKind(..) => {
                            // Nothing to do. Continue.
                        }
                        ItemRibKind => {
                            // This was an attempt to use a type parameter outside
                            // its scope.

                            resolve_error(self,
                                          span,
                                          ResolutionError::TypeParametersFromOuterFunction);
                            return None;
                        }
                        ConstantItemRibKind => {
                            // see #9186
                            resolve_error(self, span, ResolutionError::OuterTypeParameterContext);
                            return None;
                        }
                    }
                }
            }
            _ => {}
        }
        return Some(def);
    }

    // resolve a "module-relative" path, e.g. a::b::c
    fn resolve_module_relative_path(&mut self,
                                    span: Span,
                                    segments: &[ast::PathSegment],
                                    namespace: Namespace)
                                    -> Result<&'a NameBinding<'a>,
                                              bool /* true if an error was reported */> {
        let module_path = segments.split_last()
                                  .unwrap()
                                  .1
                                  .iter()
                                  .map(|ps| ps.identifier.name)
                                  .collect::<Vec<_>>();

        let containing_module;
        match self.resolve_module_path(&module_path, UseLexicalScope, span) {
            Failed(err) => {
                let (span, msg) = match err {
                    Some((span, msg)) => (span, msg),
                    None => {
                        let msg = format!("Use of undeclared type or module `{}`",
                                          names_to_string(&module_path));
                        (span, msg)
                    }
                };

                resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                return Err(true);
            }
            Indeterminate => return Err(false),
            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = segments.last().unwrap().identifier.name;
        let result = self.resolve_name_in_module(containing_module, name, namespace, false, true);
        result.success().map(|binding| {
            self.check_privacy(name, binding, span);
            binding
        }).ok_or(false)
    }

    /// Invariant: This must be called only during main resolution, not during
    /// import resolution.
    fn resolve_crate_relative_path<T>(&mut self, span: Span, segments: &[T], namespace: Namespace)
                                      -> Result<&'a NameBinding<'a>,
                                                bool /* true if an error was reported */>
        where T: Named,
    {
        let module_path = segments.split_last().unwrap().1.iter().map(T::name).collect::<Vec<_>>();
        let root_module = self.graph_root;

        let containing_module;
        match self.resolve_module_path_from_root(root_module,
                                                 &module_path,
                                                 0,
                                                 span) {
            Failed(err) => {
                let (span, msg) = match err {
                    Some((span, msg)) => (span, msg),
                    None => {
                        let msg = format!("Use of undeclared module `::{}`",
                                          names_to_string(&module_path));
                        (span, msg)
                    }
                };

                resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                return Err(true);
            }

            Indeterminate => return Err(false),

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = segments.last().unwrap().name();
        let result = self.resolve_name_in_module(containing_module, name, namespace, false, true);
        result.success().map(|binding| {
            self.check_privacy(name, binding, span);
            binding
        }).ok_or(false)
    }

    fn with_no_errors<T, F>(&mut self, f: F) -> T
        where F: FnOnce(&mut Resolver) -> T
    {
        self.emit_errors = false;
        let rs = f(self);
        self.emit_errors = true;
        rs
    }

    fn find_fallback_in_self_type(&mut self, name: Name) -> FallbackSuggestion {
        fn extract_node_id(t: &Ty) -> Option<NodeId> {
            match t.node {
                TyKind::Path(None, _) => Some(t.id),
                TyKind::Rptr(_, ref mut_ty) => extract_node_id(&mut_ty.ty),
                // This doesn't handle the remaining `Ty` variants as they are not
                // that commonly the self_type, it might be interesting to provide
                // support for those in future.
                _ => None,
            }
        }

        if let Some(node_id) = self.current_self_type.as_ref().and_then(extract_node_id) {
            // Look for a field with the same name in the current self_type.
            if let Some(resolution) = self.def_map.get(&node_id) {
                match resolution.base_def {
                    Def::Enum(did) | Def::TyAlias(did) |
                    Def::Struct(did) | Def::Variant(_, did) if resolution.depth == 0 => {
                        if let Some(fields) = self.structs.get(&did) {
                            if fields.iter().any(|&field_name| name == field_name) {
                                return Field;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Look for a method in the current trait.
        if let Some((trait_did, ref trait_ref)) = self.current_trait_ref {
            if let Some(&is_static_method) = self.trait_item_map.get(&(name, trait_did)) {
                if is_static_method {
                    return TraitMethod(path_names_to_string(&trait_ref.path, 0));
                } else {
                    return TraitItem;
                }
            }
        }

        NoSuggestion
    }

    fn find_best_match(&mut self, name: &str) -> SuggestionType {
        if let Some(macro_name) = self.session.available_macros
                                  .borrow().iter().find(|n| n.as_str() == name) {
            return SuggestionType::Macro(format!("{}!", macro_name));
        }

        let names = self.value_ribs
                    .iter()
                    .rev()
                    .flat_map(|rib| rib.bindings.keys());

        if let Some(found) = find_best_match_for_name(names, name, None) {
            if name != found {
                return SuggestionType::Function(found);
            }
        } SuggestionType::NotFound
    }

    fn resolve_labeled_block(&mut self, label: Option<ast::Ident>, id: NodeId, block: &Block) {
        if let Some(label) = label {
            let (label, def) = (mtwt::resolve(label), Def::Label(id));
            self.with_label_rib(|this| {
                this.label_ribs.last_mut().unwrap().bindings.insert(label, def);
                this.visit_block(block);
            });
        } else {
            self.visit_block(block);
        }
    }

    fn resolve_expr(&mut self, expr: &Expr, parent: Option<&Expr>) {
        // First, record candidate traits for this expression if it could
        // result in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.node {
            ExprKind::Path(ref maybe_qself, ref path) => {
                // This is a local path in the value namespace. Walk through
                // scopes looking for it.
                if let Some(path_res) = self.resolve_possibly_assoc_item(expr.id,
                                                            maybe_qself.as_ref(), path, ValueNS) {
                    // Check if struct variant
                    let is_struct_variant = if let Def::Variant(_, variant_id) = path_res.base_def {
                        self.structs.contains_key(&variant_id)
                    } else {
                        false
                    };
                    if is_struct_variant {
                        let _ = self.structs.contains_key(&path_res.base_def.def_id());
                        let path_name = path_names_to_string(path, 0);

                        let mut err = resolve_struct_error(self,
                                        expr.span,
                                        ResolutionError::StructVariantUsedAsFunction(&path_name));

                        let msg = format!("did you mean to write: `{} {{ /* fields */ }}`?",
                                          path_name);
                        if self.emit_errors {
                            err.help(&msg);
                        } else {
                            err.span_help(expr.span, &msg);
                        }
                        err.emit();
                        self.record_def(expr.id, err_path_resolution());
                    } else {
                        // Write the result into the def map.
                        debug!("(resolving expr) resolved `{}`",
                               path_names_to_string(path, 0));

                        // Partial resolutions will need the set of traits in scope,
                        // so they can be completed during typeck.
                        if path_res.depth != 0 {
                            let method_name = path.segments.last().unwrap().identifier.name;
                            let traits = self.get_traits_containing_item(method_name);
                            self.trait_map.insert(expr.id, traits);
                        }

                        self.record_def(expr.id, path_res);
                    }
                } else {
                    // Be helpful if the name refers to a struct
                    // (The pattern matching def_tys where the id is in self.structs
                    // matches on regular structs while excluding tuple- and enum-like
                    // structs, which wouldn't result in this error.)
                    let path_name = path_names_to_string(path, 0);
                    let type_res = self.with_no_errors(|this| {
                        this.resolve_path(expr.id, path, 0, TypeNS)
                    });

                    self.record_def(expr.id, err_path_resolution());

                    if let Ok(Def::Struct(..)) = type_res.map(|r| r.base_def) {
                        let error_variant =
                            ResolutionError::StructVariantUsedAsFunction(&path_name);
                        let mut err = resolve_struct_error(self, expr.span, error_variant);

                        let msg = format!("did you mean to write: `{} {{ /* fields */ }}`?",
                                          path_name);

                        if self.emit_errors {
                            err.help(&msg);
                        } else {
                            err.span_help(expr.span, &msg);
                        }
                        err.emit();
                    } else {
                        // Keep reporting some errors even if they're ignored above.
                        if let Err(true) = self.resolve_path(expr.id, path, 0, ValueNS) {
                            // `resolve_path` already reported the error
                        } else {
                            let mut method_scope = false;
                            let mut is_static = false;
                            self.value_ribs.iter().rev().all(|rib| {
                                method_scope = match rib.kind {
                                    MethodRibKind(is_static_) => {
                                        is_static = is_static_;
                                        true
                                    }
                                    ItemRibKind | ConstantItemRibKind => false,
                                    _ => return true, // Keep advancing
                                };
                                false // Stop advancing
                            });

                            if method_scope &&
                                    &path_name[..] == keywords::SelfValue.name().as_str() {
                                resolve_error(self,
                                              expr.span,
                                              ResolutionError::SelfNotAvailableInStaticMethod);
                            } else {
                                let last_name = path.segments.last().unwrap().identifier.name;
                                let (mut msg, is_field) =
                                    match self.find_fallback_in_self_type(last_name) {
                                    NoSuggestion => {
                                        // limit search to 5 to reduce the number
                                        // of stupid suggestions
                                        (match self.find_best_match(&path_name) {
                                            SuggestionType::Macro(s) => {
                                                format!("the macro `{}`", s)
                                            }
                                            SuggestionType::Function(s) => format!("`{}`", s),
                                            SuggestionType::NotFound => "".to_string(),
                                        }, false)
                                    }
                                    Field => {
                                        (if is_static && method_scope {
                                            "".to_string()
                                        } else {
                                            format!("`self.{}`", path_name)
                                        }, true)
                                    }
                                    TraitItem => (format!("to call `self.{}`", path_name), false),
                                    TraitMethod(path_str) =>
                                        (format!("to call `{}::{}`", path_str, path_name), false),
                                };

                                let mut context =  UnresolvedNameContext::Other;
                                let mut def = Def::Err;
                                if !msg.is_empty() {
                                    msg = format!(". Did you mean {}?", msg);
                                } else {
                                    // we check if this a module and if so, we display a help
                                    // message
                                    let name_path = path.segments.iter()
                                                        .map(|seg| seg.identifier.name)
                                                        .collect::<Vec<_>>();

                                    match self.resolve_module_path(&name_path[..],
                                                                   UseLexicalScope,
                                                                   expr.span) {
                                        Success(e) => {
                                            if let Some(def_type) = e.def {
                                                def = def_type;
                                            }
                                            context = UnresolvedNameContext::PathIsMod(parent);
                                        },
                                        _ => {},
                                    };
                                }

                                resolve_error(self,
                                              expr.span,
                                              ResolutionError::UnresolvedName {
                                                  path: &path_name,
                                                  message: &msg,
                                                  context: context,
                                                  is_static_method: method_scope && is_static,
                                                  is_field: is_field,
                                                  def: def,
                                              });
                            }
                        }
                    }
                }

                visit::walk_expr(self, expr);
            }

            ExprKind::Struct(ref path, _, _) => {
                // Resolve the path to the structure it goes to. We don't
                // check to ensure that the path is actually a structure; that
                // is checked later during typeck.
                match self.resolve_path(expr.id, path, 0, TypeNS) {
                    Ok(definition) => self.record_def(expr.id, definition),
                    Err(true) => self.record_def(expr.id, err_path_resolution()),
                    Err(false) => {
                        debug!("(resolving expression) didn't find struct def",);

                        resolve_error(self,
                                      path.span,
                                      ResolutionError::DoesNotNameAStruct(
                                                                &path_names_to_string(path, 0))
                                     );
                        self.record_def(expr.id, err_path_resolution());
                    }
                }

                visit::walk_expr(self, expr);
            }

            ExprKind::Loop(_, Some(label)) | ExprKind::While(_, _, Some(label)) => {
                self.with_label_rib(|this| {
                    let def = Def::Label(expr.id);

                    {
                        let rib = this.label_ribs.last_mut().unwrap();
                        rib.bindings.insert(mtwt::resolve(label.node), def);
                    }

                    visit::walk_expr(this, expr);
                })
            }

            ExprKind::Break(Some(label)) | ExprKind::Continue(Some(label)) => {
                match self.search_label(mtwt::resolve(label.node)) {
                    None => {
                        self.record_def(expr.id, err_path_resolution());
                        resolve_error(self,
                                      label.span,
                                      ResolutionError::UndeclaredLabel(&label.node.name.as_str()))
                    }
                    Some(def @ Def::Label(_)) => {
                        // Since this def is a label, it is never read.
                        self.record_def(expr.id, PathResolution::new(def))
                    }
                    Some(_) => {
                        span_bug!(expr.span, "label wasn't mapped to a label def!")
                    }
                }
            }

            ExprKind::IfLet(ref pattern, ref subexpression, ref if_block, ref optional_else) => {
                self.visit_expr(subexpression);

                self.value_ribs.push(Rib::new(NormalRibKind));
                self.resolve_pattern(pattern, PatternSource::IfLet, &mut HashMap::new());
                self.visit_block(if_block);
                self.value_ribs.pop();

                optional_else.as_ref().map(|expr| self.visit_expr(expr));
            }

            ExprKind::WhileLet(ref pattern, ref subexpression, ref block, label) => {
                self.visit_expr(subexpression);
                self.value_ribs.push(Rib::new(NormalRibKind));
                self.resolve_pattern(pattern, PatternSource::WhileLet, &mut HashMap::new());

                self.resolve_labeled_block(label.map(|l| l.node), expr.id, block);

                self.value_ribs.pop();
            }

            ExprKind::ForLoop(ref pattern, ref subexpression, ref block, label) => {
                self.visit_expr(subexpression);
                self.value_ribs.push(Rib::new(NormalRibKind));
                self.resolve_pattern(pattern, PatternSource::For, &mut HashMap::new());

                self.resolve_labeled_block(label.map(|l| l.node), expr.id, block);

                self.value_ribs.pop();
            }

            ExprKind::Field(ref subexpression, _) => {
                self.resolve_expr(subexpression, Some(expr));
            }
            ExprKind::MethodCall(_, ref types, ref arguments) => {
                let mut arguments = arguments.iter();
                self.resolve_expr(arguments.next().unwrap(), Some(expr));
                for argument in arguments {
                    self.resolve_expr(argument, None);
                }
                for ty in types.iter() {
                    self.visit_ty(ty);
                }
            }

            _ => {
                visit::walk_expr(self, expr);
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self, expr: &Expr) {
        match expr.node {
            ExprKind::Field(_, name) => {
                // FIXME(#6890): Even though you can't treat a method like a
                // field, we need to add any trait methods we find that match
                // the field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.get_traits_containing_item(name.node.name);
                self.trait_map.insert(expr.id, traits);
            }
            ExprKind::MethodCall(name, _, _) => {
                debug!("(recording candidate traits for expr) recording traits for {}",
                       expr.id);
                let traits = self.get_traits_containing_item(name.node.name);
                self.trait_map.insert(expr.id, traits);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn get_traits_containing_item(&mut self, name: Name) -> Vec<TraitCandidate> {
        debug!("(getting traits containing item) looking for '{}'", name);

        fn add_trait_info(found_traits: &mut Vec<TraitCandidate>,
                          trait_def_id: DefId,
                          import_id: Option<NodeId>,
                          name: Name) {
            debug!("(adding trait info) found trait {:?} for method '{}'",
                   trait_def_id,
                   name);
            found_traits.push(TraitCandidate {
                def_id: trait_def_id,
                import_id: import_id,
            });
        }

        let mut found_traits = Vec::new();
        // Look for the current trait.
        if let Some((trait_def_id, _)) = self.current_trait_ref {
            if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                add_trait_info(&mut found_traits, trait_def_id, None, name);
            }
        }

        let mut search_module = self.current_module;
        loop {
            // Look for trait children.
            let mut search_in_module = |this: &mut Self, module: Module<'a>| {
                let mut traits = module.traits.borrow_mut();
                if traits.is_none() {
                    let mut collected_traits = Vec::new();
                    module.for_each_child(|name, ns, binding| {
                        if ns != TypeNS { return }
                        if let Some(Def::Trait(_)) = binding.def() {
                            collected_traits.push((name, binding));
                        }
                    });
                    *traits = Some(collected_traits.into_boxed_slice());
                }

                for &(trait_name, binding) in traits.as_ref().unwrap().iter() {
                    let trait_def_id = binding.def().unwrap().def_id();
                    if this.trait_item_map.contains_key(&(name, trait_def_id)) {
                        let mut import_id = None;
                        if let NameBindingKind::Import { directive, .. } = binding.kind {
                            let id = directive.id;
                            this.maybe_unused_trait_imports.insert(id);
                            import_id = Some(id);
                        }
                        add_trait_info(&mut found_traits, trait_def_id, import_id, name);
                        this.record_use(trait_name, binding);
                    }
                }
            };
            search_in_module(self, search_module);

            match search_module.parent_link {
                NoParentLink | ModuleParentLink(..) => {
                    if !search_module.no_implicit_prelude.get() {
                        self.prelude.map(|prelude| search_in_module(self, prelude));
                    }
                    break;
                }
                BlockParentLink(parent_module, _) => {
                    search_module = parent_module;
                }
            }
        }

        found_traits
    }

    /// When name resolution fails, this method can be used to look up candidate
    /// entities with the expected name. It allows filtering them using the
    /// supplied predicate (which should be used to only accept the types of
    /// definitions expected e.g. traits). The lookup spans across all crates.
    ///
    /// NOTE: The method does not look into imports, but this is not a problem,
    /// since we report the definitions (thus, the de-aliased imports).
    fn lookup_candidates<FilterFn>(&mut self,
                                   lookup_name: Name,
                                   namespace: Namespace,
                                   filter_fn: FilterFn) -> SuggestedCandidates
        where FilterFn: Fn(Def) -> bool {

        let mut lookup_results = Vec::new();
        let mut worklist = Vec::new();
        worklist.push((self.graph_root, Vec::new(), false));

        while let Some((in_module,
                        path_segments,
                        in_module_is_extern)) = worklist.pop() {
            self.populate_module_if_necessary(in_module);

            in_module.for_each_child(|name, ns, name_binding| {

                // avoid imports entirely
                if name_binding.is_import() { return; }

                // collect results based on the filter function
                if let Some(def) = name_binding.def() {
                    if name == lookup_name && ns == namespace && filter_fn(def) {
                        // create the path
                        let ident = ast::Ident::with_empty_ctxt(name);
                        let params = PathParameters::none();
                        let segment = PathSegment {
                            identifier: ident,
                            parameters: params,
                        };
                        let span = name_binding.span;
                        let mut segms = path_segments.clone();
                        segms.push(segment);
                        let path = Path {
                            span: span,
                            global: true,
                            segments: segms,
                        };
                        // the entity is accessible in the following cases:
                        // 1. if it's defined in the same crate, it's always
                        // accessible (since private entities can be made public)
                        // 2. if it's defined in another crate, it's accessible
                        // only if both the module is public and the entity is
                        // declared as public (due to pruning, we don't explore
                        // outside crate private modules => no need to check this)
                        if !in_module_is_extern || name_binding.vis == ty::Visibility::Public {
                            lookup_results.push(path);
                        }
                    }
                }

                // collect submodules to explore
                if let Some(module) = name_binding.module() {
                    // form the path
                    let path_segments = match module.parent_link {
                        NoParentLink => path_segments.clone(),
                        ModuleParentLink(_, name) => {
                            let mut paths = path_segments.clone();
                            let ident = ast::Ident::with_empty_ctxt(name);
                            let params = PathParameters::none();
                            let segm = PathSegment {
                                identifier: ident,
                                parameters: params,
                            };
                            paths.push(segm);
                            paths
                        }
                        _ => bug!(),
                    };

                    if !in_module_is_extern || name_binding.vis == ty::Visibility::Public {
                        // add the module to the lookup
                        let is_extern = in_module_is_extern || name_binding.is_extern_crate();
                        worklist.push((module, path_segments, is_extern));
                    }
                }
            })
        }

        SuggestedCandidates {
            name: lookup_name.as_str().to_string(),
            candidates: lookup_results,
        }
    }

    fn record_def(&mut self, node_id: NodeId, resolution: PathResolution) {
        debug!("(recording def) recording {:?} for {}", resolution, node_id);
        if let Some(prev_res) = self.def_map.insert(node_id, resolution) {
            panic!("path resolved multiple times ({:?} before, {:?} now)", prev_res, resolution);
        }
    }

    fn resolve_visibility(&mut self, vis: &ast::Visibility) -> ty::Visibility {
        let (path, id) = match *vis {
            ast::Visibility::Public => return ty::Visibility::Public,
            ast::Visibility::Crate(_) => return ty::Visibility::Restricted(ast::CRATE_NODE_ID),
            ast::Visibility::Restricted { ref path, id } => (path, id),
            ast::Visibility::Inherited => {
                let current_module =
                    self.get_nearest_normal_module_parent_or_self(self.current_module);
                let id =
                    self.definitions.as_local_node_id(current_module.def_id().unwrap()).unwrap();
                return ty::Visibility::Restricted(id);
            }
        };

        let segments: Vec<_> = path.segments.iter().map(|seg| seg.identifier.name).collect();
        let mut path_resolution = err_path_resolution();
        let vis = match self.resolve_module_path(&segments, DontUseLexicalScope, path.span) {
            Success(module) => {
                let def = module.def.unwrap();
                path_resolution = PathResolution::new(def);
                ty::Visibility::Restricted(self.definitions.as_local_node_id(def.def_id()).unwrap())
            }
            Failed(Some((span, msg))) => {
                self.session.span_err(span, &format!("failed to resolve module path. {}", msg));
                ty::Visibility::Public
            }
            _ => {
                self.session.span_err(path.span, "unresolved module path");
                ty::Visibility::Public
            }
        };
        self.def_map.insert(id, path_resolution);
        if !self.is_accessible(vis) {
            let msg = format!("visibilities can only be restricted to ancestor modules");
            self.session.span_err(path.span, &msg);
        }
        vis
    }

    fn is_accessible(&self, vis: ty::Visibility) -> bool {
        let current_module = self.get_nearest_normal_module_parent_or_self(self.current_module);
        let node_id = self.definitions.as_local_node_id(current_module.def_id().unwrap()).unwrap();
        vis.is_accessible_from(node_id, self)
    }

    fn check_privacy(&mut self, name: Name, binding: &'a NameBinding<'a>, span: Span) {
        if !self.is_accessible(binding.vis) {
            self.privacy_errors.push(PrivacyError(span, name, binding));
        }
    }

    fn report_privacy_errors(&self) {
        if self.privacy_errors.len() == 0 { return }
        let mut reported_spans = HashSet::new();
        for &PrivacyError(span, name, binding) in &self.privacy_errors {
            if !reported_spans.insert(span) { continue }
            if binding.is_extern_crate() {
                // Warn when using an inaccessible extern crate.
                let node_id = binding.module().unwrap().extern_crate_id.unwrap();
                let msg = format!("extern crate `{}` is private", name);
                self.session.add_lint(lint::builtin::INACCESSIBLE_EXTERN_CRATE, node_id, span, msg);
            } else {
                let def = binding.def().unwrap();
                self.session.span_err(span, &format!("{} `{}` is private", def.kind_name(), name));
            }
        }
    }

    fn report_conflict(&self,
                       parent: Module,
                       name: Name,
                       ns: Namespace,
                       binding: &NameBinding,
                       old_binding: &NameBinding) {
        // Error on the second of two conflicting names
        if old_binding.span.lo > binding.span.lo {
            return self.report_conflict(parent, name, ns, old_binding, binding);
        }

        let container = match parent.def {
            Some(Def::Mod(_)) => "module",
            Some(Def::Trait(_)) => "trait",
            None => "block",
            _ => "enum",
        };

        let (participle, noun) = match old_binding.is_import() || old_binding.is_extern_crate() {
            true => ("imported", "import"),
            false => ("defined", "definition"),
        };

        let span = binding.span;
        let msg = {
            let kind = match (ns, old_binding.module()) {
                (ValueNS, _) => "a value",
                (TypeNS, Some(module)) if module.extern_crate_id.is_some() => "an extern crate",
                (TypeNS, Some(module)) if module.is_normal() => "a module",
                (TypeNS, Some(module)) if module.is_trait() => "a trait",
                (TypeNS, _) => "a type",
            };
            format!("{} named `{}` has already been {} in this {}",
                    kind, name, participle, container)
        };

        let mut err = match (old_binding.is_extern_crate(), binding.is_extern_crate()) {
            (true, true) => struct_span_err!(self.session, span, E0259, "{}", msg),
            (true, _) | (_, true) if binding.is_import() || old_binding.is_import() =>
                struct_span_err!(self.session, span, E0254, "{}", msg),
            (true, _) | (_, true) => struct_span_err!(self.session, span, E0260, "{}", msg),
            _ => match (old_binding.is_import(), binding.is_import()) {
                (false, false) => struct_span_err!(self.session, span, E0428, "{}", msg),
                (true, true) => struct_span_err!(self.session, span, E0252, "{}", msg),
                _ => {
                    let mut e = struct_span_err!(self.session, span, E0255, "{}", msg);
                    e.span_label(span, &format!("`{}` was already imported", name));
                    e
                }
            },
        };

        if old_binding.span != syntax_pos::DUMMY_SP {
            err.span_label(old_binding.span, &format!("previous {} of `{}` here", noun, name));
        }
        err.emit();
    }
}

fn names_to_string(names: &[Name]) -> String {
    let mut first = true;
    let mut result = String::new();
    for name in names {
        if first {
            first = false
        } else {
            result.push_str("::")
        }
        result.push_str(&name.as_str());
    }
    result
}

fn path_names_to_string(path: &Path, depth: usize) -> String {
    let names: Vec<ast::Name> = path.segments[..path.segments.len() - depth]
                                    .iter()
                                    .map(|seg| seg.identifier.name)
                                    .collect();
    names_to_string(&names[..])
}

/// When an entity with a given name is not available in scope, we search for
/// entities with that name in all crates. This method allows outputting the
/// results of this search in a programmer-friendly way
fn show_candidates(session: &mut DiagnosticBuilder,
                   candidates: &SuggestedCandidates) {

    let paths = &candidates.candidates;

    if paths.len() > 0 {
        // don't show more than MAX_CANDIDATES results, so
        // we're consistent with the trait suggestions
        const MAX_CANDIDATES: usize = 5;

        // we want consistent results across executions, but candidates are produced
        // by iterating through a hash map, so make sure they are ordered:
        let mut path_strings: Vec<_> = paths.into_iter()
                                            .map(|p| path_names_to_string(&p, 0))
                                            .collect();
        path_strings.sort();

        // behave differently based on how many candidates we have:
        if !paths.is_empty() {
            if paths.len() == 1 {
                session.help(
                    &format!("you can import it into scope: `use {};`.",
                        &path_strings[0]),
                );
            } else {
                session.help("you can import several candidates \
                    into scope (`use ...;`):");
                let count = path_strings.len() as isize - MAX_CANDIDATES as isize + 1;

                for (idx, path_string) in path_strings.iter().enumerate() {
                    if idx == MAX_CANDIDATES - 1 && count > 1 {
                        session.help(
                            &format!("  and {} other candidates", count).to_string(),
                        );
                        break;
                    } else {
                        session.help(
                            &format!("  `{}`", path_string).to_string(),
                        );
                    }
                }
            }
        }
    } else {
        // nothing found:
        session.help(
            &format!("no candidates by the name of `{}` found in your \
            project; maybe you misspelled the name or forgot to import \
            an external crate?", candidates.name.to_string()),
        );
    };
}

/// A somewhat inefficient routine to obtain the name of a module.
fn module_to_string(module: Module) -> String {
    let mut names = Vec::new();

    fn collect_mod(names: &mut Vec<ast::Name>, module: Module) {
        match module.parent_link {
            NoParentLink => {}
            ModuleParentLink(ref module, name) => {
                names.push(name);
                collect_mod(names, module);
            }
            BlockParentLink(ref module, _) => {
                // danger, shouldn't be ident?
                names.push(token::intern("<opaque>"));
                collect_mod(names, module);
            }
        }
    }
    collect_mod(&mut names, module);

    if names.is_empty() {
        return "???".to_string();
    }
    names_to_string(&names.into_iter().rev().collect::<Vec<ast::Name>>())
}

fn err_path_resolution() -> PathResolution {
    PathResolution::new(Def::Err)
}

#[derive(PartialEq,Copy, Clone)]
pub enum MakeGlobMap {
    Yes,
    No,
}

/// Entry point to crate resolution.
pub fn resolve_crate<'a, 'b>(resolver: &'b mut Resolver<'a>, krate: &'b Crate) {
    // Currently, we ignore the name resolution data structures for
    // the purposes of dependency tracking. Instead we will run name
    // resolution and include its output in the hash of each item,
    // much like we do for macro expansion. In other words, the hash
    // reflects not just its contents but the results of name
    // resolution on those contents. Hopefully we'll push this back at
    // some point.
    let _ignore = resolver.session.dep_graph.in_ignore();

    resolver.build_reduced_graph(krate);
    resolve_imports::resolve_imports(resolver);
    resolver.resolve_crate(krate);

    check_unused::check_crate(resolver, krate);
    resolver.report_privacy_errors();
}

pub fn with_resolver<'a, T, F>(session: &'a Session,
                               definitions: &'a mut Definitions,
                               make_glob_map: MakeGlobMap,
                               f: F) -> T
    where F: for<'b> FnOnce(Resolver<'b>) -> T,
{
    let arenas = Resolver::arenas();
    let resolver = Resolver::new(session, definitions, make_glob_map, &arenas);
    f(resolver)
}

__build_diagnostic_array! { librustc_resolve, DIAGNOSTICS }
