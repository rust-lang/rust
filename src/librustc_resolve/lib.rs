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
use self::FallbackSuggestion::*;
use self::TypeParameters::*;
use self::RibKind::*;

use rustc::hir::map::{Definitions, DefCollector};
use rustc::hir::{self, PrimTy, TyBool, TyChar, TyFloat, TyInt, TyUint, TyStr};
use rustc::middle::cstore::CrateLoader;
use rustc::session::Session;
use rustc::lint;
use rustc::hir::def::*;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, LOCAL_CRATE, DefId};
use rustc::ty;
use rustc::hir::{Freevar, FreevarMap, TraitCandidate, TraitMap, GlobMap};
use rustc::util::nodemap::{NodeMap, NodeSet, FxHashMap, FxHashSet};

use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::ast::{self, Name, NodeId, Ident, SpannedIdent, FloatTy, IntTy, UintTy};
use syntax::ext::base::SyntaxExtension;
use syntax::ext::base::Determinacy::{Determined, Undetermined};
use syntax::symbol::{Symbol, keywords};
use syntax::util::lev_distance::find_best_match_for_name;

use syntax::visit::{self, FnKind, Visitor};
use syntax::attr;
use syntax::ast::{Arm, BindingMode, Block, Crate, Expr, ExprKind};
use syntax::ast::{FnDecl, ForeignItem, ForeignItemKind, Generics};
use syntax::ast::{Item, ItemKind, ImplItem, ImplItemKind};
use syntax::ast::{Local, Mutability, Pat, PatKind, Path};
use syntax::ast::{QSelf, TraitItemKind, TraitRef, Ty, TyKind};

use syntax_pos::{Span, DUMMY_SP, MultiSpan};
use errors::DiagnosticBuilder;

use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem::replace;
use std::rc::Rc;

use resolve_imports::{ImportDirective, ImportDirectiveSubclass, NameResolution, ImportResolver};
use macros::{InvocationData, LegacyBinding, LegacyScope};

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
mod diagnostics;

mod macros;
mod check_unused;
mod build_reduced_graph;
mod resolve_imports;

enum SuggestionType {
    Macro(String),
    Function(Symbol),
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
    NameAlreadyUsedInTypeParameterList(Name, &'a Span),
    /// error E0404: is not a trait
    IsNotATrait(&'a str, &'a str),
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
    VariableBoundWithDifferentMode(Name, usize, Span),
    /// error E0411: use of `Self` outside of an impl or trait
    SelfUsedOutsideImplOrTrait,
    /// error E0412: use of undeclared
    UseOfUndeclared(&'a str, &'a str, SuggestedCandidates),
    /// error E0415: identifier is bound more than once in this parameter list
    IdentifierBoundMoreThanOnceInParameterList(&'a str),
    /// error E0416: identifier is bound more than once in the same pattern
    IdentifierBoundMoreThanOnceInSamePattern(&'a str),
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
    BindingShadowsSomethingUnacceptable(&'a str, Name, &'a NameBinding<'a>),
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
        ResolutionError::NameAlreadyUsedInTypeParameterList(name, first_use_span) => {
             let mut err = struct_span_err!(resolver.session,
                                            span,
                                            E0403,
                                            "the name `{}` is already used for a type parameter \
                                            in this type parameter list",
                                            name);
             err.span_label(span, &format!("already used"));
             err.span_label(first_use_span.clone(), &format!("first use of `{}`", name));
             err

        }
        ResolutionError::IsNotATrait(name, kind_name) => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0404,
                                           "`{}` is not a trait",
                                           name);
            err.span_label(span, &format!("expected trait, found {}", kind_name));
            err
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
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0407,
                                           "method `{}` is not a member of trait `{}`",
                                           method,
                                           trait_);
            err.span_label(span, &format!("not a member of trait `{}`", trait_));
            err
        }
        ResolutionError::TypeNotMemberOfTrait(type_, trait_) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0437,
                             "type `{}` is not a member of trait `{}`",
                             type_,
                             trait_);
            err.span_label(span, &format!("not a member of trait `{}`", trait_));
            err
        }
        ResolutionError::ConstNotMemberOfTrait(const_, trait_) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0438,
                             "const `{}` is not a member of trait `{}`",
                             const_,
                             trait_);
            err.span_label(span, &format!("not a member of trait `{}`", trait_));
            err
        }
        ResolutionError::VariableNotBoundInPattern(variable_name, from, to) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0408,
                             "variable `{}` from pattern #{} is not bound in pattern #{}",
                             variable_name,
                             from,
                             to);
            err.span_label(span, &format!("pattern doesn't bind `{}`", variable_name));
            err
        }
        ResolutionError::VariableBoundWithDifferentMode(variable_name,
                                                        pattern_number,
                                                        first_binding_span) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0409,
                             "variable `{}` is bound with different mode in pattern #{} than in \
                              pattern #1",
                             variable_name,
                             pattern_number);
            err.span_label(span, &format!("bound in different ways"));
            err.span_label(first_binding_span, &format!("first binding"));
            err
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
        ResolutionError::StructVariantUsedAsFunction(path_name) => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0423,
                             "`{}` is the name of a struct or struct variant, but this expression \
                             uses it like a function name",
                             path_name);
            err.span_label(span, &format!("struct called like a function"));
            err
        }
        ResolutionError::SelfNotAvailableInStaticMethod => {
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0424,
                             "`self` is not available in a static method");
            err.span_label(span, &format!("not available in static method"));
            err.note(&format!("maybe a `self` argument is missing?"));
            err
        }
        ResolutionError::UnresolvedName { path, message: msg, context, is_static_method,
                                          is_field, def } => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0425,
                                           "unresolved name `{}`",
                                           path);
            if msg != "" {
                err.span_label(span, &msg);
            } else {
                err.span_label(span, &format!("unresolved name"));
            }

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
                        Some(&ExprKind::MethodCall(ident, ..)) => {
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
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0426,
                                           "use of undeclared label `{}`",
                                           name);
            err.span_label(span, &format!("undeclared label `{}`",&name));
            err
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
                Some((n, _)) => format!("unresolved import `{}`", n),
                None => "unresolved import".to_owned(),
            };
            let mut err = struct_span_err!(resolver.session, span, E0432, "{}", msg);
            if let Some((_, p)) = name {
                err.span_label(span, &p);
            }
            err
        }
        ResolutionError::FailedToResolve(msg) => {
            let mut err = struct_span_err!(resolver.session, span, E0433,
                                           "failed to resolve. {}", msg);
            err.span_label(span, &msg);
            err
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
            let mut err = struct_span_err!(resolver.session,
                             span,
                             E0435,
                             "attempt to use a non-constant value in a constant");
            err.span_label(span, &format!("non-constant used with constant"));
            err
        }
        ResolutionError::BindingShadowsSomethingUnacceptable(what_binding, name, binding) => {
            let shadows_what = PathResolution::new(binding.def()).kind_name();
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0530,
                                           "{}s cannot shadow {}s", what_binding, shadows_what);
            err.span_label(span, &format!("cannot be named the same as a {}", shadows_what));
            let participle = if binding.is_import() { "imported" } else { "defined" };
            let msg = &format!("a {} `{}` is {} here", shadows_what, name, participle);
            err.span_label(binding.span, msg);
            err
        }
        ResolutionError::PatPathUnresolved(expected_what, path) => {
            struct_span_err!(resolver.session,
                             span,
                             E0531,
                             "unresolved {} `{}`",
                             expected_what,
                             path)
        }
        ResolutionError::PatPathUnexpected(expected_what, found_what, path) => {
            struct_span_err!(resolver.session,
                             span,
                             E0532,
                             "expected {}, found {} `{}`",
                             expected_what,
                             found_what,
                             path)
        }
    }
}

#[derive(Copy, Clone)]
struct BindingInfo {
    span: Span,
    binding_mode: BindingMode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = FxHashMap<Ident, BindingInfo>;

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
    MacroNS,
}

#[derive(Clone, Default, Debug)]
pub struct PerNS<T> {
    value_ns: T,
    type_ns: T,
    macro_ns: Option<T>,
}

impl<T> ::std::ops::Index<Namespace> for PerNS<T> {
    type Output = T;
    fn index(&self, ns: Namespace) -> &T {
        match ns {
            ValueNS => &self.value_ns,
            TypeNS => &self.type_ns,
            MacroNS => self.macro_ns.as_ref().unwrap(),
        }
    }
}

impl<T> ::std::ops::IndexMut<Namespace> for PerNS<T> {
    fn index_mut(&mut self, ns: Namespace) -> &mut T {
        match ns {
            ValueNS => &mut self.value_ns,
            TypeNS => &mut self.type_ns,
            MacroNS => self.macro_ns.as_mut().unwrap(),
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Resolver<'a> {
    fn visit_item(&mut self, item: &'tcx Item) {
        self.resolve_item(item);
    }
    fn visit_arm(&mut self, arm: &'tcx Arm) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &'tcx Block) {
        self.resolve_block(block);
    }
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        self.resolve_expr(expr, None);
    }
    fn visit_local(&mut self, local: &'tcx Local) {
        self.resolve_local(local);
    }
    fn visit_ty(&mut self, ty: &'tcx Ty) {
        self.resolve_type(ty);
    }
    fn visit_poly_trait_ref(&mut self,
                            tref: &'tcx ast::PolyTraitRef,
                            m: &'tcx ast::TraitBoundModifier) {
        let ast::Path { ref segments, span } = tref.trait_ref.path;
        let path: Vec<_> = segments.iter().map(|seg| seg.identifier).collect();
        let def = self.resolve_trait_reference(&path, None, span);
        self.record_def(tref.trait_ref.ref_id, def);
        visit::walk_poly_trait_ref(self, tref, m);
    }
    fn visit_variant(&mut self,
                     variant: &'tcx ast::Variant,
                     generics: &'tcx Generics,
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
    fn visit_foreign_item(&mut self, foreign_item: &'tcx ForeignItem) {
        let type_parameters = match foreign_item.node {
            ForeignItemKind::Fn(_, ref generics) => {
                HasTypeParameters(generics, ItemRibKind)
            }
            ForeignItemKind::Static(..) => NoTypeParameters,
        };
        self.with_type_parameter_rib(type_parameters, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }
    fn visit_fn(&mut self,
                function_kind: FnKind<'tcx>,
                declaration: &'tcx FnDecl,
                _: Span,
                node_id: NodeId) {
        let rib_kind = match function_kind {
            FnKind::ItemFn(_, generics, ..) => {
                self.visit_generics(generics);
                ItemRibKind
            }
            FnKind::Method(_, sig, _, _) => {
                self.visit_generics(&sig.generics);
                MethodRibKind(!sig.decl.has_self())
            }
            FnKind::Closure(_) => ClosureRibKind(node_id),
        };

        // Create a value rib for the function.
        self.ribs[ValueNS].push(Rib::new(rib_kind));

        // Create a label rib for the function.
        self.label_ribs.push(Rib::new(rib_kind));

        // Add each argument to the rib.
        let mut bindings_list = FxHashMap();
        for argument in &declaration.inputs {
            self.resolve_pattern(&argument.pat, PatternSource::FnParam, &mut bindings_list);

            self.visit_ty(&argument.ty);

            debug!("(resolving function) recorded argument");
        }
        visit::walk_fn_ret_ty(self, &declaration.output);

        // Resolve the function body.
        match function_kind {
            FnKind::ItemFn(.., body) |
            FnKind::Method(.., body) => {
                self.visit_block(body);
            }
            FnKind::Closure(body) => {
                self.visit_expr(body);
            }
        };

        debug!("(resolving function) leaving function");

        self.label_ribs.pop();
        self.ribs[ValueNS].pop();
    }
}

pub type ErrorMessage = Option<(Span, String)>;

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

    // We passed through a `macro_rules!` statement with the given expansion
    MacroDefinition(Mark),
}

/// One local scope.
#[derive(Debug)]
struct Rib<'a> {
    bindings: FxHashMap<Ident, Def>,
    kind: RibKind<'a>,
}

impl<'a> Rib<'a> {
    fn new(kind: RibKind<'a>) -> Rib<'a> {
        Rib {
            bindings: FxHashMap(),
            kind: kind,
        }
    }
}

/// A definition along with the index of the rib it was found on
#[derive(Copy, Clone)]
struct LocalDef {
    ribs: Option<(Namespace, usize)>,
    def: Def,
}

enum LexicalScopeBinding<'a> {
    Item(&'a NameBinding<'a>),
    Def(Def),
}

impl<'a> LexicalScopeBinding<'a> {
    fn item(self) -> Option<&'a NameBinding<'a>> {
        match self {
            LexicalScopeBinding::Item(binding) => Some(binding),
            _ => None,
        }
    }
}

#[derive(Clone)]
enum PathResult<'a> {
    Module(Module<'a>),
    NonModule(PathResolution),
    Indeterminate,
    Failed(String, bool /* is the error from the last segment? */),
}

enum ModuleKind {
    Block(NodeId),
    Def(Def, Name),
}

/// One node in the tree of modules.
pub struct ModuleData<'a> {
    parent: Option<Module<'a>>,
    kind: ModuleKind,

    // The def id of the closest normal module (`mod`) ancestor (including this module).
    normal_ancestor_id: DefId,

    resolutions: RefCell<FxHashMap<(Ident, Namespace), &'a RefCell<NameResolution<'a>>>>,
    legacy_macro_resolutions: RefCell<Vec<(Mark, Ident, Span)>>,
    macro_resolutions: RefCell<Vec<(Box<[Ident]>, Span)>>,

    // Macro invocations that can expand into items in this module.
    unresolved_invocations: RefCell<FxHashSet<Mark>>,

    no_implicit_prelude: bool,

    glob_importers: RefCell<Vec<&'a ImportDirective<'a>>>,
    globs: RefCell<Vec<&'a ImportDirective<'a>>>,

    // Used to memoize the traits in this module for faster searches through all traits in scope.
    traits: RefCell<Option<Box<[(Ident, &'a NameBinding<'a>)]>>>,

    // Whether this module is populated. If not populated, any attempt to
    // access the children must be preceded with a
    // `populate_module_if_necessary` call.
    populated: Cell<bool>,
}

pub type Module<'a> = &'a ModuleData<'a>;

impl<'a> ModuleData<'a> {
    fn new(parent: Option<Module<'a>>, kind: ModuleKind, normal_ancestor_id: DefId) -> Self {
        ModuleData {
            parent: parent,
            kind: kind,
            normal_ancestor_id: normal_ancestor_id,
            resolutions: RefCell::new(FxHashMap()),
            legacy_macro_resolutions: RefCell::new(Vec::new()),
            macro_resolutions: RefCell::new(Vec::new()),
            unresolved_invocations: RefCell::new(FxHashSet()),
            no_implicit_prelude: false,
            glob_importers: RefCell::new(Vec::new()),
            globs: RefCell::new((Vec::new())),
            traits: RefCell::new(None),
            populated: Cell::new(normal_ancestor_id.is_local()),
        }
    }

    fn for_each_child<F: FnMut(Ident, Namespace, &'a NameBinding<'a>)>(&self, mut f: F) {
        for (&(ident, ns), name_resolution) in self.resolutions.borrow().iter() {
            name_resolution.borrow().binding.map(|binding| f(ident, ns, binding));
        }
    }

    fn def(&self) -> Option<Def> {
        match self.kind {
            ModuleKind::Def(def, _) => Some(def),
            _ => None,
        }
    }

    fn def_id(&self) -> Option<DefId> {
        self.def().as_ref().map(Def::def_id)
    }

    // `self` resolves to the first module ancestor that `is_normal`.
    fn is_normal(&self) -> bool {
        match self.kind {
            ModuleKind::Def(Def::Mod(_), _) => true,
            _ => false,
        }
    }

    fn is_trait(&self) -> bool {
        match self.kind {
            ModuleKind::Def(Def::Trait(_), _) => true,
            _ => false,
        }
    }

    fn is_local(&self) -> bool {
        self.normal_ancestor_id.is_local()
    }
}

impl<'a> fmt::Debug for ModuleData<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.def())
    }
}

// Records a possibly-private value, type, or module definition.
#[derive(Clone, Debug)]
pub struct NameBinding<'a> {
    kind: NameBindingKind<'a>,
    expansion: Mark,
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
    Def(Def),
    Module(Module<'a>),
    Import {
        binding: &'a NameBinding<'a>,
        directive: &'a ImportDirective<'a>,
        used: Cell<bool>,
    },
    Ambiguity {
        b1: &'a NameBinding<'a>,
        b2: &'a NameBinding<'a>,
        legacy: bool,
    }
}

struct PrivacyError<'a>(Span, Name, &'a NameBinding<'a>);

struct AmbiguityError<'a> {
    span: Span,
    name: Name,
    lexical: bool,
    b1: &'a NameBinding<'a>,
    b2: &'a NameBinding<'a>,
    legacy: bool,
}

impl<'a> NameBinding<'a> {
    fn module(&self) -> Option<Module<'a>> {
        match self.kind {
            NameBindingKind::Module(module) => Some(module),
            NameBindingKind::Import { binding, .. } => binding.module(),
            NameBindingKind::Ambiguity { legacy: true, b1, .. } => b1.module(),
            _ => None,
        }
    }

    fn def(&self) -> Def {
        match self.kind {
            NameBindingKind::Def(def) => def,
            NameBindingKind::Module(module) => module.def().unwrap(),
            NameBindingKind::Import { binding, .. } => binding.def(),
            NameBindingKind::Ambiguity { legacy: true, b1, .. } => b1.def(),
            NameBindingKind::Ambiguity { .. } => Def::Err,
        }
    }

    fn get_macro(&self, resolver: &mut Resolver<'a>) -> Rc<SyntaxExtension> {
        match self.kind {
            NameBindingKind::Import { binding, .. } => binding.get_macro(resolver),
            NameBindingKind::Ambiguity { b1, .. } => b1.get_macro(resolver),
            _ => resolver.get_macro(self.def()),
        }
    }

    // We sometimes need to treat variants as `pub` for backwards compatibility
    fn pseudo_vis(&self) -> ty::Visibility {
        if self.is_variant() { ty::Visibility::Public } else { self.vis }
    }

    fn is_variant(&self) -> bool {
        match self.kind {
            NameBindingKind::Def(Def::Variant(..)) |
            NameBindingKind::Def(Def::VariantCtor(..)) => true,
            _ => false,
        }
    }

    fn is_extern_crate(&self) -> bool {
        match self.kind {
            NameBindingKind::Import {
                directive: &ImportDirective {
                    subclass: ImportDirectiveSubclass::ExternCrate, ..
                }, ..
            } => true,
            _ => false,
        }
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
            NameBindingKind::Ambiguity { b1, .. } => b1.is_glob_import(),
            _ => false,
        }
    }

    fn is_importable(&self) -> bool {
        match self.def() {
            Def::AssociatedConst(..) | Def::Method(..) | Def::AssociatedTy(..) => false,
            _ => true,
        }
    }
}

/// Interns the names of the primitive types.
struct PrimitiveTypeTable {
    primitive_types: FxHashMap<Name, PrimTy>,
}

impl PrimitiveTypeTable {
    fn new() -> PrimitiveTypeTable {
        let mut table = PrimitiveTypeTable { primitive_types: FxHashMap() };

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
        self.primitive_types.insert(Symbol::intern(string), primitive_type);
    }
}

/// The main resolver class.
pub struct Resolver<'a> {
    session: &'a Session,

    pub definitions: Definitions,

    // Maps the node id of a statement to the expansions of the `macro_rules!`s
    // immediately above the statement (if appropriate).
    macros_at_scope: FxHashMap<NodeId, Vec<Mark>>,

    graph_root: Module<'a>,

    prelude: Option<Module<'a>>,

    trait_item_map: FxHashMap<(Name, DefId), bool /* is static method? */>,

    // Names of fields of an item `DefId` accessible with dot syntax.
    // Used for hints during error reporting.
    field_names: FxHashMap<DefId, Vec<Name>>,

    // All imports known to succeed or fail.
    determined_imports: Vec<&'a ImportDirective<'a>>,

    // All non-determined imports.
    indeterminate_imports: Vec<&'a ImportDirective<'a>>,

    // The module that represents the current item scope.
    current_module: Module<'a>,

    // The current set of local scopes for types and values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    ribs: PerNS<Vec<Rib<'a>>>,

    // The current set of local scopes, for labels.
    label_ribs: Vec<Rib<'a>>,

    // The trait that the current context can refer to.
    current_trait_ref: Option<(DefId, TraitRef)>,

    // The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    // The idents for the primitive types.
    primitive_type_table: PrimitiveTypeTable,

    def_map: DefMap,
    pub freevars: FreevarMap,
    freevars_seen: NodeMap<NodeMap<usize>>,
    pub export_map: ExportMap,
    pub trait_map: TraitMap,

    // A map from nodes to anonymous modules.
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
    block_map: NodeMap<Module<'a>>,
    module_map: FxHashMap<DefId, Module<'a>>,
    extern_crate_roots: FxHashMap<(CrateNum, bool /* MacrosOnly? */), Module<'a>>,

    pub make_glob_map: bool,
    // Maps imports to the names of items actually imported (this actually maps
    // all imports, but only glob imports are actually interesting).
    pub glob_map: GlobMap,

    used_imports: FxHashSet<(NodeId, Namespace)>,
    used_crates: FxHashSet<CrateNum>,
    pub maybe_unused_trait_imports: NodeSet,

    privacy_errors: Vec<PrivacyError<'a>>,
    ambiguity_errors: Vec<AmbiguityError<'a>>,
    disallowed_shadowing: Vec<&'a LegacyBinding<'a>>,

    arenas: &'a ResolverArenas<'a>,
    dummy_binding: &'a NameBinding<'a>,
    use_extern_macros: bool, // true if `#![feature(use_extern_macros)]`

    pub exported_macros: Vec<ast::MacroDef>,
    crate_loader: &'a mut CrateLoader,
    macro_names: FxHashSet<Name>,
    builtin_macros: FxHashMap<Name, &'a NameBinding<'a>>,
    lexical_macro_resolutions: Vec<(Name, &'a Cell<LegacyScope<'a>>)>,
    macro_map: FxHashMap<DefId, Rc<SyntaxExtension>>,
    macro_exports: Vec<Export>,
    pub whitelisted_legacy_custom_derives: Vec<Name>,

    // Maps the `Mark` of an expansion to its containing module or block.
    invocations: FxHashMap<Mark, &'a InvocationData<'a>>,

    // Avoid duplicated errors for "name already defined".
    name_already_seen: FxHashMap<Name, Span>,
}

pub struct ResolverArenas<'a> {
    modules: arena::TypedArena<ModuleData<'a>>,
    local_modules: RefCell<Vec<Module<'a>>>,
    name_bindings: arena::TypedArena<NameBinding<'a>>,
    import_directives: arena::TypedArena<ImportDirective<'a>>,
    name_resolutions: arena::TypedArena<RefCell<NameResolution<'a>>>,
    invocation_data: arena::TypedArena<InvocationData<'a>>,
    legacy_bindings: arena::TypedArena<LegacyBinding<'a>>,
}

impl<'a> ResolverArenas<'a> {
    fn alloc_module(&'a self, module: ModuleData<'a>) -> Module<'a> {
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
    fn alloc_invocation_data(&'a self, expansion_data: InvocationData<'a>)
                             -> &'a InvocationData<'a> {
        self.invocation_data.alloc(expansion_data)
    }
    fn alloc_legacy_binding(&'a self, binding: LegacyBinding<'a>) -> &'a LegacyBinding<'a> {
        self.legacy_bindings.alloc(binding)
    }
}

impl<'a, 'b: 'a> ty::DefIdTree for &'a Resolver<'b> {
    fn parent(self, id: DefId) -> Option<DefId> {
        match id.krate {
            LOCAL_CRATE => self.definitions.def_key(id.index).parent,
            _ => self.session.cstore.def_key(id).parent,
        }.map(|index| DefId { index: index, ..id })
    }
}

impl<'a> hir::lowering::Resolver for Resolver<'a> {
    fn resolve_hir_path(&mut self, path: &mut hir::Path, is_value: bool) {
        let namespace = if is_value { ValueNS } else { TypeNS };
        let hir::Path { ref segments, span, ref mut def } = *path;
        let path: Vec<_> = segments.iter().map(|seg| Ident::with_empty_ctxt(seg.name)).collect();
        match self.resolve_path(&path, Some(namespace), Some(span)) {
            PathResult::Module(module) => *def = module.def().unwrap(),
            PathResult::NonModule(path_res) if path_res.depth == 0 => *def = path_res.base_def,
            PathResult::NonModule(..) => match self.resolve_path(&path, None, Some(span)) {
                PathResult::Failed(msg, _) => {
                    resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                }
                _ => {}
            },
            PathResult::Indeterminate => unreachable!(),
            PathResult::Failed(msg, _) => {
                resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
            }
        }
    }

    fn get_resolution(&mut self, id: NodeId) -> Option<PathResolution> {
        self.def_map.get(&id).cloned()
    }

    fn definitions(&mut self) -> &mut Definitions {
        &mut self.definitions
    }
}

impl<'a> Resolver<'a> {
    pub fn new(session: &'a Session,
               krate: &Crate,
               make_glob_map: MakeGlobMap,
               crate_loader: &'a mut CrateLoader,
               arenas: &'a ResolverArenas<'a>)
               -> Resolver<'a> {
        let root_def_id = DefId::local(CRATE_DEF_INDEX);
        let root_module_kind = ModuleKind::Def(Def::Mod(root_def_id), keywords::Invalid.name());
        let graph_root = arenas.alloc_module(ModuleData {
            no_implicit_prelude: attr::contains_name(&krate.attrs, "no_implicit_prelude"),
            ..ModuleData::new(None, root_module_kind, root_def_id)
        });
        let mut module_map = FxHashMap();
        module_map.insert(DefId::local(CRATE_DEF_INDEX), graph_root);

        let mut definitions = Definitions::new();
        DefCollector::new(&mut definitions).collect_root();

        let mut invocations = FxHashMap();
        invocations.insert(Mark::root(),
                           arenas.alloc_invocation_data(InvocationData::root(graph_root)));

        Resolver {
            session: session,

            definitions: definitions,
            macros_at_scope: FxHashMap(),

            // The outermost module has def ID 0; this is not reflected in the
            // AST.
            graph_root: graph_root,
            prelude: None,

            trait_item_map: FxHashMap(),
            field_names: FxHashMap(),

            determined_imports: Vec::new(),
            indeterminate_imports: Vec::new(),

            current_module: graph_root,
            ribs: PerNS {
                value_ns: vec![Rib::new(ModuleRibKind(graph_root))],
                type_ns: vec![Rib::new(ModuleRibKind(graph_root))],
                macro_ns: None,
            },
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
            block_map: NodeMap(),
            extern_crate_roots: FxHashMap(),

            make_glob_map: make_glob_map == MakeGlobMap::Yes,
            glob_map: NodeMap(),

            used_imports: FxHashSet(),
            used_crates: FxHashSet(),
            maybe_unused_trait_imports: NodeSet(),

            privacy_errors: Vec::new(),
            ambiguity_errors: Vec::new(),
            disallowed_shadowing: Vec::new(),

            arenas: arenas,
            dummy_binding: arenas.alloc_name_binding(NameBinding {
                kind: NameBindingKind::Def(Def::Err),
                expansion: Mark::root(),
                span: DUMMY_SP,
                vis: ty::Visibility::Public,
            }),
            use_extern_macros: session.features.borrow().use_extern_macros,

            exported_macros: Vec::new(),
            crate_loader: crate_loader,
            macro_names: FxHashSet(),
            builtin_macros: FxHashMap(),
            lexical_macro_resolutions: Vec::new(),
            macro_map: FxHashMap(),
            macro_exports: Vec::new(),
            invocations: invocations,
            name_already_seen: FxHashMap(),
            whitelisted_legacy_custom_derives: Vec::new(),
        }
    }

    pub fn arenas() -> ResolverArenas<'a> {
        ResolverArenas {
            modules: arena::TypedArena::new(),
            local_modules: RefCell::new(Vec::new()),
            name_bindings: arena::TypedArena::new(),
            import_directives: arena::TypedArena::new(),
            name_resolutions: arena::TypedArena::new(),
            invocation_data: arena::TypedArena::new(),
            legacy_bindings: arena::TypedArena::new(),
        }
    }

    fn per_ns<T, F: FnMut(&mut Self, Namespace) -> T>(&mut self, mut f: F) -> PerNS<T> {
        PerNS {
            type_ns: f(self, TypeNS),
            value_ns: f(self, ValueNS),
            macro_ns: match self.use_extern_macros {
                true => Some(f(self, MacroNS)),
                false => None,
            },
        }
    }

    /// Entry point to crate resolution.
    pub fn resolve_crate(&mut self, krate: &Crate) {
        ImportResolver { resolver: self }.finalize_imports();
        self.current_module = self.graph_root;
        self.finalize_current_module_macro_resolutions();
        visit::walk_crate(self, krate);

        check_unused::check_crate(self, krate);
        self.report_errors();
        self.crate_loader.postprocess(krate);
    }

    fn new_module(&self, parent: Module<'a>, kind: ModuleKind, normal_ancestor_id: DefId)
                  -> Module<'a> {
        self.arenas.alloc_module(ModuleData::new(Some(parent), kind, normal_ancestor_id))
    }

    fn record_use(&mut self, ident: Ident, ns: Namespace, binding: &'a NameBinding<'a>, span: Span)
                  -> bool /* true if an error was reported */ {
        // track extern crates for unused_extern_crate lint
        if let Some(DefId { krate, .. }) = binding.module().and_then(ModuleData::def_id) {
            self.used_crates.insert(krate);
        }

        match binding.kind {
            NameBindingKind::Import { directive, binding, ref used } if !used.get() => {
                used.set(true);
                self.used_imports.insert((directive.id, ns));
                self.add_to_glob_map(directive.id, ident);
                self.record_use(ident, ns, binding, span)
            }
            NameBindingKind::Import { .. } => false,
            NameBindingKind::Ambiguity { b1, b2, legacy } => {
                self.ambiguity_errors.push(AmbiguityError {
                    span: span, name: ident.name, lexical: false, b1: b1, b2: b2, legacy: legacy,
                });
                if legacy {
                    self.record_use(ident, ns, b1, span);
                }
                !legacy
            }
            _ => false
        }
    }

    fn add_to_glob_map(&mut self, id: NodeId, ident: Ident) {
        if self.make_glob_map {
            self.glob_map.entry(id).or_insert_with(FxHashSet).insert(ident.name);
        }
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
                                      mut ident: Ident,
                                      ns: Namespace,
                                      record_used: Option<Span>)
                                      -> Option<LexicalScopeBinding<'a>> {
        if ns == TypeNS {
            ident = ident.unhygienize();
        }

        // Walk backwards up the ribs in scope.
        for i in (0 .. self.ribs[ns].len()).rev() {
            if let Some(def) = self.ribs[ns][i].bindings.get(&ident).cloned() {
                // The ident resolves to a type parameter or local variable.
                return Some(LexicalScopeBinding::Def(if let Some(span) = record_used {
                    self.adjust_local_def(LocalDef { ribs: Some((ns, i)), def: def }, span)
                } else {
                    def
                }));
            }

            if let ModuleRibKind(module) = self.ribs[ns][i].kind {
                let item = self.resolve_ident_in_module(module, ident, ns, false, record_used);
                if let Ok(binding) = item {
                    // The ident resolves to an item.
                    return Some(LexicalScopeBinding::Item(binding));
                }

                if let ModuleKind::Block(..) = module.kind { // We can see through blocks
                } else if !module.no_implicit_prelude {
                    return self.prelude.and_then(|prelude| {
                        self.resolve_ident_in_module(prelude, ident, ns, false, None).ok()
                    }).map(LexicalScopeBinding::Item)
                } else {
                    return None;
                }
            }

            if let MacroDefinition(mac) = self.ribs[ns][i].kind {
                // If an invocation of this macro created `ident`, give up on `ident`
                // and switch to `ident`'s source from the macro definition.
                let (source_ctxt, source_macro) = ident.ctxt.source();
                if source_macro == mac {
                    ident.ctxt = source_ctxt;
                }
            }
        }

        None
    }

    fn resolve_crate_var(&mut self, mut crate_var_ctxt: SyntaxContext) -> Module<'a> {
        while crate_var_ctxt.source().0 != SyntaxContext::empty() {
            crate_var_ctxt = crate_var_ctxt.source().0;
        }
        let module = self.invocations[&crate_var_ctxt.source().1].module.get();
        if module.is_local() { self.graph_root } else { module }
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
        let id = self.definitions.local_def_id(id);
        let module = self.module_map.get(&id).cloned(); // clones a reference
        if let Some(module) = module {
            // Move down in the graph.
            let orig_module = replace(&mut self.current_module, module);
            self.ribs[ValueNS].push(Rib::new(ModuleRibKind(module)));
            self.ribs[TypeNS].push(Rib::new(ModuleRibKind(module)));

            self.finalize_current_module_macro_resolutions();
            f(self);

            self.current_module = orig_module;
            self.ribs[ValueNS].pop();
            self.ribs[TypeNS].pop();
        } else {
            f(self);
        }
    }

    /// Searches the current set of local scopes for labels.
    /// Stops after meeting a closure.
    fn search_label(&self, mut ident: Ident) -> Option<Def> {
        for rib in self.label_ribs.iter().rev() {
            match rib.kind {
                NormalRibKind => {
                    // Continue
                }
                MacroDefinition(mac) => {
                    // If an invocation of this macro created `ident`, give up on `ident`
                    // and switch to `ident`'s source from the macro definition.
                    let (source_ctxt, source_macro) = ident.ctxt.source();
                    if source_macro == mac {
                        ident.ctxt = source_ctxt;
                    }
                }
                _ => {
                    // Do not resolve labels across function boundary
                    return None;
                }
            }
            let result = rib.bindings.get(&ident).cloned();
            if result.is_some() {
                return result;
            }
        }
        None
    }

    fn resolve_item(&mut self, item: &Item) {
        let name = item.ident.name;

        debug!("(resolving item) resolving {}", name);

        match item.node {
            ItemKind::Enum(_, ref generics) |
            ItemKind::Ty(_, ref generics) |
            ItemKind::Struct(_, ref generics) |
            ItemKind::Union(_, ref generics) |
            ItemKind::Fn(.., ref generics, _) => {
                self.with_type_parameter_rib(HasTypeParameters(generics, ItemRibKind),
                                             |this| visit::walk_item(this, item));
            }

            ItemKind::DefaultImpl(_, ref trait_ref) => {
                self.with_optional_trait_ref(Some(trait_ref), |_, _| {}, None);
            }
            ItemKind::Impl(.., ref generics, ref opt_trait_ref, ref self_type, ref impl_items) =>
                self.resolve_implementation(generics,
                                            opt_trait_ref,
                                            &self_type,
                                            item.id,
                                            impl_items),

            ItemKind::Trait(_, ref generics, ref bounds, ref trait_items) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_type_parameter_rib(HasTypeParameters(generics, ItemRibKind), |this| {
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
                        let path: Vec<_> =
                            prefix.segments.iter().map(|seg| seg.identifier).collect();
                        // Resolve prefix of an import with empty braces (issue #28388)
                        if items.is_empty() && !prefix.segments.is_empty() {
                            let span = prefix.span;
                            // FIXME(#38012) This should be a module path, not anything in TypeNS.
                            let result = self.resolve_path(&path, Some(TypeNS), Some(span));
                            let (def, msg) = match result {
                                PathResult::Module(module) => (module.def().unwrap(), None),
                                PathResult::NonModule(res) if res.depth == 0 =>
                                    (res.base_def, None),
                                PathResult::NonModule(_) => {
                                    // Resolve a module path for better errors
                                    match self.resolve_path(&path, None, Some(span)) {
                                        PathResult::Failed(msg, _) => (Def::Err, Some(msg)),
                                        _ => unreachable!(),
                                    }
                                }
                                PathResult::Indeterminate => unreachable!(),
                                PathResult::Failed(msg, _) => (Def::Err, Some(msg)),
                            };
                            if let Some(msg) = msg {
                                resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                            }
                            self.record_def(item.id, PathResolution::new(def));
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
            HasTypeParameters(generics, rib_kind) => {
                let mut function_type_rib = Rib::new(rib_kind);
                let mut seen_bindings = FxHashMap();
                for type_parameter in &generics.ty_params {
                    let name = type_parameter.ident.name;
                    debug!("with_type_parameter_rib: {}", type_parameter.id);

                    if seen_bindings.contains_key(&name) {
                        let span = seen_bindings.get(&name).unwrap();
                        resolve_error(self,
                                      type_parameter.span,
                                      ResolutionError::NameAlreadyUsedInTypeParameterList(name,
                                                                                          span));
                    }
                    seen_bindings.entry(name).or_insert(type_parameter.span);

                    // plain insert (no renaming)
                    let def_id = self.definitions.local_def_id(type_parameter.id);
                    let def = Def::TyParam(def_id);
                    function_type_rib.bindings.insert(Ident::with_empty_ctxt(name), def);
                    self.record_def(type_parameter.id, PathResolution::new(def));
                }
                self.ribs[TypeNS].push(function_type_rib);
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f(self);

        if let HasTypeParameters(..) = type_parameters {
            self.ribs[TypeNS].pop();
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
        self.ribs[ValueNS].push(Rib::new(ConstantItemRibKind));
        self.ribs[TypeNS].push(Rib::new(ConstantItemRibKind));
        f(self);
        self.ribs[TypeNS].pop();
        self.ribs[ValueNS].pop();
    }

    fn resolve_trait_reference(&mut self,
                               path: &[Ident],
                               generics: Option<&Generics>,
                               span: Span)
                               -> PathResolution {
        let def = match self.resolve_path(path, None, Some(span)) {
            PathResult::Module(module) => Some(module.def().unwrap()),
            PathResult::NonModule(..) => return err_path_resolution(),
            PathResult::Failed(msg, false) => {
                resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                return err_path_resolution();
            }
            _ => match self.resolve_path(path, Some(TypeNS), None) {
                PathResult::NonModule(path_resolution) => Some(path_resolution.base_def),
                _ => None,
            },
        };

        if let Some(def) = def {
            if let Def::Trait(_) = def {
                return PathResolution::new(def);
            }

            let mut err = resolve_struct_error(self, span, {
                ResolutionError::IsNotATrait(&names_to_string(path), def.kind_name())
            });
            if let Some(generics) = generics {
                if let Some(span) = generics.span_for_name(&names_to_string(path)) {
                    err.span_label(span, &"type parameter defined here");
                }
            }

            // If it's a typedef, give a note
            if let Def::TyAlias(..) = def {
                err.note(&format!("type aliases cannot be used for traits"));
            }
            err.emit();
        } else {
            // find possible candidates
            let is_trait = |def| match def { Def::Trait(_) => true, _ => false };
            let candidates = self.lookup_candidates(path.last().unwrap().name, TypeNS, is_trait);

            let path = names_to_string(path);
            resolve_error(self, span, ResolutionError::UndeclaredTraitName(&path, candidates));
        }
        err_path_resolution()
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

    fn with_optional_trait_ref<T, F>(&mut self,
                                     opt_trait_ref: Option<&TraitRef>,
                                     f: F,
                                     generics: Option<&Generics>)
        -> T
        where F: FnOnce(&mut Resolver, Option<DefId>) -> T
    {
        let mut new_val = None;
        let mut new_id = None;
        if let Some(trait_ref) = opt_trait_ref {
            let ast::Path { ref segments, span } = trait_ref.path;
            let path: Vec<_> = segments.iter().map(|seg| seg.identifier).collect();
            let path_res = self.resolve_trait_reference(&path, generics, span);
            assert!(path_res.depth == 0);
            self.record_def(trait_ref.ref_id, path_res);
            if path_res.base_def != Def::Err {
                new_val = Some((path_res.base_def.def_id(), trait_ref.clone()));
                new_id = Some(path_res.base_def.def_id());
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
        self_type_rib.bindings.insert(keywords::SelfType.ident(), self_def);
        self.ribs[TypeNS].push(self_type_rib);
        f(self);
        self.ribs[TypeNS].pop();
    }

    fn resolve_implementation(&mut self,
                              generics: &Generics,
                              opt_trait_reference: &Option<TraitRef>,
                              self_type: &Ty,
                              item_id: NodeId,
                              impl_items: &[ImplItem]) {
        // If applicable, create a rib for the type parameters.
        self.with_type_parameter_rib(HasTypeParameters(generics, ItemRibKind), |this| {
            // Resolve the type parameters.
            this.visit_generics(generics);

            // Resolve the trait reference, if necessary.
            this.with_optional_trait_ref(opt_trait_reference.as_ref(), |this, trait_id| {
                // Resolve the self type.
                this.visit_ty(self_type);

                let item_def_id = this.definitions.local_def_id(item_id);
                this.with_self_rib(Def::SelfTy(trait_id, Some(item_def_id)), |this| {
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
            }, Some(&generics));
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
        self.resolve_pattern(&local.pat, PatternSource::Let, &mut FxHashMap());
    }

    // build a map from pattern identifiers to binding-info's.
    // this is done hygienically. This could arise for a macro
    // that expands into an or-pattern where one 'x' was from the
    // user and one 'x' came from the macro.
    fn binding_mode_map(&mut self, pat: &Pat) -> BindingMap {
        let mut binding_map = FxHashMap();

        pat.walk(&mut |pat| {
            if let PatKind::Ident(binding_mode, ident, ref sub_pat) = pat.node {
                if sub_pat.is_some() || match self.def_map.get(&pat.id) {
                    Some(&PathResolution { base_def: Def::Local(..), .. }) => true,
                    _ => false,
                } {
                    let binding_info = BindingInfo { span: ident.span, binding_mode: binding_mode };
                    binding_map.insert(ident.node, binding_info);
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
                        let error = ResolutionError::VariableNotBoundInPattern(key.name, 1, i + 1);
                        resolve_error(self, p.span, error);
                    }
                    Some(binding_i) => {
                        if binding_0.binding_mode != binding_i.binding_mode {
                            resolve_error(self,
                                          binding_i.span,
                                          ResolutionError::VariableBoundWithDifferentMode(
                                              key.name,
                                              i + 1,
                                              binding_0.span));
                        }
                    }
                }
            }

            for (&key, &binding) in &map_i {
                if !map_0.contains_key(&key) {
                    resolve_error(self,
                                  binding.span,
                                  ResolutionError::VariableNotBoundInPattern(key.name, i + 1, 1));
                }
            }
        }
    }

    fn resolve_arm(&mut self, arm: &Arm) {
        self.ribs[ValueNS].push(Rib::new(NormalRibKind));

        let mut bindings_list = FxHashMap();
        for pattern in &arm.pats {
            self.resolve_pattern(&pattern, PatternSource::Match, &mut bindings_list);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        walk_list!(self, visit_expr, &arm.guard);
        self.visit_expr(&arm.body);

        self.ribs[ValueNS].pop();
    }

    fn resolve_block(&mut self, block: &Block) {
        debug!("(resolving block) entering block");
        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        let anonymous_module = self.block_map.get(&block.id).cloned(); // clones a reference

        let mut num_macro_definition_ribs = 0;
        if let Some(anonymous_module) = anonymous_module {
            debug!("(resolving block) found anonymous module, moving down");
            self.ribs[ValueNS].push(Rib::new(ModuleRibKind(anonymous_module)));
            self.ribs[TypeNS].push(Rib::new(ModuleRibKind(anonymous_module)));
            self.current_module = anonymous_module;
            self.finalize_current_module_macro_resolutions();
        } else {
            self.ribs[ValueNS].push(Rib::new(NormalRibKind));
        }

        // Descend into the block.
        for stmt in &block.stmts {
            if let Some(marks) = self.macros_at_scope.remove(&stmt.id) {
                num_macro_definition_ribs += marks.len() as u32;
                for mark in marks {
                    self.ribs[ValueNS].push(Rib::new(MacroDefinition(mark)));
                    self.label_ribs.push(Rib::new(MacroDefinition(mark)));
                }
            }

            self.visit_stmt(stmt);
        }

        // Move back up.
        self.current_module = orig_module;
        for _ in 0 .. num_macro_definition_ribs {
            self.ribs[ValueNS].pop();
            self.label_ribs.pop();
        }
        self.ribs[ValueNS].pop();
        if let Some(_) = anonymous_module {
            self.ribs[TypeNS].pop();
        }
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(&mut self, ty: &Ty) {
        if let TyKind::Path(ref maybe_qself, ref path) = ty.node {
            // This is a path in the type namespace. Walk through scopes looking for it.
            if let Some(def) =
                    self.resolve_possibly_assoc_item(ty.id, maybe_qself.as_ref(), path, TypeNS) {
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
                let kind = if maybe_qself.is_some() { "associated type" } else { "type name" };
                let is_invalid_self_type_name = {
                    path.segments.len() > 0 &&
                    maybe_qself.is_none() &&
                    path.segments[0].identifier.name == keywords::SelfType.name()
                };

                if is_invalid_self_type_name {
                    resolve_error(self, ty.span, ResolutionError::SelfUsedOutsideImplOrTrait);
                } else {
                    let type_name = path.segments.last().unwrap().identifier.name;
                    let candidates = self.lookup_candidates(type_name, TypeNS, |def| {
                        match def {
                            Def::Trait(_) |
                            Def::Enum(_) |
                            Def::Struct(_) |
                            Def::Union(_) |
                            Def::TyAlias(_) => true,
                            _ => false,
                        }
                    });

                    let name = &path_names_to_string(path, 0);
                    let error = ResolutionError::UseOfUndeclared(kind, name, candidates);
                    resolve_error(self, ty.span, error);
                }
            }
        }
        // Resolve embedded types.
        visit::walk_ty(self, ty);
    }

    fn fresh_binding(&mut self,
                     ident: &SpannedIdent,
                     pat_id: NodeId,
                     outer_pat_id: NodeId,
                     pat_src: PatternSource,
                     bindings: &mut FxHashMap<Ident, NodeId>)
                     -> PathResolution {
        // Add the binding to the local ribs, if it
        // doesn't already exist in the bindings map. (We
        // must not add it if it's in the bindings map
        // because that breaks the assumptions later
        // passes make about or-patterns.)
        let mut def = Def::Local(self.definitions.local_def_id(pat_id));
        match bindings.get(&ident.node).cloned() {
            Some(id) if id == outer_pat_id => {
                // `Variant(a, a)`, error
                resolve_error(
                    self,
                    ident.span,
                    ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(
                        &ident.node.name.as_str())
                );
            }
            Some(..) if pat_src == PatternSource::FnParam => {
                // `fn f(a: u8, a: u8)`, error
                resolve_error(
                    self,
                    ident.span,
                    ResolutionError::IdentifierBoundMoreThanOnceInParameterList(
                        &ident.node.name.as_str())
                );
            }
            Some(..) if pat_src == PatternSource::Match => {
                // `Variant1(a) | Variant2(a)`, ok
                // Reuse definition from the first `a`.
                def = self.ribs[ValueNS].last_mut().unwrap().bindings[&ident.node];
            }
            Some(..) => {
                span_bug!(ident.span, "two bindings with the same name from \
                                       unexpected pattern source {:?}", pat_src);
            }
            None => {
                // A completely fresh binding, add to the lists if it's valid.
                if ident.node.name != keywords::Invalid.name() {
                    bindings.insert(ident.node, outer_pat_id);
                    self.ribs[ValueNS].last_mut().unwrap().bindings.insert(ident.node, def);
                }
            }
        }

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
                if expected_fn(resolution.base_def) || resolution.base_def == Def::Err {
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
            let error = ResolutionError::PatPathUnresolved(expected_what, path);
            resolve_error(self, path.span, error);
            err_path_resolution()
        };

        self.record_def(pat_id, resolution);
    }

    fn resolve_struct_path(&mut self, node_id: NodeId, path: &Path) {
        // Resolution logic is equivalent for expressions and patterns,
        // reuse `resolve_pattern_path` for both.
        self.resolve_pattern_path(node_id, None, path, TypeNS, |def| {
            match def {
                Def::Struct(..) | Def::Union(..) | Def::Variant(..) |
                Def::TyAlias(..) | Def::AssociatedTy(..) | Def::SelfTy(..) => true,
                _ => false,
            }
        }, "struct, variant or union type");
    }

    fn resolve_pattern(&mut self,
                       pat: &Pat,
                       pat_src: PatternSource,
                       // Maps idents to the node ID for the
                       // outermost pattern that binds them.
                       bindings: &mut FxHashMap<Ident, NodeId>) {
        // Visit all direct subpatterns of this pattern.
        let outer_pat_id = pat.id;
        pat.walk(&mut |pat| {
            match pat.node {
                PatKind::Ident(bmode, ref ident, ref opt_pat) => {
                    // First try to resolve the identifier as some existing
                    // entity, then fall back to a fresh binding.
                    let binding = self.resolve_ident_in_lexical_scope(ident.node, ValueNS, None)
                                      .and_then(LexicalScopeBinding::item);
                    let resolution = binding.map(NameBinding::def).and_then(|def| {
                        let always_binding = !pat_src.is_refutable() || opt_pat.is_some() ||
                                             bmode != BindingMode::ByValue(Mutability::Immutable);
                        match def {
                            Def::StructCtor(_, CtorKind::Const) |
                            Def::VariantCtor(_, CtorKind::Const) |
                            Def::Const(..) if !always_binding => {
                                // A unit struct/variant or constant pattern.
                                self.record_use(ident.node, ValueNS, binding.unwrap(), ident.span);
                                Some(PathResolution::new(def))
                            }
                            Def::StructCtor(..) | Def::VariantCtor(..) |
                            Def::Const(..) | Def::Static(..) => {
                                // A fresh binding that shadows something unacceptable.
                                resolve_error(
                                    self,
                                    ident.span,
                                    ResolutionError::BindingShadowsSomethingUnacceptable(
                                        pat_src.descr(), ident.node.name, binding.unwrap())
                                );
                                None
                            }
                            Def::Local(..) | Def::Upvar(..) | Def::Fn(..) | Def::Err => {
                                // These entities are explicitly allowed
                                // to be shadowed by fresh bindings.
                                None
                            }
                            def => {
                                span_bug!(ident.span, "unexpected definition for an \
                                                       identifier in pattern: {:?}", def);
                            }
                        }
                    }).unwrap_or_else(|| {
                        self.fresh_binding(ident, pat.id, outer_pat_id, pat_src, bindings)
                    });

                    self.record_def(pat.id, resolution);
                }

                PatKind::TupleStruct(ref path, ..) => {
                    self.resolve_pattern_path(pat.id, None, path, ValueNS, |def| {
                        match def {
                            Def::StructCtor(_, CtorKind::Fn) |
                            Def::VariantCtor(_, CtorKind::Fn) => true,
                            _ => false,
                        }
                    }, "tuple struct/variant");
                }

                PatKind::Path(ref qself, ref path) => {
                    self.resolve_pattern_path(pat.id, qself.as_ref(), path, ValueNS, |def| {
                        match def {
                            Def::StructCtor(_, CtorKind::Const) |
                            Def::VariantCtor(_, CtorKind::Const) |
                            Def::Const(..) | Def::AssociatedConst(..) => true,
                            _ => false,
                        }
                    }, "unit struct/variant or constant");
                }

                PatKind::Struct(ref path, ..) => {
                    self.resolve_struct_path(pat.id, path);
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
                                   ns: Namespace)
                                   -> Option<PathResolution> {
        let ast::Path { ref segments, span } = *path;
        let path: Vec<_> = segments.iter().map(|seg| seg.identifier).collect();

        if let Some(qself) = maybe_qself {
            if qself.position == 0 {
                // FIXME: Create some fake resolution that can't possibly be a type.
                return Some(PathResolution {
                    base_def: Def::Mod(self.definitions.local_def_id(ast::CRATE_NODE_ID)),
                    depth: path.len(),
                });
            }
            // Make sure the trait is valid.
            self.resolve_trait_reference(&path[..qself.position], None, span);
        }

        let result = match self.resolve_path(&path, Some(ns), Some(span)) {
            PathResult::NonModule(path_res) => match path_res.base_def {
                Def::Trait(..) if maybe_qself.is_some() => return None,
                _ => path_res,
            },
            PathResult::Module(module) if !module.is_normal() => {
                PathResolution::new(module.def().unwrap())
            }
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
            PathResult::Module(..) | PathResult::Failed(..)
                    if (ns == TypeNS || path.len() > 1) &&
                       self.primitive_type_table.primitive_types.contains_key(&path[0].name) => {
                PathResolution {
                    base_def: Def::PrimTy(self.primitive_type_table.primitive_types[&path[0].name]),
                    depth: segments.len() - 1,
                }
            }
            PathResult::Module(module) => PathResolution::new(module.def().unwrap()),
            PathResult::Failed(msg, false) => {
                resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                err_path_resolution()
            }
            _ => return None,
        };

        if path.len() == 1 || result.base_def == Def::Err {
            return Some(result);
        }

        let unqualified_result = {
            match self.resolve_path(&[*path.last().unwrap()], Some(ns), None) {
                PathResult::NonModule(path_res) => path_res.base_def,
                PathResult::Module(module) => module.def().unwrap(),
                _ => return Some(result),
            }
        };
        if result.base_def == unqualified_result && path[0].name != "$crate" {
            let lint = lint::builtin::UNUSED_QUALIFICATIONS;
            self.session.add_lint(lint, id, span, "unnecessary qualification".to_string());
        }

        Some(result)
    }

    fn resolve_path(&mut self,
                    path: &[Ident],
                    opt_ns: Option<Namespace>, // `None` indicates a module path
                    record_used: Option<Span>)
                    -> PathResult<'a> {
        let mut module = None;
        let mut allow_super = true;

        for (i, &ident) in path.iter().enumerate() {
            let is_last = i == path.len() - 1;
            let ns = if is_last { opt_ns.unwrap_or(TypeNS) } else { TypeNS };

            if i == 0 && ns == TypeNS && ident.name == keywords::SelfValue.name() {
                module = Some(self.module_map[&self.current_module.normal_ancestor_id]);
                continue
            } else if allow_super && ns == TypeNS && ident.name == keywords::Super.name() {
                let current_module = if i == 0 { self.current_module } else { module.unwrap() };
                let self_module = self.module_map[&current_module.normal_ancestor_id];
                if let Some(parent) = self_module.parent {
                    module = Some(self.module_map[&parent.normal_ancestor_id]);
                    continue
                } else {
                    let msg = "There are too many initial `super`s.".to_string();
                    return PathResult::Failed(msg, false);
                }
            }
            allow_super = false;

            if i == 0 && ns == TypeNS && ident.name == keywords::CrateRoot.name() {
                module = Some(self.graph_root);
                continue
            } else if i == 0 && ns == TypeNS && ident.name == "$crate" {
                module = Some(self.resolve_crate_var(ident.ctxt));
                continue
            }

            let binding = if let Some(module) = module {
                self.resolve_ident_in_module(module, ident, ns, false, record_used)
            } else if opt_ns == Some(MacroNS) {
                self.resolve_lexical_macro_path_segment(ident, ns, record_used)
            } else {
                match self.resolve_ident_in_lexical_scope(ident, ns, record_used) {
                    Some(LexicalScopeBinding::Item(binding)) => Ok(binding),
                    Some(LexicalScopeBinding::Def(def))
                            if opt_ns == Some(TypeNS) || opt_ns == Some(ValueNS) => {
                        return PathResult::NonModule(PathResolution {
                            base_def: def,
                            depth: path.len() - 1,
                        });
                    }
                    _ => Err(if record_used.is_some() { Determined } else { Undetermined }),
                }
            };

            match binding {
                Ok(binding) => {
                    if let Some(next_module) = binding.module() {
                        module = Some(next_module);
                    } else if binding.def() == Def::Err {
                        return PathResult::NonModule(err_path_resolution());
                    } else if opt_ns.is_some() && !(opt_ns == Some(MacroNS) && !is_last) {
                        return PathResult::NonModule(PathResolution {
                            base_def: binding.def(),
                            depth: path.len() - i - 1,
                        });
                    } else {
                        return PathResult::Failed(format!("Not a module `{}`", ident), is_last);
                    }
                }
                Err(Undetermined) => return PathResult::Indeterminate,
                Err(Determined) => {
                    if let Some(module) = module {
                        if opt_ns.is_some() && !module.is_normal() {
                            return PathResult::NonModule(PathResolution {
                                base_def: module.def().unwrap(),
                                depth: path.len() - i,
                            });
                        }
                    }
                    let msg = if module.and_then(ModuleData::def) == self.graph_root.def() {
                        let is_mod = |def| match def { Def::Mod(..) => true, _ => false };
                        let mut candidates =
                            self.lookup_candidates(ident.name, TypeNS, is_mod).candidates;
                        candidates.sort_by_key(|path| (path.segments.len(), path.to_string()));
                        if let Some(candidate) = candidates.get(0) {
                            format!("Did you mean `{}`?", candidate)
                        } else {
                            format!("Maybe a missing `extern crate {};`?", ident)
                        }
                    } else if i == 0 {
                        format!("Use of undeclared type or module `{}`", ident)
                    } else {
                        format!("Could not find `{}` in `{}`", ident, path[i - 1])
                    };
                    return PathResult::Failed(msg, is_last);
                }
            }
        }

        PathResult::Module(module.unwrap_or(self.graph_root))
    }

    // Resolve a local definition, potentially adjusting for closures.
    fn adjust_local_def(&mut self, local_def: LocalDef, span: Span) -> Def {
        let ribs = match local_def.ribs {
            Some((ns, i)) => &self.ribs[ns][i + 1..],
            None => &[] as &[_],
        };
        let mut def = local_def.def;
        match def {
            Def::Upvar(..) => {
                span_bug!(span, "unexpected {:?} in bindings", def)
            }
            Def::Local(def_id) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind | ModuleRibKind(..) | MacroDefinition(..) => {
                            // Nothing to do. Continue.
                        }
                        ClosureRibKind(function_id) => {
                            let prev_def = def;
                            let node_id = self.definitions.as_local_node_id(def_id).unwrap();

                            let seen = self.freevars_seen
                                           .entry(function_id)
                                           .or_insert_with(|| NodeMap());
                            if let Some(&index) = seen.get(&node_id) {
                                def = Def::Upvar(def_id, index, function_id);
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

                            def = Def::Upvar(def_id, depth, function_id);
                            seen.insert(node_id, depth);
                        }
                        ItemRibKind | MethodRibKind(_) => {
                            // This was an attempt to access an upvar inside a
                            // named function item. This is not allowed, so we
                            // report an error.
                            resolve_error(self,
                                          span,
                                          ResolutionError::CannotCaptureDynamicEnvironmentInFnItem);
                            return Def::Err;
                        }
                        ConstantItemRibKind => {
                            // Still doesn't deal with upvars
                            resolve_error(self,
                                          span,
                                          ResolutionError::AttemptToUseNonConstantValueInConstant);
                            return Def::Err;
                        }
                    }
                }
            }
            Def::TyParam(..) | Def::SelfTy(..) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind | MethodRibKind(_) | ClosureRibKind(..) |
                        ModuleRibKind(..) | MacroDefinition(..) => {
                            // Nothing to do. Continue.
                        }
                        ItemRibKind => {
                            // This was an attempt to use a type parameter outside
                            // its scope.

                            resolve_error(self,
                                          span,
                                          ResolutionError::TypeParametersFromOuterFunction);
                            return Def::Err;
                        }
                        ConstantItemRibKind => {
                            // see #9186
                            resolve_error(self, span, ResolutionError::OuterTypeParameterContext);
                            return Def::Err;
                        }
                    }
                }
            }
            _ => {}
        }
        return def;
    }

    // Calls `f` with a `Resolver` whose current lexical scope is `module`'s lexical scope,
    // i.e. the module's items and the prelude (unless the module is `#[no_implicit_prelude]`).
    // FIXME #34673: This needs testing.
    pub fn with_module_lexical_scope<T, F>(&mut self, module: Module<'a>, f: F) -> T
        where F: FnOnce(&mut Resolver<'a>) -> T,
    {
        self.with_empty_ribs(|this| {
            this.ribs[ValueNS].push(Rib::new(ModuleRibKind(module)));
            this.ribs[TypeNS].push(Rib::new(ModuleRibKind(module)));
            f(this)
        })
    }

    fn with_empty_ribs<T, F>(&mut self, f: F) -> T
        where F: FnOnce(&mut Resolver<'a>) -> T,
    {
        let ribs = replace(&mut self.ribs, PerNS::<Vec<Rib>>::default());
        let label_ribs = replace(&mut self.label_ribs, Vec::new());

        let result = f(self);
        self.ribs = ribs;
        self.label_ribs = label_ribs;
        result
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
                    Def::Struct(did) | Def::Union(did) if resolution.depth == 0 => {
                        if let Some(field_names) = self.field_names.get(&did) {
                            if field_names.iter().any(|&field_name| name == field_name) {
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
        if let Some(macro_name) = self.macro_names.iter().find(|&n| n == &name) {
            return SuggestionType::Macro(format!("{}!", macro_name));
        }

        let names = self.ribs[ValueNS]
                    .iter()
                    .rev()
                    .flat_map(|rib| rib.bindings.keys().map(|ident| &ident.name));

        if let Some(found) = find_best_match_for_name(names, name, None) {
            if found != name {
                return SuggestionType::Function(found);
            }
        } SuggestionType::NotFound
    }

    fn resolve_labeled_block(&mut self, label: Option<SpannedIdent>, id: NodeId, block: &Block) {
        if let Some(label) = label {
            let def = Def::Label(id);
            self.with_label_rib(|this| {
                this.label_ribs.last_mut().unwrap().bindings.insert(label.node, def);
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
                    let is_struct_variant = match path_res.base_def {
                        Def::VariantCtor(_, CtorKind::Fictive) => true,
                        _ => false,
                    };
                    if is_struct_variant {
                        let path_name = path_names_to_string(path, 0);

                        let mut err = resolve_struct_error(self,
                                        expr.span,
                                        ResolutionError::StructVariantUsedAsFunction(&path_name));

                        let msg = format!("did you mean to write: `{} {{ /* fields */ }}`?",
                                          path_name);
                        err.help(&msg);
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
                    let path_name = path_names_to_string(path, 0);
                    let path: Vec<_> = path.segments.iter().map(|seg| seg.identifier).collect();
                    let type_res = match self.resolve_path(&path, Some(TypeNS), None) {
                        PathResult::NonModule(type_res) => Some(type_res),
                        _ => None,
                    };

                    self.record_def(expr.id, err_path_resolution());

                    if let Some(Def::Struct(..)) = type_res.map(|r| r.base_def) {
                        let error_variant =
                            ResolutionError::StructVariantUsedAsFunction(&path_name);
                        let mut err = resolve_struct_error(self, expr.span, error_variant);

                        let msg = format!("did you mean to write: `{} {{ /* fields */ }}`?",
                                          path_name);

                        err.help(&msg);
                        err.emit();
                    } else {
                        // Keep reporting some errors even if they're ignored above.
                        let mut method_scope = false;
                        let mut is_static = false;
                        self.ribs[ValueNS].iter().rev().all(|rib| {
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

                        if method_scope && keywords::SelfValue.name() == &*path_name {
                            let error = ResolutionError::SelfNotAvailableInStaticMethod;
                            resolve_error(self, expr.span, error);
                        } else {
                            let fallback =
                                self.find_fallback_in_self_type(path.last().unwrap().name);
                            let (mut msg, is_field) = match fallback {
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

                            let mut context = UnresolvedNameContext::Other;
                            let mut def = Def::Err;
                            if !msg.is_empty() {
                                msg = format!("did you mean {}?", msg);
                            } else {
                                // we display a help message if this is a module
                                if let PathResult::Module(module) =
                                        self.resolve_path(&path, None, None) {
                                    def = module.def().unwrap();
                                    context = UnresolvedNameContext::PathIsMod(parent);
                                }
                            }

                            let error = ResolutionError::UnresolvedName {
                                path: &path_name,
                                message: &msg,
                                context: context,
                                is_static_method: method_scope && is_static,
                                is_field: is_field,
                                def: def,
                            };
                            resolve_error(self, expr.span, error);
                        }
                    }
                }

                visit::walk_expr(self, expr);
            }

            ExprKind::Struct(ref path, ..) => {
                self.resolve_struct_path(expr.id, path);

                visit::walk_expr(self, expr);
            }

            ExprKind::Break(Some(label), _) | ExprKind::Continue(Some(label)) => {
                match self.search_label(label.node) {
                    None => {
                        self.record_def(expr.id, err_path_resolution());
                        resolve_error(self,
                                      label.span,
                                      ResolutionError::UndeclaredLabel(&label.node.name.as_str()));
                    }
                    Some(def @ Def::Label(_)) => {
                        // Since this def is a label, it is never read.
                        self.record_def(expr.id, PathResolution::new(def));
                    }
                    Some(_) => {
                        span_bug!(expr.span, "label wasn't mapped to a label def!");
                    }
                }

                // visit `break` argument if any
                visit::walk_expr(self, expr);
            }

            ExprKind::IfLet(ref pattern, ref subexpression, ref if_block, ref optional_else) => {
                self.visit_expr(subexpression);

                self.ribs[ValueNS].push(Rib::new(NormalRibKind));
                self.resolve_pattern(pattern, PatternSource::IfLet, &mut FxHashMap());
                self.visit_block(if_block);
                self.ribs[ValueNS].pop();

                optional_else.as_ref().map(|expr| self.visit_expr(expr));
            }

            ExprKind::Loop(ref block, label) => self.resolve_labeled_block(label, expr.id, &block),

            ExprKind::While(ref subexpression, ref block, label) => {
                self.visit_expr(subexpression);
                self.resolve_labeled_block(label, expr.id, &block);
            }

            ExprKind::WhileLet(ref pattern, ref subexpression, ref block, label) => {
                self.visit_expr(subexpression);
                self.ribs[ValueNS].push(Rib::new(NormalRibKind));
                self.resolve_pattern(pattern, PatternSource::WhileLet, &mut FxHashMap());

                self.resolve_labeled_block(label, expr.id, block);

                self.ribs[ValueNS].pop();
            }

            ExprKind::ForLoop(ref pattern, ref subexpression, ref block, label) => {
                self.visit_expr(subexpression);
                self.ribs[ValueNS].push(Rib::new(NormalRibKind));
                self.resolve_pattern(pattern, PatternSource::For, &mut FxHashMap());

                self.resolve_labeled_block(label, expr.id, block);

                self.ribs[ValueNS].pop();
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
            ExprKind::MethodCall(name, ..) => {
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

        let mut found_traits = Vec::new();
        // Look for the current trait.
        if let Some((trait_def_id, _)) = self.current_trait_ref {
            if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                found_traits.push(TraitCandidate { def_id: trait_def_id, import_id: None });
            }
        }

        let mut search_module = self.current_module;
        loop {
            self.get_traits_in_module_containing_item(name, search_module, &mut found_traits);
            match search_module.kind {
                ModuleKind::Block(..) => search_module = search_module.parent.unwrap(),
                _ => break,
            }
        }

        if let Some(prelude) = self.prelude {
            if !search_module.no_implicit_prelude {
                self.get_traits_in_module_containing_item(name, prelude, &mut found_traits);
            }
        }

        found_traits
    }

    fn get_traits_in_module_containing_item(&mut self,
                                            name: Name,
                                            module: Module,
                                            found_traits: &mut Vec<TraitCandidate>) {
        let mut traits = module.traits.borrow_mut();
        if traits.is_none() {
            let mut collected_traits = Vec::new();
            module.for_each_child(|name, ns, binding| {
                if ns != TypeNS { return }
                if let Def::Trait(_) = binding.def() {
                    collected_traits.push((name, binding));
                }
            });
            *traits = Some(collected_traits.into_boxed_slice());
        }

        for &(trait_name, binding) in traits.as_ref().unwrap().iter() {
            let trait_def_id = binding.def().def_id();
            if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                let import_id = match binding.kind {
                    NameBindingKind::Import { directive, .. } => {
                        self.maybe_unused_trait_imports.insert(directive.id);
                        self.add_to_glob_map(directive.id, trait_name);
                        Some(directive.id)
                    }
                    _ => None,
                };
                found_traits.push(TraitCandidate { def_id: trait_def_id, import_id: import_id });
            }
        }
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

            in_module.for_each_child(|ident, ns, name_binding| {

                // avoid imports entirely
                if name_binding.is_import() && !name_binding.is_extern_crate() { return; }

                // collect results based on the filter function
                if ident.name == lookup_name && ns == namespace {
                    if filter_fn(name_binding.def()) {
                        // create the path
                        let span = name_binding.span;
                        let mut segms = path_segments.clone();
                        segms.push(ident.into());
                        let path = Path {
                            span: span,
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
                    let mut path_segments = path_segments.clone();
                    path_segments.push(ident.into());

                    if !in_module_is_extern || name_binding.vis == ty::Visibility::Public {
                        // add the module to the lookup
                        let is_extern = in_module_is_extern || name_binding.is_extern_crate();
                        if !worklist.iter().any(|&(m, ..)| m.def() == module.def()) {
                            worklist.push((module, path_segments, is_extern));
                        }
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
        let (segments, span, id) = match *vis {
            ast::Visibility::Public => return ty::Visibility::Public,
            ast::Visibility::Crate(_) => {
                return ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX));
            }
            ast::Visibility::Restricted { ref path, id } => (&path.segments, path.span, id),
            ast::Visibility::Inherited => {
                return ty::Visibility::Restricted(self.current_module.normal_ancestor_id);
            }
        };

        let path: Vec<_> = segments.iter().map(|seg| seg.identifier).collect();
        let mut path_resolution = err_path_resolution();
        let vis = match self.resolve_path(&path, None, Some(span)) {
            PathResult::Module(module) => {
                path_resolution = PathResolution::new(module.def().unwrap());
                ty::Visibility::Restricted(module.normal_ancestor_id)
            }
            PathResult::Failed(msg, _) => {
                self.session.span_err(span, &format!("failed to resolve module path. {}", msg));
                ty::Visibility::Public
            }
            _ => ty::Visibility::Public,
        };
        self.def_map.insert(id, path_resolution);
        if !self.is_accessible(vis) {
            let msg = format!("visibilities can only be restricted to ancestor modules");
            self.session.span_err(span, &msg);
        }
        vis
    }

    fn is_accessible(&self, vis: ty::Visibility) -> bool {
        vis.is_accessible_from(self.current_module.normal_ancestor_id, self)
    }

    fn is_accessible_from(&self, vis: ty::Visibility, module: Module<'a>) -> bool {
        vis.is_accessible_from(module.normal_ancestor_id, self)
    }

    fn report_errors(&mut self) {
        self.report_shadowing_errors();
        let mut reported_spans = FxHashSet();

        for &AmbiguityError { span, name, b1, b2, lexical, legacy } in &self.ambiguity_errors {
            if !reported_spans.insert(span) { continue }
            let participle = |binding: &NameBinding| {
                if binding.is_import() { "imported" } else { "defined" }
            };
            let msg1 = format!("`{}` could resolve to the name {} here", name, participle(b1));
            let msg2 = format!("`{}` could also resolve to the name {} here", name, participle(b2));
            let note = if !lexical && b1.is_glob_import() {
                format!("consider adding an explicit import of `{}` to disambiguate", name)
            } else if let Def::Macro(..) = b1.def() {
                format!("macro-expanded {} do not shadow",
                        if b1.is_import() { "macro imports" } else { "macros" })
            } else {
                format!("macro-expanded {} do not shadow when used in a macro invocation path",
                        if b1.is_import() { "imports" } else { "items" })
            };
            if legacy {
                let id = match b2.kind {
                    NameBindingKind::Import { directive, .. } => directive.id,
                    _ => unreachable!(),
                };
                let mut span = MultiSpan::from_span(span);
                span.push_span_label(b1.span, msg1);
                span.push_span_label(b2.span, msg2);
                let msg = format!("`{}` is ambiguous", name);
                self.session.add_lint(lint::builtin::LEGACY_IMPORTS, id, span, msg);
            } else {
                self.session.struct_span_err(span, &format!("`{}` is ambiguous", name))
                    .span_note(b1.span, &msg1)
                    .span_note(b2.span, &msg2)
                    .note(&note)
                    .emit();
            }
        }

        for &PrivacyError(span, name, binding) in &self.privacy_errors {
            if !reported_spans.insert(span) { continue }
            if binding.is_extern_crate() {
                // Warn when using an inaccessible extern crate.
                let node_id = match binding.kind {
                    NameBindingKind::Import { directive, .. } => directive.id,
                    _ => unreachable!(),
                };
                let msg = format!("extern crate `{}` is private", name);
                self.session.add_lint(lint::builtin::INACCESSIBLE_EXTERN_CRATE, node_id, span, msg);
            } else {
                let def = binding.def();
                self.session.span_err(span, &format!("{} `{}` is private", def.kind_name(), name));
            }
        }
    }

    fn report_shadowing_errors(&mut self) {
        for (name, scope) in replace(&mut self.lexical_macro_resolutions, Vec::new()) {
            self.resolve_legacy_scope(scope, name, true);
        }

        let mut reported_errors = FxHashSet();
        for binding in replace(&mut self.disallowed_shadowing, Vec::new()) {
            if self.resolve_legacy_scope(&binding.parent, binding.name, false).is_some() &&
               reported_errors.insert((binding.name, binding.span)) {
                let msg = format!("`{}` is already in scope", binding.name);
                self.session.struct_span_err(binding.span, &msg)
                    .note("macro-expanded `macro_rules!`s may not shadow \
                           existing macros (see RFC 1560)")
                    .emit();
            }
        }
    }

    fn report_conflict(&mut self,
                       parent: Module,
                       ident: Ident,
                       ns: Namespace,
                       binding: &NameBinding,
                       old_binding: &NameBinding) {
        // Error on the second of two conflicting names
        if old_binding.span.lo > binding.span.lo {
            return self.report_conflict(parent, ident, ns, old_binding, binding);
        }

        let container = match parent.kind {
            ModuleKind::Def(Def::Mod(_), _) => "module",
            ModuleKind::Def(Def::Trait(_), _) => "trait",
            ModuleKind::Block(..) => "block",
            _ => "enum",
        };

        let (participle, noun) = match old_binding.is_import() {
            true => ("imported", "import"),
            false => ("defined", "definition"),
        };

        let (name, span) = (ident.name, binding.span);

        if let Some(s) = self.name_already_seen.get(&name) {
            if s == &span {
                return;
            }
        }

        let msg = {
            let kind = match (ns, old_binding.module()) {
                (ValueNS, _) => "a value",
                (MacroNS, _) => "a macro",
                (TypeNS, _) if old_binding.is_extern_crate() => "an extern crate",
                (TypeNS, Some(module)) if module.is_normal() => "a module",
                (TypeNS, Some(module)) if module.is_trait() => "a trait",
                (TypeNS, _) => "a type",
            };
            format!("{} named `{}` has already been {} in this {}",
                    kind, name, participle, container)
        };

        let mut err = match (old_binding.is_extern_crate(), binding.is_extern_crate()) {
            (true, true) => struct_span_err!(self.session, span, E0259, "{}", msg),
            (true, _) | (_, true) => match binding.is_import() && old_binding.is_import() {
                true => struct_span_err!(self.session, span, E0254, "{}", msg),
                false => struct_span_err!(self.session, span, E0260, "{}", msg),
            },
            _ => match (old_binding.is_import(), binding.is_import()) {
                (false, false) => struct_span_err!(self.session, span, E0428, "{}", msg),
                (true, true) => struct_span_err!(self.session, span, E0252, "{}", msg),
                _ => struct_span_err!(self.session, span, E0255, "{}", msg),
            },
        };

        err.span_label(span, &format!("`{}` already {}", name, participle));
        if old_binding.span != syntax_pos::DUMMY_SP {
            err.span_label(old_binding.span, &format!("previous {} of `{}` here", noun, name));
        }
        err.emit();
        self.name_already_seen.insert(name, span);
    }
}

fn names_to_string(names: &[Ident]) -> String {
    let mut result = String::new();
    for (i, ident) in names.iter().enumerate() {
        if i > 0 {
            result.push_str("::");
        }
        if ident.name != keywords::CrateRoot.name() {
            result.push_str(&ident.name.as_str());
        }
    }
    result
}

fn path_names_to_string(path: &Path, depth: usize) -> String {
    let names: Vec<_> =
        path.segments[..path.segments.len() - depth].iter().map(|seg| seg.identifier).collect();
    names_to_string(&names)
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

    fn collect_mod(names: &mut Vec<Ident>, module: Module) {
        if let ModuleKind::Def(_, name) = module.kind {
            if let Some(parent) = module.parent {
                names.push(Ident::with_empty_ctxt(name));
                collect_mod(names, parent);
            }
        } else {
            // danger, shouldn't be ident?
            names.push(Ident::from_str("<opaque>"));
            collect_mod(names, module.parent.unwrap());
        }
    }
    collect_mod(&mut names, module);

    if names.is_empty() {
        return "???".to_string();
    }
    names_to_string(&names.into_iter().rev().collect::<Vec<_>>())
}

fn err_path_resolution() -> PathResolution {
    PathResolution::new(Def::Err)
}

#[derive(PartialEq,Copy, Clone)]
pub enum MakeGlobMap {
    Yes,
    No,
}

__build_diagnostic_array! { librustc_resolve, DIAGNOSTICS }
