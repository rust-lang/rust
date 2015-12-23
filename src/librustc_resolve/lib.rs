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

#![feature(associated_consts)]
#![feature(borrow_state)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
#[macro_use]
#[no_link]
extern crate rustc_bitflags;
extern crate rustc_front;
extern crate rustc;

use self::PatternBindingMode::*;
use self::Namespace::*;
use self::NamespaceResult::*;
use self::ResolveResult::*;
use self::FallbackSuggestion::*;
use self::TypeParameters::*;
use self::RibKind::*;
use self::UseLexicalScopeFlag::*;
use self::ModulePrefixResult::*;
use self::AssocItemResolveResult::*;
use self::NameSearchType::*;
use self::BareIdentifierPatternResolution::*;
use self::ParentLink::*;
use self::FallbackChecks::*;

use rustc::front::map as hir_map;
use rustc::session::Session;
use rustc::lint;
use rustc::middle::cstore::{CrateStore, DefLike, DlDef};
use rustc::middle::def::*;
use rustc::middle::def_id::DefId;
use rustc::middle::pat_util::pat_bindings;
use rustc::middle::privacy::*;
use rustc::middle::subst::{ParamSpace, FnSpace, TypeSpace};
use rustc::middle::ty::{Freevar, FreevarMap, TraitMap, GlobMap};
use rustc::util::nodemap::{NodeMap, DefIdSet, FnvHashMap};

use syntax::ast;
use syntax::ast::{CRATE_NODE_ID, Name, NodeId, CrateNum, TyIs, TyI8, TyI16, TyI32, TyI64};
use syntax::ast::{TyUs, TyU8, TyU16, TyU32, TyU64, TyF64, TyF32};
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{self, Span, Pos};
use syntax::errors::DiagnosticBuilder;
use syntax::parse::token::{self, special_names, special_idents};
use syntax::util::lev_distance::find_best_match_for_name;

use rustc_front::intravisit::{self, FnKind, Visitor};
use rustc_front::hir;
use rustc_front::hir::{Arm, BindByRef, BindByValue, BindingMode, Block};
use rustc_front::hir::Crate;
use rustc_front::hir::{Expr, ExprAgain, ExprBreak, ExprField};
use rustc_front::hir::{ExprLoop, ExprWhile, ExprMethodCall};
use rustc_front::hir::{ExprPath, ExprStruct, FnDecl};
use rustc_front::hir::{ForeignItemFn, ForeignItemStatic, Generics};
use rustc_front::hir::{ImplItem, Item, ItemConst, ItemEnum, ItemExternCrate};
use rustc_front::hir::{ItemFn, ItemForeignMod, ItemImpl, ItemMod, ItemStatic, ItemDefaultImpl};
use rustc_front::hir::{ItemStruct, ItemTrait, ItemTy, ItemUse};
use rustc_front::hir::Local;
use rustc_front::hir::{Pat, PatEnum, PatIdent, PatLit, PatQPath};
use rustc_front::hir::{PatRange, PatStruct, Path, PrimTy};
use rustc_front::hir::{TraitRef, Ty, TyBool, TyChar, TyFloat, TyInt};
use rustc_front::hir::{TyRptr, TyStr, TyUint, TyPath, TyPtr};
use rustc_front::util::walk_pat;

use std::collections::{HashMap, HashSet};
use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem::replace;
use std::rc::{Rc, Weak};

use resolve_imports::{Target, ImportDirective, ImportResolutionPerNamespace};
use resolve_imports::Shadowable;

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostics;

mod check_unused;
mod record_exports;
mod build_reduced_graph;
mod resolve_imports;

// Perform the callback, not walking deeper if the return is true
macro_rules! execute_callback {
    ($node: expr, $walker: expr) => (
        if let Some(ref callback) = $walker.callback {
            if callback($node, &mut $walker.resolved) {
                return;
            }
        }
    )
}

enum SuggestionType {
    Macro(String),
    Function(token::InternedString),
    NotFound,
}

pub enum ResolutionError<'a> {
    /// error E0401: can't use type parameters from outer function
    TypeParametersFromOuterFunction,
    /// error E0402: cannot use an outer type parameter in this context
    OuterTypeParameterContext,
    /// error E0403: the name is already used for a type parameter in this type parameter list
    NameAlreadyUsedInTypeParameterList(Name),
    /// error E0404: is not a trait
    IsNotATrait(&'a str),
    /// error E0405: use of undeclared trait name
    UndeclaredTraitName(&'a str),
    /// error E0406: undeclared associated type
    UndeclaredAssociatedType,
    /// error E0407: method is not a member of trait
    MethodNotMemberOfTrait(Name, &'a str),
    /// error E0437: type is not a member of trait
    TypeNotMemberOfTrait(Name, &'a str),
    /// error E0438: const is not a member of trait
    ConstNotMemberOfTrait(Name, &'a str),
    /// error E0408: variable `{}` from pattern #1 is not bound in pattern
    VariableNotBoundInPattern(Name, usize),
    /// error E0409: variable is bound with different mode in pattern #{} than in pattern #1
    VariableBoundWithDifferentMode(Name, usize),
    /// error E0410: variable from pattern is not bound in pattern #1
    VariableNotBoundInParentPattern(Name, usize),
    /// error E0411: use of `Self` outside of an impl or trait
    SelfUsedOutsideImplOrTrait,
    /// error E0412: use of undeclared
    UseOfUndeclared(&'a str, &'a str),
    /// error E0413: declaration shadows an enum variant or unit-like struct in scope
    DeclarationShadowsEnumVariantOrUnitLikeStruct(Name),
    /// error E0414: only irrefutable patterns allowed here
    OnlyIrrefutablePatternsAllowedHere(DefId, Name),
    /// error E0415: identifier is bound more than once in this parameter list
    IdentifierBoundMoreThanOnceInParameterList(&'a str),
    /// error E0416: identifier is bound more than once in the same pattern
    IdentifierBoundMoreThanOnceInSamePattern(&'a str),
    /// error E0417: static variables cannot be referenced in a pattern
    StaticVariableReference,
    /// error E0418: is not an enum variant, struct or const
    NotAnEnumVariantStructOrConst(&'a str),
    /// error E0419: unresolved enum variant, struct or const
    UnresolvedEnumVariantStructOrConst(&'a str),
    /// error E0420: is not an associated const
    NotAnAssociatedConst(&'a str),
    /// error E0421: unresolved associated const
    UnresolvedAssociatedConst(&'a str),
    /// error E0422: does not name a struct
    DoesNotNameAStruct(&'a str),
    /// error E0423: is a struct variant name, but this expression uses it like a function name
    StructVariantUsedAsFunction(&'a str),
    /// error E0424: `self` is not available in a static method
    SelfNotAvailableInStaticMethod,
    /// error E0425: unresolved name
    UnresolvedName(&'a str, &'a str, UnresolvedNameContext),
    /// error E0426: use of undeclared label
    UndeclaredLabel(&'a str),
    /// error E0427: cannot use `ref` binding mode with ...
    CannotUseRefBindingModeWith(&'a str),
    /// error E0428: duplicate definition
    DuplicateDefinition(&'a str, Name),
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
}

/// Context of where `ResolutionError::UnresolvedName` arose.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum UnresolvedNameContext {
    /// `PathIsMod(id)` indicates that a given path, used in
    /// expression context, actually resolved to a module rather than
    /// a value. The `id` attached to the variant is the node id of
    /// the erroneous path expression.
    PathIsMod(ast::NodeId),

    /// `Other` means we have no extra information about the context
    /// of the unresolved name error. (Maybe we could eliminate all
    /// such cases; but for now, this is an information-free default.)
    Other,
}

fn resolve_error<'b, 'a: 'b, 'tcx: 'a>(resolver: &'b Resolver<'a, 'tcx>,
                                       span: syntax::codemap::Span,
                                       resolution_error: ResolutionError<'b>) {
    resolve_struct_error(resolver, span, resolution_error).emit();
}

fn resolve_struct_error<'b, 'a: 'b, 'tcx: 'a>(resolver: &'b Resolver<'a, 'tcx>,
                                              span: syntax::codemap::Span,
                                              resolution_error: ResolutionError<'b>)
                                              -> DiagnosticBuilder<'a> {
    if !resolver.emit_errors {
        return resolver.session.diagnostic().struct_dummy();
    }

    match resolution_error {
        ResolutionError::TypeParametersFromOuterFunction => {
            struct_span_err!(resolver.session,
                             span,
                             E0401,
                             "can't use type parameters from outer function; try using a local \
                              type parameter instead")
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
        ResolutionError::UndeclaredTraitName(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0405,
                             "use of undeclared trait name `{}`",
                             name)
        }
        ResolutionError::UndeclaredAssociatedType => {
            struct_span_err!(resolver.session, span, E0406, "undeclared associated type")
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
        ResolutionError::VariableNotBoundInPattern(variable_name, pattern_number) => {
            struct_span_err!(resolver.session,
                             span,
                             E0408,
                             "variable `{}` from pattern #1 is not bound in pattern #{}",
                             variable_name,
                             pattern_number)
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
        ResolutionError::VariableNotBoundInParentPattern(variable_name, pattern_number) => {
            struct_span_err!(resolver.session,
                             span,
                             E0410,
                             "variable `{}` from pattern #{} is not bound in pattern #1",
                             variable_name,
                             pattern_number)
        }
        ResolutionError::SelfUsedOutsideImplOrTrait => {
            struct_span_err!(resolver.session,
                             span,
                             E0411,
                             "use of `Self` outside of an impl or trait")
        }
        ResolutionError::UseOfUndeclared(kind, name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0412,
                             "use of undeclared {} `{}`",
                             kind,
                             name)
        }
        ResolutionError::DeclarationShadowsEnumVariantOrUnitLikeStruct(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0413,
                             "declaration of `{}` shadows an enum variant \
                              or unit-like struct in scope",
                             name)
        }
        ResolutionError::OnlyIrrefutablePatternsAllowedHere(did, name) => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0414,
                                           "only irrefutable patterns allowed here");
            err.span_note(span,
                          "there already is a constant in scope sharing the same \
                           name as this pattern");
            if let Some(sp) = resolver.ast_map.span_if_local(did) {
                err.span_note(sp, "constant defined here");
            }
            if let Some(directive) = resolver.current_module
                                             .import_resolutions
                                             .borrow()
                                             .get(&name) {
                let item = resolver.ast_map.expect_item(directive.value_ns.id);
                err.span_note(item.span, "constant imported here");
            }
            err
        }
        ResolutionError::IdentifierBoundMoreThanOnceInParameterList(identifier) => {
            struct_span_err!(resolver.session,
                             span,
                             E0415,
                             "identifier `{}` is bound more than once in this parameter list",
                             identifier)
        }
        ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(identifier) => {
            struct_span_err!(resolver.session,
                             span,
                             E0416,
                             "identifier `{}` is bound more than once in the same pattern",
                             identifier)
        }
        ResolutionError::StaticVariableReference => {
            struct_span_err!(resolver.session,
                             span,
                             E0417,
                             "static variables cannot be referenced in a pattern, use a \
                              `const` instead")
        }
        ResolutionError::NotAnEnumVariantStructOrConst(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0418,
                             "`{}` is not an enum variant, struct or const",
                             name)
        }
        ResolutionError::UnresolvedEnumVariantStructOrConst(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0419,
                             "unresolved enum variant, struct or const `{}`",
                             name)
        }
        ResolutionError::NotAnAssociatedConst(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0420,
                             "`{}` is not an associated const",
                             name)
        }
        ResolutionError::UnresolvedAssociatedConst(name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0421,
                             "unresolved associated const `{}`",
                             name)
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
        ResolutionError::UnresolvedName(path, msg, context) => {
            let mut err = struct_span_err!(resolver.session,
                                           span,
                                           E0425,
                                           "unresolved name `{}`{}",
                                           path,
                                           msg);

            match context {
                UnresolvedNameContext::Other => {} // no help available
                UnresolvedNameContext::PathIsMod(id) => {
                    let mut help_msg = String::new();
                    let parent_id = resolver.ast_map.get_parent_node(id);
                    if let Some(hir_map::Node::NodeExpr(e)) = resolver.ast_map.find(parent_id) {
                        match e.node {
                            ExprField(_, ident) => {
                                help_msg = format!("To reference an item from the \
                                                    `{module}` module, use \
                                                    `{module}::{ident}`",
                                                   module = &*path,
                                                   ident = ident.node);
                            }

                            ExprMethodCall(ident, _, _) => {
                                help_msg = format!("To call a function from the \
                                                    `{module}` module, use \
                                                    `{module}::{ident}(..)`",
                                                   module = &*path,
                                                   ident = ident.node);
                            }

                            _ => {} // no help available
                        }
                    }

                    if !help_msg.is_empty() {
                        err.fileline_help(span, &help_msg);
                    }
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
        ResolutionError::CannotUseRefBindingModeWith(descr) => {
            struct_span_err!(resolver.session,
                             span,
                             E0427,
                             "cannot use `ref` binding mode with {}",
                             descr)
        }
        ResolutionError::DuplicateDefinition(namespace, name) => {
            struct_span_err!(resolver.session,
                             span,
                             E0428,
                             "duplicate definition of {} `{}`",
                             namespace,
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
    }
}

#[derive(Copy, Clone)]
struct BindingInfo {
    span: Span,
    binding_mode: BindingMode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = HashMap<Name, BindingInfo>;

#[derive(Copy, Clone, PartialEq)]
enum PatternBindingMode {
    RefutableMode,
    LocalIrrefutableMode,
    ArgumentIrrefutableMode,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Namespace {
    TypeNS,
    ValueNS,
}

/// A NamespaceResult represents the result of resolving an import in
/// a particular namespace. The result is either definitely-resolved,
/// definitely- unresolved, or unknown.
#[derive(Clone)]
enum NamespaceResult {
    /// Means that resolve hasn't gathered enough information yet to determine
    /// whether the name is bound in this namespace. (That is, it hasn't
    /// resolved all `use` directives yet.)
    UnknownResult,
    /// Means that resolve has determined that the name is definitely
    /// not bound in the namespace.
    UnboundResult,
    /// Means that resolve has determined that the name is bound in the Module
    /// argument, and specified by the NameBinding argument.
    BoundResult(Rc<Module>, NameBinding),
}

impl NamespaceResult {
    fn is_unknown(&self) -> bool {
        match *self {
            UnknownResult => true,
            _ => false,
        }
    }
    fn is_unbound(&self) -> bool {
        match *self {
            UnboundResult => true,
            _ => false,
        }
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for Resolver<'a, 'tcx> {
    fn visit_nested_item(&mut self, item: hir::ItemId) {
        self.visit_item(self.ast_map.expect_item(item.id))
    }
    fn visit_item(&mut self, item: &Item) {
        execute_callback!(hir_map::Node::NodeItem(item), self);
        self.resolve_item(item);
    }
    fn visit_arm(&mut self, arm: &Arm) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &Block) {
        execute_callback!(hir_map::Node::NodeBlock(block), self);
        self.resolve_block(block);
    }
    fn visit_expr(&mut self, expr: &Expr) {
        execute_callback!(hir_map::Node::NodeExpr(expr), self);
        self.resolve_expr(expr);
    }
    fn visit_local(&mut self, local: &Local) {
        execute_callback!(hir_map::Node::NodeLocal(&*local.pat), self);
        self.resolve_local(local);
    }
    fn visit_ty(&mut self, ty: &Ty) {
        self.resolve_type(ty);
    }
    fn visit_generics(&mut self, generics: &Generics) {
        self.resolve_generics(generics);
    }
    fn visit_poly_trait_ref(&mut self, tref: &hir::PolyTraitRef, m: &hir::TraitBoundModifier) {
        match self.resolve_trait_reference(tref.trait_ref.ref_id, &tref.trait_ref.path, 0) {
            Ok(def) => self.record_def(tref.trait_ref.ref_id, def),
            Err(_) => {
                // error already reported
                self.record_def(tref.trait_ref.ref_id, err_path_resolution())
            }
        }
        intravisit::walk_poly_trait_ref(self, tref, m);
    }
    fn visit_variant(&mut self,
                     variant: &hir::Variant,
                     generics: &Generics,
                     item_id: ast::NodeId) {
        execute_callback!(hir_map::Node::NodeVariant(variant), self);
        if let Some(ref dis_expr) = variant.node.disr_expr {
            // resolve the discriminator expr as a constant
            self.with_constant_rib(|this| {
                this.visit_expr(dis_expr);
            });
        }

        // `intravisit::walk_variant` without the discriminant expression.
        self.visit_variant_data(&variant.node.data,
                                variant.node.name,
                                generics,
                                item_id,
                                variant.span);
    }
    fn visit_foreign_item(&mut self, foreign_item: &hir::ForeignItem) {
        execute_callback!(hir_map::Node::NodeForeignItem(foreign_item), self);
        let type_parameters = match foreign_item.node {
            ForeignItemFn(_, ref generics) => {
                HasTypeParameters(generics, FnSpace, ItemRibKind)
            }
            ForeignItemStatic(..) => NoTypeParameters,
        };
        self.with_type_parameter_rib(type_parameters, |this| {
            intravisit::walk_foreign_item(this, foreign_item);
        });
    }
    fn visit_fn(&mut self,
                function_kind: FnKind<'v>,
                declaration: &'v FnDecl,
                block: &'v Block,
                _: Span,
                node_id: NodeId) {
        let rib_kind = match function_kind {
            FnKind::ItemFn(_, generics, _, _, _, _) => {
                self.visit_generics(generics);
                ItemRibKind
            }
            FnKind::Method(_, sig, _) => {
                self.visit_generics(&sig.generics);
                self.visit_explicit_self(&sig.explicit_self);
                MethodRibKind
            }
            FnKind::Closure => ClosureRibKind(node_id),
        };
        self.resolve_function(rib_kind, declaration, block);
    }
}

type ErrorMessage = Option<(Span, String)>;

enum ResolveResult<T> {
    Failed(ErrorMessage), // Failed to resolve the name, optional helpful error message.
    Indeterminate, // Couldn't determine due to unresolved globs.
    Success(T), // Successfully resolved the import.
}

impl<T> ResolveResult<T> {
    fn success(&self) -> bool {
        match *self {
            Success(_) => true,
            _ => false,
        }
    }
}

enum FallbackSuggestion {
    NoSuggestion,
    Field,
    Method,
    TraitItem,
    StaticMethod(String),
    TraitMethod(String),
}

#[derive(Copy, Clone)]
enum TypeParameters<'a> {
    NoTypeParameters,
    HasTypeParameters(// Type parameters.
                      &'a Generics,

                      // Identifies the things that these parameters
                      // were declared on (type, fn, etc)
                      ParamSpace,

                      // The kind of the rib used for type parameters.
                      RibKind),
}

// The rib kind controls the translation of local
// definitions (`DefLocal`) to upvars (`DefUpvar`).
#[derive(Copy, Clone, Debug)]
enum RibKind {
    // No translation needs to be applied.
    NormalRibKind,

    // We passed through a closure scope at the given node ID.
    // Translate upvars as appropriate.
    ClosureRibKind(NodeId /* func id */),

    // We passed through an impl or trait and are now in one of its
    // methods. Allow references to ty params that impl or trait
    // binds. Disallow any other upvars (including other ty params that are
    // upvars).
    MethodRibKind,

    // We passed through an item scope. Disallow upvars.
    ItemRibKind,

    // We're in a constant item. Can't refer to dynamic stuff.
    ConstantItemRibKind,
}

#[derive(Copy, Clone)]
enum UseLexicalScopeFlag {
    DontUseLexicalScope,
    UseLexicalScope,
}

enum ModulePrefixResult {
    NoPrefixFound,
    PrefixFound(Rc<Module>, usize),
}

#[derive(Copy, Clone)]
enum AssocItemResolveResult {
    /// Syntax such as `<T>::item`, which can't be resolved until type
    /// checking.
    TypecheckRequired,
    /// We should have been able to resolve the associated item.
    ResolveAttempt(Option<PathResolution>),
}

#[derive(Copy, Clone, PartialEq)]
enum NameSearchType {
    /// We're doing a name search in order to resolve a `use` directive.
    ImportSearch,

    /// We're doing a name search in order to resolve a path type, a path
    /// expression, or a path pattern.
    PathSearch,
}

#[derive(Copy, Clone)]
enum BareIdentifierPatternResolution {
    FoundStructOrEnumVariant(Def, LastPrivate),
    FoundConst(Def, LastPrivate, Name),
    BareIdentifierPatternUnresolved,
}

/// One local scope.
#[derive(Debug)]
struct Rib {
    bindings: HashMap<Name, DefLike>,
    kind: RibKind,
}

impl Rib {
    fn new(kind: RibKind) -> Rib {
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

/// The link from a module up to its nearest parent node.
#[derive(Clone,Debug)]
enum ParentLink {
    NoParentLink,
    ModuleParentLink(Weak<Module>, Name),
    BlockParentLink(Weak<Module>, NodeId),
}

/// One node in the tree of modules.
pub struct Module {
    parent_link: ParentLink,
    def: Cell<Option<Def>>,
    is_public: bool,

    children: RefCell<HashMap<Name, NameBindings>>,
    imports: RefCell<Vec<ImportDirective>>,

    // The external module children of this node that were declared with
    // `extern crate`.
    external_module_children: RefCell<HashMap<Name, Rc<Module>>>,

    // The anonymous children of this node. Anonymous children are pseudo-
    // modules that are implicitly created around items contained within
    // blocks.
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
    anonymous_children: RefCell<NodeMap<Rc<Module>>>,

    // The status of resolving each import in this module.
    import_resolutions: RefCell<HashMap<Name, ImportResolutionPerNamespace>>,

    // The number of unresolved globs that this module exports.
    glob_count: Cell<usize>,

    // The number of unresolved pub imports (both regular and globs) in this module
    pub_count: Cell<usize>,

    // The number of unresolved pub glob imports in this module
    pub_glob_count: Cell<usize>,

    // The index of the import we're resolving.
    resolved_import_count: Cell<usize>,

    // Whether this module is populated. If not populated, any attempt to
    // access the children must be preceded with a
    // `populate_module_if_necessary` call.
    populated: Cell<bool>,
}

impl Module {
    fn new(parent_link: ParentLink,
           def: Option<Def>,
           external: bool,
           is_public: bool)
           -> Rc<Module> {
        Rc::new(Module {
            parent_link: parent_link,
            def: Cell::new(def),
            is_public: is_public,
            children: RefCell::new(HashMap::new()),
            imports: RefCell::new(Vec::new()),
            external_module_children: RefCell::new(HashMap::new()),
            anonymous_children: RefCell::new(NodeMap()),
            import_resolutions: RefCell::new(HashMap::new()),
            glob_count: Cell::new(0),
            pub_count: Cell::new(0),
            pub_glob_count: Cell::new(0),
            resolved_import_count: Cell::new(0),
            populated: Cell::new(!external),
        })
    }

    fn def_id(&self) -> Option<DefId> {
        self.def.get().as_ref().map(Def::def_id)
    }

    fn is_normal(&self) -> bool {
        match self.def.get() {
            Some(DefMod(_)) | Some(DefForeignMod(_)) => true,
            _ => false,
        }
    }

    fn is_trait(&self) -> bool {
        match self.def.get() {
            Some(DefTrait(_)) => true,
            _ => false,
        }
    }

    fn all_imports_resolved(&self) -> bool {
        if self.imports.borrow_state() == ::std::cell::BorrowState::Writing {
            // it is currently being resolved ! so nope
            false
        } else {
            self.imports.borrow().len() == self.resolved_import_count.get()
        }
    }
}

impl Module {
    pub fn inc_glob_count(&self) {
        self.glob_count.set(self.glob_count.get() + 1);
    }
    pub fn dec_glob_count(&self) {
        assert!(self.glob_count.get() > 0);
        self.glob_count.set(self.glob_count.get() - 1);
    }
    pub fn inc_pub_count(&self) {
        self.pub_count.set(self.pub_count.get() + 1);
    }
    pub fn dec_pub_count(&self) {
        assert!(self.pub_count.get() > 0);
        self.pub_count.set(self.pub_count.get() - 1);
    }
    pub fn inc_pub_glob_count(&self) {
        self.pub_glob_count.set(self.pub_glob_count.get() + 1);
    }
    pub fn dec_pub_glob_count(&self) {
        assert!(self.pub_glob_count.get() > 0);
        self.pub_glob_count.set(self.pub_glob_count.get() - 1);
    }
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "{:?}, {}",
               self.def,
               if self.is_public {
                   "public"
               } else {
                   "private"
               })
    }
}

bitflags! {
    #[derive(Debug)]
    flags DefModifiers: u8 {
        // Enum variants are always considered `PUBLIC`, this is needed for `use Enum::Variant`
        // or `use Enum::*` to work on private enums.
        const PUBLIC     = 1 << 0,
        const IMPORTABLE = 1 << 1,
        // Variants are considered `PUBLIC`, but some of them live in private enums.
        // We need to track them to prohibit reexports like `pub use PrivEnum::Variant`.
        const PRIVATE_VARIANT = 1 << 2,
    }
}

// Records a possibly-private value, type, or module definition.
#[derive(Debug)]
struct NsDef {
    modifiers: DefModifiers, // see note in ImportResolutionPerNamespace about how to use this
    def_or_module: DefOrModule,
    span: Option<Span>,
}

#[derive(Debug)]
enum DefOrModule {
    Def(Def),
    Module(Rc<Module>),
}

impl NsDef {
    fn create_from_module(module: Rc<Module>, span: Option<Span>) -> Self {
        let modifiers = if module.is_public {
            DefModifiers::PUBLIC
        } else {
            DefModifiers::empty()
        } | DefModifiers::IMPORTABLE;

        NsDef { modifiers: modifiers, def_or_module: DefOrModule::Module(module), span: span }
    }

    fn create_from_def(def: Def, modifiers: DefModifiers, span: Option<Span>) -> Self {
        NsDef { modifiers: modifiers, def_or_module: DefOrModule::Def(def), span: span }
    }

    fn module(&self) -> Option<Rc<Module>> {
        match self.def_or_module {
            DefOrModule::Module(ref module) => Some(module.clone()),
            DefOrModule::Def(_) => None,
        }
    }

    fn def(&self) -> Option<Def> {
        match self.def_or_module {
            DefOrModule::Def(def) => Some(def),
            DefOrModule::Module(ref module) => module.def.get(),
        }
    }
}

// Records at most one definition that a name in a namespace is bound to
#[derive(Clone,Debug)]
pub struct NameBinding(Rc<RefCell<Option<NsDef>>>);

impl NameBinding {
    fn new() -> Self {
        NameBinding(Rc::new(RefCell::new(None)))
    }

    fn create_from_module(module: Rc<Module>) -> Self {
        NameBinding(Rc::new(RefCell::new(Some(NsDef::create_from_module(module, None)))))
    }

    fn set(&self, ns_def: NsDef) {
        *self.0.borrow_mut() = Some(ns_def);
    }

    fn set_modifiers(&self, modifiers: DefModifiers) {
        if let Some(ref mut ns_def) = *self.0.borrow_mut() {
            ns_def.modifiers = modifiers
        }
    }

    fn borrow(&self) -> ::std::cell::Ref<Option<NsDef>> {
        self.0.borrow()
    }

    // Lifted versions of the NsDef methods and fields
    fn def(&self) -> Option<Def> {
        self.borrow().as_ref().and_then(NsDef::def)
    }
    fn module(&self) -> Option<Rc<Module>> {
        self.borrow().as_ref().and_then(NsDef::module)
    }
    fn span(&self) -> Option<Span> {
        self.borrow().as_ref().and_then(|def| def.span)
    }
    fn modifiers(&self) -> Option<DefModifiers> {
        self.borrow().as_ref().and_then(|def| Some(def.modifiers))
    }

    fn defined(&self) -> bool {
        self.borrow().is_some()
    }

    fn defined_with(&self, modifiers: DefModifiers) -> bool {
        self.modifiers().map(|m| m.contains(modifiers)).unwrap_or(false)
    }

    fn is_public(&self) -> bool {
        self.defined_with(DefModifiers::PUBLIC)
    }

    fn def_and_lp(&self) -> (Def, LastPrivate) {
        let def = self.def().unwrap();
        (def, LastMod(if self.is_public() { AllPublic } else { DependsOn(def.def_id()) }))
    }
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
#[derive(Clone,Debug)]
pub struct NameBindings {
    type_ns: NameBinding, // < Meaning in type namespace.
    value_ns: NameBinding, // < Meaning in value namespace.
}

impl ::std::ops::Index<Namespace> for NameBindings {
    type Output = NameBinding;
    fn index(&self, namespace: Namespace) -> &NameBinding {
        match namespace { TypeNS => &self.type_ns, ValueNS => &self.value_ns }
    }
}

impl NameBindings {
    fn new() -> NameBindings {
        NameBindings {
            type_ns: NameBinding::new(),
            value_ns: NameBinding::new(),
        }
    }

    /// Creates a new module in this set of name bindings.
    fn define_module(&self, module: Rc<Module>, sp: Span) {
        self.type_ns.set(NsDef::create_from_module(module, Some(sp)));
    }

    /// Records a type definition.
    fn define_type(&self, def: Def, sp: Span, modifiers: DefModifiers) {
        debug!("defining type for def {:?} with modifiers {:?}", def, modifiers);
        self.type_ns.set(NsDef::create_from_def(def, modifiers, Some(sp)));
    }

    /// Records a value definition.
    fn define_value(&self, def: Def, sp: Span, modifiers: DefModifiers) {
        debug!("defining value for def {:?} with modifiers {:?}", def, modifiers);
        self.value_ns.set(NsDef::create_from_def(def, modifiers, Some(sp)));
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
        table.intern("f32", TyFloat(TyF32));
        table.intern("f64", TyFloat(TyF64));
        table.intern("isize", TyInt(TyIs));
        table.intern("i8", TyInt(TyI8));
        table.intern("i16", TyInt(TyI16));
        table.intern("i32", TyInt(TyI32));
        table.intern("i64", TyInt(TyI64));
        table.intern("str", TyStr);
        table.intern("usize", TyUint(TyUs));
        table.intern("u8", TyUint(TyU8));
        table.intern("u16", TyUint(TyU16));
        table.intern("u32", TyUint(TyU32));
        table.intern("u64", TyUint(TyU64));

        table
    }

    fn intern(&mut self, string: &str, primitive_type: PrimTy) {
        self.primitive_types.insert(token::intern(string), primitive_type);
    }
}

/// The main resolver class.
pub struct Resolver<'a, 'tcx: 'a> {
    session: &'a Session,

    ast_map: &'a hir_map::Map<'tcx>,

    graph_root: Rc<Module>,

    trait_item_map: FnvHashMap<(Name, DefId), DefId>,

    structs: FnvHashMap<DefId, Vec<Name>>,

    // The number of imports that are currently unresolved.
    unresolved_imports: usize,

    // The module that represents the current item scope.
    current_module: Rc<Module>,

    // The current set of local scopes, for values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    value_ribs: Vec<Rib>,

    // The current set of local scopes, for types.
    type_ribs: Vec<Rib>,

    // The current set of local scopes, for labels.
    label_ribs: Vec<Rib>,

    // The trait that the current context can refer to.
    current_trait_ref: Option<(DefId, TraitRef)>,

    // The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    // The idents for the primitive types.
    primitive_type_table: PrimitiveTypeTable,

    def_map: RefCell<DefMap>,
    freevars: FreevarMap,
    freevars_seen: NodeMap<NodeMap<usize>>,
    export_map: ExportMap,
    trait_map: TraitMap,
    external_exports: ExternalExports,

    // Whether or not to print error messages. Can be set to true
    // when getting additional info for error message suggestions,
    // so as to avoid printing duplicate errors
    emit_errors: bool,

    make_glob_map: bool,
    // Maps imports to the names of items actually imported (this actually maps
    // all imports, but only glob imports are actually interesting).
    glob_map: GlobMap,

    used_imports: HashSet<(NodeId, Namespace)>,
    used_crates: HashSet<CrateNum>,

    // Callback function for intercepting walks
    callback: Option<Box<Fn(hir_map::Node, &mut bool) -> bool>>,
    // The intention is that the callback modifies this flag.
    // Once set, the resolver falls out of the walk, preserving the ribs.
    resolved: bool,
}

#[derive(PartialEq)]
enum FallbackChecks {
    Everything,
    OnlyTraitAndStatics,
}

impl<'a, 'tcx> Resolver<'a, 'tcx> {
    fn new(session: &'a Session,
           ast_map: &'a hir_map::Map<'tcx>,
           make_glob_map: MakeGlobMap)
           -> Resolver<'a, 'tcx> {
        let root_def_id = ast_map.local_def_id(CRATE_NODE_ID);
        let graph_root = Module::new(NoParentLink, Some(DefMod(root_def_id)), false, true);

        Resolver {
            session: session,

            ast_map: ast_map,

            // The outermost module has def ID 0; this is not reflected in the
            // AST.
            graph_root: graph_root.clone(),

            trait_item_map: FnvHashMap(),
            structs: FnvHashMap(),

            unresolved_imports: 0,

            current_module: graph_root,
            value_ribs: Vec::new(),
            type_ribs: Vec::new(),
            label_ribs: Vec::new(),

            current_trait_ref: None,
            current_self_type: None,

            primitive_type_table: PrimitiveTypeTable::new(),

            def_map: RefCell::new(NodeMap()),
            freevars: NodeMap(),
            freevars_seen: NodeMap(),
            export_map: NodeMap(),
            trait_map: NodeMap(),
            used_imports: HashSet::new(),
            used_crates: HashSet::new(),
            external_exports: DefIdSet(),

            emit_errors: true,
            make_glob_map: make_glob_map == MakeGlobMap::Yes,
            glob_map: HashMap::new(),

            callback: None,
            resolved: false,
        }
    }

    #[inline]
    fn record_import_use(&mut self, import_id: NodeId, name: Name) {
        if !self.make_glob_map {
            return;
        }
        if self.glob_map.contains_key(&import_id) {
            self.glob_map.get_mut(&import_id).unwrap().insert(name);
            return;
        }

        let mut new_set = HashSet::new();
        new_set.insert(name);
        self.glob_map.insert(import_id, new_set);
    }

    fn get_trait_name(&self, did: DefId) -> Name {
        if let Some(node_id) = self.ast_map.as_local_node_id(did) {
            self.ast_map.expect_item(node_id).name
        } else {
            self.session.cstore.item_name(did)
        }
    }

    /// Checks that the names of external crates don't collide with other
    /// external crates.
    fn check_for_conflicts_between_external_crates(&self,
                                                   module: &Module,
                                                   name: Name,
                                                   span: Span) {
        if module.external_module_children.borrow().contains_key(&name) {
            span_err!(self.session,
                      span,
                      E0259,
                      "an external crate named `{}` has already been imported into this module",
                      name);
        }
    }

    /// Checks that the names of items don't collide with external crates.
    fn check_for_conflicts_between_external_crates_and_items(&self,
                                                             module: &Module,
                                                             name: Name,
                                                             span: Span) {
        if module.external_module_children.borrow().contains_key(&name) {
            span_err!(self.session,
                      span,
                      E0260,
                      "the name `{}` conflicts with an external crate that has been imported \
                       into this module",
                      name);
        }
    }

    /// Resolves the given module path from the given root `module_`.
    fn resolve_module_path_from_root(&mut self,
                                     module_: Rc<Module>,
                                     module_path: &[Name],
                                     index: usize,
                                     span: Span,
                                     name_search_type: NameSearchType,
                                     lp: LastPrivate)
                                     -> ResolveResult<(Rc<Module>, LastPrivate)> {
        fn search_parent_externals(needle: Name, module: &Rc<Module>) -> Option<Rc<Module>> {
            match module.external_module_children.borrow().get(&needle) {
                Some(_) => Some(module.clone()),
                None => match module.parent_link {
                    ModuleParentLink(ref parent, _) => {
                        search_parent_externals(needle, &parent.upgrade().unwrap())
                    }
                    _ => None,
                },
            }
        }

        let mut search_module = module_;
        let mut index = index;
        let module_path_len = module_path.len();
        let mut closest_private = lp;

        // Resolve the module part of the path. This does not involve looking
        // upward though scope chains; we simply resolve names directly in
        // modules as we go.
        while index < module_path_len {
            let name = module_path[index];
            match self.resolve_name_in_module(search_module.clone(),
                                              name,
                                              TypeNS,
                                              name_search_type,
                                              false) {
                Failed(None) => {
                    let segment_name = name.as_str();
                    let module_name = module_to_string(&*search_module);
                    let mut span = span;
                    let msg = if "???" == &module_name[..] {
                        span.hi = span.lo + Pos::from_usize(segment_name.len());

                        match search_parent_externals(name, &self.current_module) {
                            Some(module) => {
                                let path_str = names_to_string(module_path);
                                let target_mod_str = module_to_string(&*module);
                                let current_mod_str = module_to_string(&*self.current_module);

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
                Success((target, used_proxy)) => {
                    // Check to see whether there are type bindings, and, if
                    // so, whether there is a module within.
                    if let Some(module_def) = target.binding.module() {
                        // track extern crates for unused_extern_crate lint
                        if let Some(did) = module_def.def_id() {
                            self.used_crates.insert(did.krate);
                        }

                        search_module = module_def;

                        // Keep track of the closest private module used
                        // when resolving this import chain.
                        if !used_proxy && !search_module.is_public {
                            if let Some(did) = search_module.def_id() {
                                closest_private = LastMod(DependsOn(did));
                            }
                        }
                    } else {
                        let msg = format!("Not a module `{}`", name);
                        return Failed(Some((span, msg)));
                    }
                }
            }

            index += 1;
        }

        return Success((search_module, closest_private));
    }

    /// Attempts to resolve the module part of an import directive or path
    /// rooted at the given module.
    ///
    /// On success, returns the resolved module, and the closest *private*
    /// module found to the destination when resolving this path.
    fn resolve_module_path(&mut self,
                           module_: Rc<Module>,
                           module_path: &[Name],
                           use_lexical_scope: UseLexicalScopeFlag,
                           span: Span,
                           name_search_type: NameSearchType)
                           -> ResolveResult<(Rc<Module>, LastPrivate)> {
        let module_path_len = module_path.len();
        assert!(module_path_len > 0);

        debug!("(resolving module path for import) processing `{}` rooted at `{}`",
               names_to_string(module_path),
               module_to_string(&*module_));

        // Resolve the module prefix, if any.
        let module_prefix_result = self.resolve_module_prefix(module_.clone(), module_path);

        let search_module;
        let start_index;
        let last_private;
        match module_prefix_result {
            Failed(None) => {
                let mpath = names_to_string(module_path);
                let mpath = &mpath[..];
                match mpath.rfind(':') {
                    Some(idx) => {
                        let msg = format!("Could not find `{}` in `{}`",
                                          // idx +- 1 to account for the
                                          // colons on either side
                                          &mpath[idx + 1..],
                                          &mpath[..idx - 1]);
                        return Failed(Some((span, msg)));
                    }
                    None => {
                        return Failed(None);
                    }
                }
            }
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
                        search_module = self.graph_root.clone();
                        start_index = 0;
                        last_private = LastMod(AllPublic);
                    }
                    UseLexicalScope => {
                        // This is not a crate-relative path. We resolve the
                        // first component of the path in the current lexical
                        // scope and then proceed to resolve below that.
                        match self.resolve_module_in_lexical_scope(module_, module_path[0]) {
                            Failed(err) => return Failed(err),
                            Indeterminate => {
                                debug!("(resolving module path for import) indeterminate; bailing");
                                return Indeterminate;
                            }
                            Success(containing_module) => {
                                search_module = containing_module;
                                start_index = 1;
                                last_private = LastMod(AllPublic);
                            }
                        }
                    }
                }
            }
            Success(PrefixFound(ref containing_module, index)) => {
                search_module = containing_module.clone();
                start_index = index;
                last_private = LastMod(DependsOn(containing_module.def_id()
                                                                  .unwrap()));
            }
        }

        self.resolve_module_path_from_root(search_module,
                                           module_path,
                                           start_index,
                                           span,
                                           name_search_type,
                                           last_private)
    }

    /// Invariant: This must only be called during main resolution, not during
    /// import resolution.
    fn resolve_item_in_lexical_scope(&mut self,
                                     module_: Rc<Module>,
                                     name: Name,
                                     namespace: Namespace,
                                     record_used: bool)
                                     -> ResolveResult<(Target, bool)> {
        debug!("(resolving item in lexical scope) resolving `{}` in namespace {:?} in `{}`",
               name,
               namespace,
               module_to_string(&*module_));

        // The current module node is handled specially. First, check for
        // its immediate children.
        build_reduced_graph::populate_module_if_necessary(self, &module_);

        match module_.children.borrow().get(&name) {
            Some(name_bindings) if name_bindings[namespace].defined() => {
                debug!("top name bindings succeeded");
                return Success((Target::new(module_.clone(),
                                            name_bindings[namespace].clone(),
                                            Shadowable::Never),
                                false));
            }
            Some(_) | None => {
                // Not found; continue.
            }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.
        if let Some(import_resolution) = module_.import_resolutions.borrow().get(&name) {
            match import_resolution[namespace].target.clone() {
                None => {
                    // Not found; continue.
                    debug!("(resolving item in lexical scope) found import resolution, but not \
                            in namespace {:?}",
                           namespace);
                }
                Some(target) => {
                    debug!("(resolving item in lexical scope) using import resolution");
                    // track used imports and extern crates as well
                    let id = import_resolution[namespace].id;
                    if record_used {
                        self.used_imports.insert((id, namespace));
                        self.record_import_use(id, name);
                        if let Some(DefId{krate: kid, ..}) = target.target_module.def_id() {
                            self.used_crates.insert(kid);
                        }
                    }
                    return Success((target, false));
                }
            }
        }

        // Search for external modules.
        if namespace == TypeNS {
            // FIXME (21114): In principle unclear `child` *has* to be lifted.
            let child = module_.external_module_children.borrow().get(&name).cloned();
            if let Some(module) = child {
                let name_binding = NameBinding::create_from_module(module);
                debug!("lower name bindings succeeded");
                return Success((Target::new(module_, name_binding, Shadowable::Never),
                                false));
            }
        }

        // Finally, proceed up the scope chain looking for parent modules.
        let mut search_module = module_;
        loop {
            // Go to the next parent.
            match search_module.parent_link.clone() {
                NoParentLink => {
                    // No more parents. This module was unresolved.
                    debug!("(resolving item in lexical scope) unresolved module");
                    return Failed(None);
                }
                ModuleParentLink(parent_module_node, _) => {
                    if search_module.is_normal() {
                        // We stop the search here.
                        debug!("(resolving item in lexical scope) unresolved module: not \
                                searching through module parents");
                            return Failed(None);
                    } else {
                        search_module = parent_module_node.upgrade().unwrap();
                    }
                }
                BlockParentLink(ref parent_module_node, _) => {
                    search_module = parent_module_node.upgrade().unwrap();
                }
            }

            // Resolve the name in the parent module.
            match self.resolve_name_in_module(search_module.clone(),
                                              name,
                                              namespace,
                                              PathSearch,
                                              true) {
                Failed(Some((span, msg))) => {
                    resolve_error(self, span, ResolutionError::FailedToResolve(&*msg));
                }
                Failed(None) => (), // Continue up the search chain.
                Indeterminate => {
                    // We couldn't see through the higher scope because of an
                    // unresolved import higher up. Bail.

                    debug!("(resolving item in lexical scope) indeterminate higher scope; bailing");
                    return Indeterminate;
                }
                Success((target, used_reexport)) => {
                    // We found the module.
                    debug!("(resolving item in lexical scope) found name in module, done");
                    return Success((target, used_reexport));
                }
            }
        }
    }

    /// Resolves a module name in the current lexical scope.
    fn resolve_module_in_lexical_scope(&mut self,
                                       module_: Rc<Module>,
                                       name: Name)
                                       -> ResolveResult<Rc<Module>> {
        // If this module is an anonymous module, resolve the item in the
        // lexical scope. Otherwise, resolve the item from the crate root.
        let resolve_result = self.resolve_item_in_lexical_scope(module_, name, TypeNS, true);
        match resolve_result {
            Success((target, _)) => {
                if let Some(module_def) = target.binding.module() {
                    return Success(module_def)
                } else {
                    debug!("!!! (resolving module in lexical scope) module \
                            wasn't actually a module!");
                    return Failed(None);
                }
            }
            Indeterminate => {
                debug!("(resolving module in lexical scope) indeterminate; bailing");
                return Indeterminate;
            }
            Failed(err) => {
                debug!("(resolving module in lexical scope) failed to resolve");
                return Failed(err);
            }
        }
    }

    /// Returns the nearest normal module parent of the given module.
    fn get_nearest_normal_module_parent(&mut self, module_: Rc<Module>) -> Option<Rc<Module>> {
        let mut module_ = module_;
        loop {
            match module_.parent_link.clone() {
                NoParentLink => return None,
                ModuleParentLink(new_module, _) |
                BlockParentLink(new_module, _) => {
                    let new_module = new_module.upgrade().unwrap();
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
    fn get_nearest_normal_module_parent_or_self(&mut self, module_: Rc<Module>) -> Rc<Module> {
        if module_.is_normal() {
            return module_;
        }
        match self.get_nearest_normal_module_parent(module_.clone()) {
            None => module_,
            Some(new_module) => new_module,
        }
    }

    /// Resolves a "module prefix". A module prefix is one or both of (a) `self::`;
    /// (b) some chain of `super::`.
    /// grammar: (SELF MOD_SEP ) ? (SUPER MOD_SEP) *
    fn resolve_module_prefix(&mut self,
                             module_: Rc<Module>,
                             module_path: &[Name])
                             -> ResolveResult<ModulePrefixResult> {
        // Start at the current module if we see `self` or `super`, or at the
        // top of the crate otherwise.
        let mut i = match &*module_path[0].as_str() {
            "self" => 1,
            "super" => 0,
            _ => return Success(NoPrefixFound),
        };
        let mut containing_module = self.get_nearest_normal_module_parent_or_self(module_);

        // Now loop through all the `super`s we find.
        while i < module_path.len() && "super" == module_path[i].as_str() {
            debug!("(resolving module prefix) resolving `super` at {}",
                   module_to_string(&*containing_module));
            match self.get_nearest_normal_module_parent(containing_module) {
                None => return Failed(None),
                Some(new_module) => {
                    containing_module = new_module;
                    i += 1;
                }
            }
        }

        debug!("(resolving module prefix) finished resolving prefix at {}",
               module_to_string(&*containing_module));

        return Success(PrefixFound(containing_module, i));
    }

    /// Attempts to resolve the supplied name in the given module for the
    /// given namespace. If successful, returns the target corresponding to
    /// the name.
    ///
    /// The boolean returned on success is an indicator of whether this lookup
    /// passed through a public re-export proxy.
    fn resolve_name_in_module(&mut self,
                              module_: Rc<Module>,
                              name: Name,
                              namespace: Namespace,
                              name_search_type: NameSearchType,
                              allow_private_imports: bool)
                              -> ResolveResult<(Target, bool)> {
        debug!("(resolving name in module) resolving `{}` in `{}`",
               name,
               module_to_string(&*module_));

        // First, check the direct children of the module.
        build_reduced_graph::populate_module_if_necessary(self, &module_);

        match module_.children.borrow().get(&name) {
            Some(name_bindings) if name_bindings[namespace].defined() => {
                debug!("(resolving name in module) found node as child");
                return Success((Target::new(module_.clone(),
                                            name_bindings[namespace].clone(),
                                            Shadowable::Never),
                                false));
            }
            Some(_) | None => {
                // Continue.
            }
        }

        // Next, check the module's imports if necessary.

        // If this is a search of all imports, we should be done with glob
        // resolution at this point.
        if name_search_type == PathSearch {
            assert_eq!(module_.glob_count.get(), 0);
        }

        // Check the list of resolved imports.
        match module_.import_resolutions.borrow().get(&name) {
            Some(import_resolution) if allow_private_imports ||
                                       import_resolution[namespace].is_public => {

                if import_resolution[namespace].is_public &&
                   import_resolution.outstanding_references != 0 {
                    debug!("(resolving name in module) import unresolved; bailing out");
                    return Indeterminate;
                }
                match import_resolution[namespace].target.clone() {
                    None => {
                        debug!("(resolving name in module) name found, but not in namespace {:?}",
                               namespace);
                    }
                    Some(target) => {
                        debug!("(resolving name in module) resolved to import");
                        // track used imports and extern crates as well
                        let id = import_resolution[namespace].id;
                        self.used_imports.insert((id, namespace));
                        self.record_import_use(id, name);
                        if let Some(DefId{krate: kid, ..}) = target.target_module.def_id() {
                            self.used_crates.insert(kid);
                        }
                        return Success((target, true));
                    }
                }
            }
            Some(..) | None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            // FIXME (21114): In principle unclear `child` *has* to be lifted.
            let child = module_.external_module_children.borrow().get(&name).cloned();
            if let Some(module) = child {
                let name_binding = NameBinding::create_from_module(module);
                return Success((Target::new(module_, name_binding, Shadowable::Never),
                                false));
            }
        }

        // We're out of luck.
        debug!("(resolving name in module) failed to resolve `{}`", name);
        return Failed(None);
    }

    fn report_unresolved_imports(&mut self, module_: Rc<Module>) {
        let index = module_.resolved_import_count.get();
        let imports = module_.imports.borrow();
        let import_count = imports.len();
        if index != import_count {
            resolve_error(self,
                          (*imports)[index].span,
                          ResolutionError::UnresolvedImport(None));
        }

        // Descend into children and anonymous children.
        build_reduced_graph::populate_module_if_necessary(self, &module_);

        for (_, child_node) in module_.children.borrow().iter() {
            match child_node.type_ns.module() {
                None => {
                    // Continue.
                }
                Some(child_module) => {
                    self.report_unresolved_imports(child_module);
                }
            }
        }

        for (_, module_) in module_.anonymous_children.borrow().iter() {
            self.report_unresolved_imports(module_.clone());
        }
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

    fn with_scope<F>(&mut self, name: Option<Name>, f: F)
        where F: FnOnce(&mut Resolver)
    {
        let orig_module = self.current_module.clone();

        // Move down in the graph.
        match name {
            None => {
                // Nothing to do.
            }
            Some(name) => {
                build_reduced_graph::populate_module_if_necessary(self, &orig_module);

                match orig_module.children.borrow().get(&name) {
                    None => {
                        debug!("!!! (with scope) didn't find `{}` in `{}`",
                               name,
                               module_to_string(&*orig_module));
                    }
                    Some(name_bindings) => {
                        match name_bindings.type_ns.module() {
                            None => {
                                debug!("!!! (with scope) didn't find module for `{}` in `{}`",
                                       name,
                                       module_to_string(&*orig_module));
                            }
                            Some(module_) => {
                                self.current_module = module_;
                            }
                        }
                    }
                }
            }
        }

        f(self);

        self.current_module = orig_module;
    }

    /// Searches the current set of local scopes for labels.
    /// Stops after meeting a closure.
    fn search_label(&self, name: Name) -> Option<DefLike> {
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

    fn resolve_crate(&mut self, krate: &hir::Crate) {
        debug!("(resolving crate) starting");

        intravisit::walk_crate(self, krate);
    }

    fn check_if_primitive_type_name(&self, name: Name, span: Span) {
        if let Some(_) = self.primitive_type_table.primitive_types.get(&name) {
            span_err!(self.session,
                      span,
                      E0317,
                      "user-defined types or type parameters cannot shadow the primitive types");
        }
    }

    fn resolve_item(&mut self, item: &Item) {
        let name = item.name;

        debug!("(resolving item) resolving {}", name);

        match item.node {
            ItemEnum(_, ref generics) |
            ItemTy(_, ref generics) |
            ItemStruct(_, ref generics) => {
                self.check_if_primitive_type_name(name, item.span);

                self.with_type_parameter_rib(HasTypeParameters(generics, TypeSpace, ItemRibKind),
                                             |this| intravisit::walk_item(this, item));
            }
            ItemFn(_, _, _, _, ref generics, _) => {
                self.with_type_parameter_rib(HasTypeParameters(generics, FnSpace, ItemRibKind),
                                             |this| intravisit::walk_item(this, item));
            }

            ItemDefaultImpl(_, ref trait_ref) => {
                self.with_optional_trait_ref(Some(trait_ref), |_, _| {});
            }
            ItemImpl(_, _, ref generics, ref opt_trait_ref, ref self_type, ref impl_items) => {
                self.resolve_implementation(generics,
                                            opt_trait_ref,
                                            &**self_type,
                                            item.id,
                                            impl_items);
            }

            ItemTrait(_, ref generics, ref bounds, ref trait_items) => {
                self.check_if_primitive_type_name(name, item.span);

                // Create a new rib for the trait-wide type parameters.
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               TypeSpace,
                                                               ItemRibKind),
                                             |this| {
                    let local_def_id = this.ast_map.local_def_id(item.id);
                    this.with_self_rib(DefSelfTy(Some(local_def_id), None), |this| {
                        this.visit_generics(generics);
                        walk_list!(this, visit_ty_param_bound, bounds);

                        for trait_item in trait_items {
                            match trait_item.node {
                                hir::ConstTraitItem(_, ref default) => {
                                    // Only impose the restrictions of
                                    // ConstRibKind if there's an actual constant
                                    // expression in a provided default.
                                    if default.is_some() {
                                        this.with_constant_rib(|this| {
                                            intravisit::walk_trait_item(this, trait_item)
                                        });
                                    } else {
                                        intravisit::walk_trait_item(this, trait_item)
                                    }
                                }
                                hir::MethodTraitItem(ref sig, _) => {
                                    let type_parameters =
                                        HasTypeParameters(&sig.generics,
                                                          FnSpace,
                                                          MethodRibKind);
                                    this.with_type_parameter_rib(type_parameters, |this| {
                                        intravisit::walk_trait_item(this, trait_item)
                                    });
                                }
                                hir::TypeTraitItem(..) => {
                                    this.check_if_primitive_type_name(trait_item.name,
                                                                      trait_item.span);
                                    this.with_type_parameter_rib(NoTypeParameters, |this| {
                                        intravisit::walk_trait_item(this, trait_item)
                                    });
                                }
                            };
                        }
                    });
                });
            }

            ItemMod(_) | ItemForeignMod(_) => {
                self.with_scope(Some(name), |this| {
                    intravisit::walk_item(this, item);
                });
            }

            ItemConst(..) | ItemStatic(..) => {
                self.with_constant_rib(|this| {
                    intravisit::walk_item(this, item);
                });
            }

            ItemUse(ref view_path) => {
                // check for imports shadowing primitive types
                let check_rename = |this: &Self, id, name| {
                    match this.def_map.borrow().get(&id).map(|d| d.full_def()) {
                        Some(DefTy(..)) | Some(DefStruct(..)) | Some(DefTrait(..)) | None => {
                            this.check_if_primitive_type_name(name, item.span);
                        }
                        _ => {}
                    }
                };

                match view_path.node {
                    hir::ViewPathSimple(name, _) => {
                        check_rename(self, item.id, name);
                    }
                    hir::ViewPathList(ref prefix, ref items) => {
                        for item in items {
                            if let Some(name) = item.node.rename() {
                                check_rename(self, item.node.id(), name);
                            }
                        }

                        // Resolve prefix of an import with empty braces (issue #28388)
                        if items.is_empty() && !prefix.segments.is_empty() {
                            match self.resolve_crate_relative_path(prefix.span,
                                                                   &prefix.segments,
                                                                   TypeNS) {
                                Some((def, lp)) =>
                                    self.record_def(item.id, PathResolution::new(def, lp, 0)),
                                None => {
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

            ItemExternCrate(_) => {
                // do nothing, these are just around to be encoded
            }
        }
    }

    fn with_type_parameter_rib<F>(&mut self, type_parameters: TypeParameters, f: F)
        where F: FnOnce(&mut Resolver)
    {
        match type_parameters {
            HasTypeParameters(generics, space, rib_kind) => {
                let mut function_type_rib = Rib::new(rib_kind);
                let mut seen_bindings = HashSet::new();
                for (index, type_parameter) in generics.ty_params.iter().enumerate() {
                    let name = type_parameter.name;
                    debug!("with_type_parameter_rib: {}", type_parameter.id);

                    if seen_bindings.contains(&name) {
                        resolve_error(self,
                                      type_parameter.span,
                                      ResolutionError::NameAlreadyUsedInTypeParameterList(name));
                    }
                    seen_bindings.insert(name);

                    // plain insert (no renaming)
                    function_type_rib.bindings
                                     .insert(name,
                                             DlDef(DefTyParam(space,
                                                              index as u32,
                                                              self.ast_map
                                                                  .local_def_id(type_parameter.id),
                                                              name)));
                }
                self.type_ribs.push(function_type_rib);
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f(self);

        match type_parameters {
            HasTypeParameters(..) => {
                if !self.resolved {
                    self.type_ribs.pop();
                }
            }
            NoTypeParameters => {}
        }
    }

    fn with_label_rib<F>(&mut self, f: F)
        where F: FnOnce(&mut Resolver)
    {
        self.label_ribs.push(Rib::new(NormalRibKind));
        f(self);
        if !self.resolved {
            self.label_ribs.pop();
        }
    }

    fn with_constant_rib<F>(&mut self, f: F)
        where F: FnOnce(&mut Resolver)
    {
        self.value_ribs.push(Rib::new(ConstantItemRibKind));
        self.type_ribs.push(Rib::new(ConstantItemRibKind));
        f(self);
        if !self.resolved {
            self.type_ribs.pop();
            self.value_ribs.pop();
        }
    }

    fn resolve_function(&mut self, rib_kind: RibKind, declaration: &FnDecl, block: &Block) {
        // Create a value rib for the function.
        self.value_ribs.push(Rib::new(rib_kind));

        // Create a label rib for the function.
        self.label_ribs.push(Rib::new(rib_kind));

        // Add each argument to the rib.
        let mut bindings_list = HashMap::new();
        for argument in &declaration.inputs {
            self.resolve_pattern(&*argument.pat, ArgumentIrrefutableMode, &mut bindings_list);

            self.visit_ty(&*argument.ty);

            debug!("(resolving function) recorded argument");
        }
        intravisit::walk_fn_ret_ty(self, &declaration.output);

        // Resolve the function body.
        self.visit_block(block);

        debug!("(resolving function) leaving function");

        if !self.resolved {
            self.label_ribs.pop();
            self.value_ribs.pop();
        }
    }

    fn resolve_trait_reference(&mut self,
                               id: NodeId,
                               trait_path: &Path,
                               path_depth: usize)
                               -> Result<PathResolution, ()> {
        if let Some(path_res) = self.resolve_path(id, trait_path, path_depth, TypeNS, true) {
            if let DefTrait(_) = path_res.base_def {
                debug!("(resolving trait) found trait def: {:?}", path_res);
                Ok(path_res)
            } else {
                let mut err =
                    resolve_struct_error(self,
                                  trait_path.span,
                                  ResolutionError::IsNotATrait(&*path_names_to_string(trait_path,
                                                                                      path_depth)));

                // If it's a typedef, give a note
                if let DefTy(..) = path_res.base_def {
                    err.span_note(trait_path.span,
                                  "`type` aliases cannot be used for traits");
                }
                err.emit();
                Err(())
            }
        } else {
            resolve_error(self,
                          trait_path.span,
                          ResolutionError::UndeclaredTraitName(&*path_names_to_string(trait_path,
                                                                                      path_depth)));
            Err(())
        }
    }

    fn resolve_generics(&mut self, generics: &Generics) {
        for type_parameter in generics.ty_params.iter() {
            self.check_if_primitive_type_name(type_parameter.name, type_parameter.span);
        }
        for predicate in &generics.where_clause.predicates {
            match predicate {
                &hir::WherePredicate::BoundPredicate(_) |
                &hir::WherePredicate::RegionPredicate(_) => {}
                &hir::WherePredicate::EqPredicate(ref eq_pred) => {
                    let path_res = self.resolve_path(eq_pred.id, &eq_pred.path, 0, TypeNS, true);
                    if let Some(PathResolution { base_def: DefTyParam(..), .. }) = path_res {
                        self.record_def(eq_pred.id, path_res.unwrap());
                    } else {
                        resolve_error(self,
                                      eq_pred.span,
                                      ResolutionError::UndeclaredAssociatedType);
                        self.record_def(eq_pred.id, err_path_resolution());
                    }
                }
            }
        }
        intravisit::walk_generics(self, generics);
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
            intravisit::walk_trait_ref(self, trait_ref);
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
        let name = special_names::type_self;
        self_type_rib.bindings.insert(name, DlDef(self_def));
        self.type_ribs.push(self_type_rib);
        f(self);
        if !self.resolved {
            self.type_ribs.pop();
        }
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

                this.with_self_rib(DefSelfTy(trait_id, Some((item_id, self_type.id))), |this| {
                    this.with_current_self_type(self_type, |this| {
                        for impl_item in impl_items {
                            match impl_item.node {
                                hir::ImplItemKind::Const(..) => {
                                    // If this is a trait impl, ensure the const
                                    // exists in trait
                                    this.check_trait_item(impl_item.name,
                                                          impl_item.span,
                                        |n, s| ResolutionError::ConstNotMemberOfTrait(n, s));
                                    this.with_constant_rib(|this| {
                                        intravisit::walk_impl_item(this, impl_item);
                                    });
                                }
                                hir::ImplItemKind::Method(ref sig, _) => {
                                    // If this is a trait impl, ensure the method
                                    // exists in trait
                                    this.check_trait_item(impl_item.name,
                                                          impl_item.span,
                                        |n, s| ResolutionError::MethodNotMemberOfTrait(n, s));

                                    // We also need a new scope for the method-
                                    // specific type parameters.
                                    let type_parameters =
                                        HasTypeParameters(&sig.generics,
                                                          FnSpace,
                                                          MethodRibKind);
                                    this.with_type_parameter_rib(type_parameters, |this| {
                                        intravisit::walk_impl_item(this, impl_item);
                                    });
                                }
                                hir::ImplItemKind::Type(ref ty) => {
                                    // If this is a trait impl, ensure the type
                                    // exists in trait
                                    this.check_trait_item(impl_item.name,
                                                          impl_item.span,
                                        |n, s| ResolutionError::TypeNotMemberOfTrait(n, s));

                                    this.visit_ty(ty);
                                }
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
                resolve_error(self, span, err(name, &*path_str));
            }
        }
    }

    fn resolve_local(&mut self, local: &Local) {
        // Resolve the type.
        walk_list!(self, visit_ty, &local.ty);

        // Resolve the initializer.
        walk_list!(self, visit_expr, &local.init);

        // Resolve the pattern.
        self.resolve_pattern(&*local.pat, LocalIrrefutableMode, &mut HashMap::new());
    }

    // build a map from pattern identifiers to binding-info's.
    // this is done hygienically. This could arise for a macro
    // that expands into an or-pattern where one 'x' was from the
    // user and one 'x' came from the macro.
    fn binding_mode_map(&mut self, pat: &Pat) -> BindingMap {
        let mut result = HashMap::new();
        pat_bindings(&self.def_map, pat, |binding_mode, _id, sp, path1| {
            let name = path1.node;
            result.insert(name,
                          BindingInfo {
                              span: sp,
                              binding_mode: binding_mode,
                          });
        });
        return result;
    }

    // check that all of the arms in an or-pattern have exactly the
    // same set of bindings, with the same binding modes for each.
    fn check_consistent_bindings(&mut self, arm: &Arm) {
        if arm.pats.is_empty() {
            return;
        }
        let map_0 = self.binding_mode_map(&*arm.pats[0]);
        for (i, p) in arm.pats.iter().enumerate() {
            let map_i = self.binding_mode_map(&**p);

            for (&key, &binding_0) in &map_0 {
                match map_i.get(&key) {
                    None => {
                        resolve_error(self,
                                      p.span,
                                      ResolutionError::VariableNotBoundInPattern(key, i + 1));
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
                                  ResolutionError::VariableNotBoundInParentPattern(key, i + 1));
                }
            }
        }
    }

    fn resolve_arm(&mut self, arm: &Arm) {
        self.value_ribs.push(Rib::new(NormalRibKind));

        let mut bindings_list = HashMap::new();
        for pattern in &arm.pats {
            self.resolve_pattern(&**pattern, RefutableMode, &mut bindings_list);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        walk_list!(self, visit_expr, &arm.guard);
        self.visit_expr(&*arm.body);

        if !self.resolved {
            self.value_ribs.pop();
        }
    }

    fn resolve_block(&mut self, block: &Block) {
        debug!("(resolving block) entering block");
        self.value_ribs.push(Rib::new(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module.clone();
        match orig_module.anonymous_children.borrow().get(&block.id) {
            None => {
                // Nothing to do.
            }
            Some(anonymous_module) => {
                debug!("(resolving block) found anonymous module, moving down");
                self.current_module = anonymous_module.clone();
            }
        }

        // Check for imports appearing after non-item statements.
        let mut found_non_item = false;
        for statement in &block.stmts {
            if let hir::StmtDecl(ref declaration, _) = statement.node {
                if let hir::DeclItem(i) = declaration.node {
                    let i = self.ast_map.expect_item(i.id);
                    match i.node {
                        ItemExternCrate(_) | ItemUse(_) if found_non_item => {
                            span_err!(self.session,
                                      i.span,
                                      E0154,
                                      "imports are not allowed after non-item statements");
                        }
                        _ => {}
                    }
                } else {
                    found_non_item = true
                }
            } else {
                found_non_item = true;
            }
        }

        // Descend into the block.
        intravisit::walk_block(self, block);

        // Move back up.
        if !self.resolved {
            self.current_module = orig_module;
            self.value_ribs.pop();
        }
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(&mut self, ty: &Ty) {
        match ty.node {
            TyPath(ref maybe_qself, ref path) => {
                let resolution = match self.resolve_possibly_assoc_item(ty.id,
                                                                        maybe_qself.as_ref(),
                                                                        path,
                                                                        TypeNS,
                                                                        true) {
                    // `<T>::a::b::c` is resolved by typeck alone.
                    TypecheckRequired => {
                        // Resolve embedded types.
                        intravisit::walk_ty(self, ty);
                        return;
                    }
                    ResolveAttempt(resolution) => resolution,
                };

                // This is a path in the type namespace. Walk through scopes
                // looking for it.
                match resolution {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving type) writing resolution for `{}` (id {}) = {:?}",
                               path_names_to_string(path, 0),
                               ty.id,
                               def);
                        self.record_def(ty.id, def);
                    }
                    None => {
                        self.record_def(ty.id, err_path_resolution());

                        // Keep reporting some errors even if they're ignored above.
                        self.resolve_path(ty.id, path, 0, TypeNS, true);

                        let kind = if maybe_qself.is_some() {
                            "associated type"
                        } else {
                            "type name"
                        };

                        let self_type_name = special_idents::type_self.name;
                        let is_invalid_self_type_name = path.segments.len() > 0 &&
                                                        maybe_qself.is_none() &&
                                                        path.segments[0].identifier.name ==
                                                        self_type_name;
                        if is_invalid_self_type_name {
                            resolve_error(self,
                                          ty.span,
                                          ResolutionError::SelfUsedOutsideImplOrTrait);
                        } else {
                            resolve_error(self,
                                          ty.span,
                                          ResolutionError::UseOfUndeclared(
                                                                    kind,
                                                                    &*path_names_to_string(path,
                                                                                           0))
                                         );
                        }
                    }
                }
            }
            _ => {}
        }
        // Resolve embedded types.
        intravisit::walk_ty(self, ty);
    }

    fn resolve_pattern(&mut self,
                       pattern: &Pat,
                       mode: PatternBindingMode,
                       // Maps idents to the node ID for the (outermost)
                       // pattern that binds them
                       bindings_list: &mut HashMap<Name, NodeId>) {
        let pat_id = pattern.id;
        walk_pat(pattern, |pattern| {
            match pattern.node {
                PatIdent(binding_mode, ref path1, ref at_rhs) => {
                    // The meaning of PatIdent with no type parameters
                    // depends on whether an enum variant or unit-like struct
                    // with that name is in scope. The probing lookup has to
                    // be careful not to emit spurious errors. Only matching
                    // patterns (match) can match nullary variants or
                    // unit-like structs. For binding patterns (let
                    // and the LHS of @-patterns), matching such a value is
                    // simply disallowed (since it's rarely what you want).
                    let const_ok = mode == RefutableMode && at_rhs.is_none();

                    let ident = path1.node;
                    let renamed = ident.name;

                    match self.resolve_bare_identifier_pattern(ident.unhygienic_name,
                                                               pattern.span) {
                        FoundStructOrEnumVariant(def, lp) if const_ok => {
                            debug!("(resolving pattern) resolving `{}` to struct or enum variant",
                                   renamed);

                            self.enforce_default_binding_mode(pattern,
                                                              binding_mode,
                                                              "an enum variant");
                            self.record_def(pattern.id,
                                            PathResolution {
                                                base_def: def,
                                                last_private: lp,
                                                depth: 0,
                                            });
                        }
                        FoundStructOrEnumVariant(..) => {
                            resolve_error(
                                self,
                                pattern.span,
                                ResolutionError::DeclarationShadowsEnumVariantOrUnitLikeStruct(
                                    renamed)
                            );
                            self.record_def(pattern.id, err_path_resolution());
                        }
                        FoundConst(def, lp, _) if const_ok => {
                            debug!("(resolving pattern) resolving `{}` to constant", renamed);

                            self.enforce_default_binding_mode(pattern, binding_mode, "a constant");
                            self.record_def(pattern.id,
                                            PathResolution {
                                                base_def: def,
                                                last_private: lp,
                                                depth: 0,
                                            });
                        }
                        FoundConst(def, _, name) => {
                            resolve_error(
                                self,
                                pattern.span,
                                ResolutionError::OnlyIrrefutablePatternsAllowedHere(def.def_id(),
                                                                                    name)
                            );
                            self.record_def(pattern.id, err_path_resolution());
                        }
                        BareIdentifierPatternUnresolved => {
                            debug!("(resolving pattern) binding `{}`", renamed);

                            let def_id = self.ast_map.local_def_id(pattern.id);
                            let def = DefLocal(def_id, pattern.id);

                            // Record the definition so that later passes
                            // will be able to distinguish variants from
                            // locals in patterns.

                            self.record_def(pattern.id,
                                            PathResolution {
                                                base_def: def,
                                                last_private: LastMod(AllPublic),
                                                depth: 0,
                                            });

                            // Add the binding to the local ribs, if it
                            // doesn't already exist in the bindings list. (We
                            // must not add it if it's in the bindings list
                            // because that breaks the assumptions later
                            // passes make about or-patterns.)
                            if !bindings_list.contains_key(&renamed) {
                                let this = &mut *self;
                                let last_rib = this.value_ribs.last_mut().unwrap();
                                last_rib.bindings.insert(renamed, DlDef(def));
                                bindings_list.insert(renamed, pat_id);
                            } else if mode == ArgumentIrrefutableMode &&
                               bindings_list.contains_key(&renamed) {
                                // Forbid duplicate bindings in the same
                                // parameter list.
                                resolve_error(
                                    self,
                                    pattern.span,
                                    ResolutionError::IdentifierBoundMoreThanOnceInParameterList(
                                        &ident.name.as_str())
                                );
                            } else if bindings_list.get(&renamed) == Some(&pat_id) {
                                // Then this is a duplicate variable in the
                                // same disjunction, which is an error.
                                resolve_error(
                                    self,
                                    pattern.span,
                                    ResolutionError::IdentifierBoundMoreThanOnceInSamePattern(
                                        &ident.name.as_str())
                                );
                            }
                            // Else, not bound in the same pattern: do
                            // nothing.
                        }
                    }
                }

                PatEnum(ref path, _) => {
                    // This must be an enum variant, struct or const.
                    let resolution = match self.resolve_possibly_assoc_item(pat_id,
                                                                            None,
                                                                            path,
                                                                            ValueNS,
                                                                            false) {
                        // The below shouldn't happen because all
                        // qualified paths should be in PatQPath.
                        TypecheckRequired =>
                            self.session.span_bug(path.span,
                                                  "resolve_possibly_assoc_item claimed
                                     \
                                                   that a path in PatEnum requires typecheck
                                     \
                                                   to resolve, but qualified paths should be
                                     \
                                                   PatQPath"),
                        ResolveAttempt(resolution) => resolution,
                    };
                    if let Some(path_res) = resolution {
                        match path_res.base_def {
                            DefVariant(..) | DefStruct(..) | DefConst(..) => {
                                self.record_def(pattern.id, path_res);
                            }
                            DefStatic(..) => {
                                resolve_error(&self,
                                              path.span,
                                              ResolutionError::StaticVariableReference);
                                self.record_def(pattern.id, err_path_resolution());
                            }
                            _ => {
                                // If anything ends up here entirely resolved,
                                // it's an error. If anything ends up here
                                // partially resolved, that's OK, because it may
                                // be a `T::CONST` that typeck will resolve.
                                if path_res.depth == 0 {
                                    resolve_error(
                                        self,
                                        path.span,
                                        ResolutionError::NotAnEnumVariantStructOrConst(
                                            &path.segments
                                                 .last()
                                                 .unwrap()
                                                 .identifier
                                                 .name
                                                 .as_str())
                                    );
                                    self.record_def(pattern.id, err_path_resolution());
                                } else {
                                    let const_name = path.segments
                                                         .last()
                                                         .unwrap()
                                                         .identifier
                                                         .name;
                                    let traits = self.get_traits_containing_item(const_name);
                                    self.trait_map.insert(pattern.id, traits);
                                    self.record_def(pattern.id, path_res);
                                }
                            }
                        }
                    } else {
                        resolve_error(
                            self,
                            path.span,
                            ResolutionError::UnresolvedEnumVariantStructOrConst(
                                &path.segments.last().unwrap().identifier.name.as_str())
                        );
                        self.record_def(pattern.id, err_path_resolution());
                    }
                    intravisit::walk_path(self, path);
                }

                PatQPath(ref qself, ref path) => {
                    // Associated constants only.
                    let resolution = match self.resolve_possibly_assoc_item(pat_id,
                                                                            Some(qself),
                                                                            path,
                                                                            ValueNS,
                                                                            false) {
                        TypecheckRequired => {
                            // All `<T>::CONST` should end up here, and will
                            // require use of the trait map to resolve
                            // during typechecking.
                            let const_name = path.segments
                                                 .last()
                                                 .unwrap()
                                                 .identifier
                                                 .name;
                            let traits = self.get_traits_containing_item(const_name);
                            self.trait_map.insert(pattern.id, traits);
                            intravisit::walk_pat(self, pattern);
                            return true;
                        }
                        ResolveAttempt(resolution) => resolution,
                    };
                    if let Some(path_res) = resolution {
                        match path_res.base_def {
                            // All `<T as Trait>::CONST` should end up here, and
                            // have the trait already selected.
                            DefAssociatedConst(..) => {
                                self.record_def(pattern.id, path_res);
                            }
                            _ => {
                                resolve_error(
                                    self,
                                    path.span,
                                    ResolutionError::NotAnAssociatedConst(
                                        &path.segments.last().unwrap().identifier.name.as_str()
                                    )
                                );
                                self.record_def(pattern.id, err_path_resolution());
                            }
                        }
                    } else {
                        resolve_error(self,
                                      path.span,
                                      ResolutionError::UnresolvedAssociatedConst(&path.segments
                                                                                      .last()
                                                                                      .unwrap()
                                                                                      .identifier
                                                                                      .name
                                                                                      .as_str()));
                        self.record_def(pattern.id, err_path_resolution());
                    }
                    intravisit::walk_pat(self, pattern);
                }

                PatStruct(ref path, _, _) => {
                    match self.resolve_path(pat_id, path, 0, TypeNS, false) {
                        Some(definition) => {
                            self.record_def(pattern.id, definition);
                        }
                        result => {
                            debug!("(resolving pattern) didn't find struct def: {:?}", result);
                            resolve_error(
                                self,
                                path.span,
                                ResolutionError::DoesNotNameAStruct(
                                    &*path_names_to_string(path, 0))
                            );
                            self.record_def(pattern.id, err_path_resolution());
                        }
                    }
                    intravisit::walk_path(self, path);
                }

                PatLit(_) | PatRange(..) => {
                    intravisit::walk_pat(self, pattern);
                }

                _ => {
                    // Nothing to do.
                }
            }
            true
        });
    }

    fn resolve_bare_identifier_pattern(&mut self,
                                       name: Name,
                                       span: Span)
                                       -> BareIdentifierPatternResolution {
        let module = self.current_module.clone();
        match self.resolve_item_in_lexical_scope(module, name, ValueNS, true) {
            Success((target, _)) => {
                debug!("(resolve bare identifier pattern) succeeded in finding {} at {:?}",
                       name,
                       target.binding.borrow());
                match target.binding.def() {
                    None => {
                        panic!("resolved name in the value namespace to a set of name bindings \
                                with no def?!");
                    }
                    // For the two success cases, this lookup can be
                    // considered as not having a private component because
                    // the lookup happened only within the current module.
                    Some(def @ DefVariant(..)) | Some(def @ DefStruct(..)) => {
                        return FoundStructOrEnumVariant(def, LastMod(AllPublic));
                    }
                    Some(def @ DefConst(..)) | Some(def @ DefAssociatedConst(..)) => {
                        return FoundConst(def, LastMod(AllPublic), name);
                    }
                    Some(DefStatic(..)) => {
                        resolve_error(self, span, ResolutionError::StaticVariableReference);
                        return BareIdentifierPatternUnresolved;
                    }
                    _ => return BareIdentifierPatternUnresolved
                }
            }

            Indeterminate => {
                panic!("unexpected indeterminate result");
            }
            Failed(err) => {
                match err {
                    Some((span, msg)) => {
                        resolve_error(self, span, ResolutionError::FailedToResolve(&*msg));
                    }
                    None => (),
                }

                debug!("(resolve bare identifier pattern) failed to find {}", name);
                return BareIdentifierPatternUnresolved;
            }
        }
    }

    /// Handles paths that may refer to associated items
    fn resolve_possibly_assoc_item(&mut self,
                                   id: NodeId,
                                   maybe_qself: Option<&hir::QSelf>,
                                   path: &Path,
                                   namespace: Namespace,
                                   check_ribs: bool)
                                   -> AssocItemResolveResult {
        let max_assoc_types;

        match maybe_qself {
            Some(qself) => {
                if qself.position == 0 {
                    return TypecheckRequired;
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
            this.resolve_path(id, path, 0, namespace, check_ribs)
        });
        for depth in 1..max_assoc_types {
            if resolution.is_some() {
                break;
            }
            self.with_no_errors(|this| {
                resolution = this.resolve_path(id, path, depth, TypeNS, true);
            });
        }
        if let Some(DefMod(_)) = resolution.map(|r| r.base_def) {
            // A module is not a valid type or value.
            resolution = None;
        }
        ResolveAttempt(resolution)
    }

    /// If `check_ribs` is true, checks the local definitions first; i.e.
    /// doesn't skip straight to the containing module.
    /// Skips `path_depth` trailing segments, which is also reflected in the
    /// returned value. See `middle::def::PathResolution` for more info.
    pub fn resolve_path(&mut self,
                        id: NodeId,
                        path: &Path,
                        path_depth: usize,
                        namespace: Namespace,
                        check_ribs: bool)
                        -> Option<PathResolution> {
        let span = path.span;
        let segments = &path.segments[..path.segments.len() - path_depth];

        let mk_res = |(def, lp)| PathResolution::new(def, lp, path_depth);

        if path.global {
            let def = self.resolve_crate_relative_path(span, segments, namespace);
            return def.map(mk_res);
        }

        // Try to find a path to an item in a module.
        let last_ident = segments.last().unwrap().identifier;
        if segments.len() <= 1 {
            let unqualified_def = self.resolve_identifier(last_ident, namespace, check_ribs, true);
            return unqualified_def.and_then(|def| self.adjust_local_def(def, span))
                                  .map(|def| {
                                      PathResolution::new(def, LastMod(AllPublic), path_depth)
                                  });
        }

        let unqualified_def = self.resolve_identifier(last_ident, namespace, check_ribs, false);
        let def = self.resolve_module_relative_path(span, segments, namespace);
        match (def, unqualified_def) {
            (Some((ref d, _)), Some(ref ud)) if *d == ud.def => {
                self.session
                    .add_lint(lint::builtin::UNUSED_QUALIFICATIONS,
                              id,
                              span,
                              "unnecessary qualification".to_string());
            }
            _ => {}
        }

        def.map(mk_res)
    }

    // Resolve a single identifier
    fn resolve_identifier(&mut self,
                          identifier: hir::Ident,
                          namespace: Namespace,
                          check_ribs: bool,
                          record_used: bool)
                          -> Option<LocalDef> {
        // First, check to see whether the name is a primitive type.
        if namespace == TypeNS {
            if let Some(&prim_ty) = self.primitive_type_table
                                        .primitive_types
                                        .get(&identifier.unhygienic_name) {
                return Some(LocalDef::from_def(DefPrimTy(prim_ty)));
            }
        }

        if check_ribs {
            if let Some(def) = self.resolve_identifier_in_local_ribs(identifier, namespace) {
                return Some(def);
            }
        }

        let name = identifier.unhygienic_name;
        self.resolve_item_by_name_in_lexical_scope(name, namespace, record_used)
            .map(LocalDef::from_def)
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
            DefUpvar(..) => {
                self.session.span_bug(span, &format!("unexpected {:?} in bindings", def))
            }
            DefLocal(_, node_id) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind => {
                            // Nothing to do. Continue.
                        }
                        ClosureRibKind(function_id) => {
                            let prev_def = def;
                            let node_def_id = self.ast_map.local_def_id(node_id);

                            let seen = self.freevars_seen
                                           .entry(function_id)
                                           .or_insert_with(|| NodeMap());
                            if let Some(&index) = seen.get(&node_id) {
                                def = DefUpvar(node_def_id, node_id, index, function_id);
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

                            def = DefUpvar(node_def_id, node_id, depth, function_id);
                            seen.insert(node_id, depth);
                        }
                        ItemRibKind | MethodRibKind => {
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
            DefTyParam(..) | DefSelfTy(..) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind | MethodRibKind | ClosureRibKind(..) => {
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
                                    segments: &[hir::PathSegment],
                                    namespace: Namespace)
                                    -> Option<(Def, LastPrivate)> {
        let module_path = segments.split_last()
                                  .unwrap()
                                  .1
                                  .iter()
                                  .map(|ps| ps.identifier.name)
                                  .collect::<Vec<_>>();

        let containing_module;
        let last_private;
        let current_module = self.current_module.clone();
        match self.resolve_module_path(current_module,
                                       &module_path[..],
                                       UseLexicalScope,
                                       span,
                                       PathSearch) {
            Failed(err) => {
                let (span, msg) = match err {
                    Some((span, msg)) => (span, msg),
                    None => {
                        let msg = format!("Use of undeclared type or module `{}`",
                                          names_to_string(&module_path));
                        (span, msg)
                    }
                };

                resolve_error(self, span, ResolutionError::FailedToResolve(&*msg));
                return None;
            }
            Indeterminate => panic!("indeterminate unexpected"),
            Success((resulting_module, resulting_last_private)) => {
                containing_module = resulting_module;
                last_private = resulting_last_private;
            }
        }

        let name = segments.last().unwrap().identifier.name;
        let def = match self.resolve_name_in_module(containing_module.clone(),
                                                    name,
                                                    namespace,
                                                    NameSearchType::PathSearch,
                                                    false) {
            Success((Target { binding, .. }, _)) => {
                let (def, lp) = binding.def_and_lp();
                (def, last_private.or(lp))
            }
            _ => return None,
        };
        if let Some(DefId{krate: kid, ..}) = containing_module.def_id() {
            self.used_crates.insert(kid);
        }
        return Some(def);
    }

    /// Invariant: This must be called only during main resolution, not during
    /// import resolution.
    fn resolve_crate_relative_path(&mut self,
                                   span: Span,
                                   segments: &[hir::PathSegment],
                                   namespace: Namespace)
                                   -> Option<(Def, LastPrivate)> {
        let module_path = segments.split_last()
                                  .unwrap()
                                  .1
                                  .iter()
                                  .map(|ps| ps.identifier.name)
                                  .collect::<Vec<_>>();

        let root_module = self.graph_root.clone();

        let containing_module;
        let last_private;
        match self.resolve_module_path_from_root(root_module,
                                                 &module_path[..],
                                                 0,
                                                 span,
                                                 PathSearch,
                                                 LastMod(AllPublic)) {
            Failed(err) => {
                let (span, msg) = match err {
                    Some((span, msg)) => (span, msg),
                    None => {
                        let msg = format!("Use of undeclared module `::{}`",
                                          names_to_string(&module_path[..]));
                        (span, msg)
                    }
                };

                resolve_error(self, span, ResolutionError::FailedToResolve(&*msg));
                return None;
            }

            Indeterminate => {
                panic!("indeterminate unexpected");
            }

            Success((resulting_module, resulting_last_private)) => {
                containing_module = resulting_module;
                last_private = resulting_last_private;
            }
        }

        let name = segments.last().unwrap().identifier.name;
        match self.resolve_name_in_module(containing_module,
                                          name,
                                          namespace,
                                          NameSearchType::PathSearch,
                                          false) {
            Success((Target { binding, .. }, _)) => {
                let (def, lp) = binding.def_and_lp();
                Some((def, last_private.or(lp)))
            }
            _ => None,
        }
    }

    fn resolve_identifier_in_local_ribs(&mut self,
                                        ident: hir::Ident,
                                        namespace: Namespace)
                                        -> Option<LocalDef> {
        // Check the local set of ribs.
        let (name, ribs) = match namespace {
            ValueNS => (ident.name, &self.value_ribs),
            TypeNS => (ident.unhygienic_name, &self.type_ribs),
        };

        for (i, rib) in ribs.iter().enumerate().rev() {
            if let Some(def_like) = rib.bindings.get(&name).cloned() {
                match def_like {
                    DlDef(def) => {
                        debug!("(resolving path in local ribs) resolved `{}` to {:?} at {}",
                               name,
                               def,
                               i);
                        return Some(LocalDef {
                            ribs: Some((namespace, i)),
                            def: def,
                        });
                    }
                    def_like => {
                        debug!("(resolving path in local ribs) resolved `{}` to pseudo-def {:?}",
                               name,
                               def_like);
                        return None;
                    }
                }
            }
        }

        None
    }

    fn resolve_item_by_name_in_lexical_scope(&mut self,
                                             name: Name,
                                             namespace: Namespace,
                                             record_used: bool)
                                             -> Option<Def> {
        // Check the items.
        let module = self.current_module.clone();
        match self.resolve_item_in_lexical_scope(module, name, namespace, record_used) {
            Success((target, _)) => {
                match target.binding.def() {
                    None => {
                        // This can happen if we were looking for a type and
                        // found a module instead. Modules don't have defs.
                        debug!("(resolving item path by identifier in lexical scope) failed to \
                                resolve {} after success...",
                               name);
                        None
                    }
                    Some(def) => {
                        debug!("(resolving item path in lexical scope) resolved `{}` to item",
                               name);
                        // This lookup is "all public" because it only searched
                        // for one identifier in the current module (couldn't
                        // have passed through reexports or anything like that.
                        Some(def)
                    }
                }
            }
            Indeterminate => {
                panic!("unexpected indeterminate result");
            }
            Failed(err) => {
                debug!("(resolving item path by identifier in lexical scope) failed to resolve {}",
                       name);

                if let Some((span, msg)) = err {
                    resolve_error(self, span, ResolutionError::FailedToResolve(&*msg))
                }

                None
            }
        }
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
        fn extract_path_and_node_id(t: &Ty,
                                    allow: FallbackChecks)
                                    -> Option<(Path, NodeId, FallbackChecks)> {
            match t.node {
                TyPath(None, ref path) => Some((path.clone(), t.id, allow)),
                TyPtr(ref mut_ty) => extract_path_and_node_id(&*mut_ty.ty, OnlyTraitAndStatics),
                TyRptr(_, ref mut_ty) => extract_path_and_node_id(&*mut_ty.ty, allow),
                // This doesn't handle the remaining `Ty` variants as they are not
                // that commonly the self_type, it might be interesting to provide
                // support for those in future.
                _ => None,
            }
        }

        fn get_module(this: &mut Resolver,
                      span: Span,
                      name_path: &[ast::Name])
                      -> Option<Rc<Module>> {
            let root = this.current_module.clone();
            let last_name = name_path.last().unwrap();

            if name_path.len() == 1 {
                match this.primitive_type_table.primitive_types.get(last_name) {
                    Some(_) => None,
                    None => {
                        match this.current_module.children.borrow().get(last_name) {
                            Some(child) => child.type_ns.module(),
                            None => None,
                        }
                    }
                }
            } else {
                match this.resolve_module_path(root,
                                               &name_path[..],
                                               UseLexicalScope,
                                               span,
                                               PathSearch) {
                    Success((module, _)) => Some(module),
                    _ => None,
                }
            }
        }

        fn is_static_method(this: &Resolver, did: DefId) -> bool {
            if let Some(node_id) = this.ast_map.as_local_node_id(did) {
                let sig = match this.ast_map.get(node_id) {
                    hir_map::NodeTraitItem(trait_item) => match trait_item.node {
                        hir::MethodTraitItem(ref sig, _) => sig,
                        _ => return false,
                    },
                    hir_map::NodeImplItem(impl_item) => match impl_item.node {
                        hir::ImplItemKind::Method(ref sig, _) => sig,
                        _ => return false,
                    },
                    _ => return false,
                };
                sig.explicit_self.node == hir::SelfStatic
            } else {
                this.session.cstore.is_static_method(did)
            }
        }

        let (path, node_id, allowed) = match self.current_self_type {
            Some(ref ty) => match extract_path_and_node_id(ty, Everything) {
                Some(x) => x,
                None => return NoSuggestion,
            },
            None => return NoSuggestion,
        };

        if allowed == Everything {
            // Look for a field with the same name in the current self_type.
            match self.def_map.borrow().get(&node_id).map(|d| d.full_def()) {
                Some(DefTy(did, _)) |
                Some(DefStruct(did)) |
                Some(DefVariant(_, did, _)) => match self.structs.get(&did) {
                    None => {}
                    Some(fields) => {
                        if fields.iter().any(|&field_name| name == field_name) {
                            return Field;
                        }
                    }
                },
                _ => {} // Self type didn't resolve properly
            }
        }

        let name_path = path.segments.iter().map(|seg| seg.identifier.name).collect::<Vec<_>>();

        // Look for a method in the current self type's impl module.
        if let Some(module) = get_module(self, path.span, &name_path) {
            if let Some(binding) = module.children.borrow().get(&name) {
                if let Some(DefMethod(did)) = binding.value_ns.def() {
                    if is_static_method(self, did) {
                        return StaticMethod(path_names_to_string(&path, 0));
                    }
                    if self.current_trait_ref.is_some() {
                        return TraitItem;
                    } else if allowed == Everything {
                        return Method;
                    }
                }
            }
        }

        // Look for a method in the current trait.
        if let Some((trait_did, ref trait_ref)) = self.current_trait_ref {
            if let Some(&did) = self.trait_item_map.get(&(name, trait_did)) {
                if is_static_method(self, did) {
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
            if name != &*found {
                return SuggestionType::Function(found);
            }
        } SuggestionType::NotFound
    }

    fn resolve_expr(&mut self, expr: &Expr) {
        // First, record candidate traits for this expression if it could
        // result in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.node {
            ExprPath(ref maybe_qself, ref path) => {
                let resolution = match self.resolve_possibly_assoc_item(expr.id,
                                                                        maybe_qself.as_ref(),
                                                                        path,
                                                                        ValueNS,
                                                                        true) {
                    // `<T>::a::b::c` is resolved by typeck alone.
                    TypecheckRequired => {
                        let method_name = path.segments.last().unwrap().identifier.name;
                        let traits = self.get_traits_containing_item(method_name);
                        self.trait_map.insert(expr.id, traits);
                        intravisit::walk_expr(self, expr);
                        return;
                    }
                    ResolveAttempt(resolution) => resolution,
                };

                // This is a local path in the value namespace. Walk through
                // scopes looking for it.
                if let Some(path_res) = resolution {
                    // Check if struct variant
                    if let DefVariant(_, _, true) = path_res.base_def {
                        let path_name = path_names_to_string(path, 0);

                        let mut err = resolve_struct_error(self,
                                        expr.span,
                                        ResolutionError::StructVariantUsedAsFunction(&*path_name));

                        let msg = format!("did you mean to write: `{} {{ /* fields */ }}`?",
                                          path_name);
                        if self.emit_errors {
                            err.fileline_help(expr.span, &msg);
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
                        this.resolve_path(expr.id, path, 0, TypeNS, false)
                    });

                    self.record_def(expr.id, err_path_resolution());
                    match type_res.map(|r| r.base_def) {
                        Some(DefTy(struct_id, _)) if self.structs.contains_key(&struct_id) => {
                            let mut err = resolve_struct_error(self,
                                expr.span,
                                ResolutionError::StructVariantUsedAsFunction(&*path_name));

                            let msg = format!("did you mean to write: `{} {{ /* fields */ }}`?",
                                              path_name);
                            if self.emit_errors {
                                err.fileline_help(expr.span, &msg);
                            } else {
                                err.span_help(expr.span, &msg);
                            }
                            err.emit();
                        }
                        _ => {
                            // Keep reporting some errors even if they're ignored above.
                            self.resolve_path(expr.id, path, 0, ValueNS, true);

                            let mut method_scope = false;
                            self.value_ribs.iter().rev().all(|rib| {
                                method_scope = match rib.kind {
                                    MethodRibKind => true,
                                    ItemRibKind | ConstantItemRibKind => false,
                                    _ => return true, // Keep advancing
                                };
                                false // Stop advancing
                            });

                            if method_scope && special_names::self_.as_str() == &path_name[..] {
                                resolve_error(self,
                                              expr.span,
                                              ResolutionError::SelfNotAvailableInStaticMethod);
                            } else {
                                let last_name = path.segments.last().unwrap().identifier.name;
                                let mut msg = match self.find_fallback_in_self_type(last_name) {
                                    NoSuggestion => {
                                        // limit search to 5 to reduce the number
                                        // of stupid suggestions
                                        match self.find_best_match(&path_name) {
                                            SuggestionType::Macro(s) => {
                                                format!("the macro `{}`", s)
                                            }
                                            SuggestionType::Function(s) => format!("`{}`", s),
                                            SuggestionType::NotFound => "".to_string(),
                                        }
                                    }
                                    Field => format!("`self.{}`", path_name),
                                    Method |
                                    TraitItem => format!("to call `self.{}`", path_name),
                                    TraitMethod(path_str) |
                                    StaticMethod(path_str) =>
                                        format!("to call `{}::{}`", path_str, path_name),
                                };

                                let mut context =  UnresolvedNameContext::Other;
                                if !msg.is_empty() {
                                    msg = format!(". Did you mean {}?", msg);
                                } else {
                                    // we check if this a module and if so, we display a help
                                    // message
                                    let name_path = path.segments.iter()
                                                        .map(|seg| seg.identifier.name)
                                                        .collect::<Vec<_>>();
                                    let current_module = self.current_module.clone();

                                    match self.resolve_module_path(current_module,
                                                   &name_path[..],
                                                   UseLexicalScope,
                                                   expr.span,
                                                   PathSearch) {
                                        Success(_) => {
                                            context = UnresolvedNameContext::PathIsMod(expr.id);
                                        },
                                        _ => {},
                                    };
                                }

                                resolve_error(self,
                                              expr.span,
                                              ResolutionError::UnresolvedName(
                                                  &*path_name, &*msg, context));
                            }
                        }
                    }
                }

                intravisit::walk_expr(self, expr);
            }

            ExprStruct(ref path, _, _) => {
                // Resolve the path to the structure it goes to. We don't
                // check to ensure that the path is actually a structure; that
                // is checked later during typeck.
                match self.resolve_path(expr.id, path, 0, TypeNS, false) {
                    Some(definition) => self.record_def(expr.id, definition),
                    None => {
                        debug!("(resolving expression) didn't find struct def",);

                        resolve_error(self,
                                      path.span,
                                      ResolutionError::DoesNotNameAStruct(
                                                                &*path_names_to_string(path, 0))
                                     );
                        self.record_def(expr.id, err_path_resolution());
                    }
                }

                intravisit::walk_expr(self, expr);
            }

            ExprLoop(_, Some(label)) | ExprWhile(_, _, Some(label)) => {
                self.with_label_rib(|this| {
                    let def_like = DlDef(DefLabel(expr.id));

                    {
                        let rib = this.label_ribs.last_mut().unwrap();
                        rib.bindings.insert(label.name, def_like);
                    }

                    intravisit::walk_expr(this, expr);
                })
            }

            ExprBreak(Some(label)) | ExprAgain(Some(label)) => {
                match self.search_label(label.node.name) {
                    None => {
                        self.record_def(expr.id, err_path_resolution());
                        resolve_error(self,
                                      label.span,
                                      ResolutionError::UndeclaredLabel(&label.node.name.as_str()))
                    }
                    Some(DlDef(def @ DefLabel(_))) => {
                        // Since this def is a label, it is never read.
                        self.record_def(expr.id,
                                        PathResolution {
                                            base_def: def,
                                            last_private: LastMod(AllPublic),
                                            depth: 0,
                                        })
                    }
                    Some(_) => {
                        self.session.span_bug(expr.span, "label wasn't mapped to a label def!")
                    }
                }
            }

            _ => {
                intravisit::walk_expr(self, expr);
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self, expr: &Expr) {
        match expr.node {
            ExprField(_, name) => {
                // FIXME(#6890): Even though you can't treat a method like a
                // field, we need to add any trait methods we find that match
                // the field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.get_traits_containing_item(name.node);
                self.trait_map.insert(expr.id, traits);
            }
            ExprMethodCall(name, _, _) => {
                debug!("(recording candidate traits for expr) recording traits for {}",
                       expr.id);
                let traits = self.get_traits_containing_item(name.node);
                self.trait_map.insert(expr.id, traits);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn get_traits_containing_item(&mut self, name: Name) -> Vec<DefId> {
        debug!("(getting traits containing item) looking for '{}'", name);

        fn add_trait_info(found_traits: &mut Vec<DefId>, trait_def_id: DefId, name: Name) {
            debug!("(adding trait info) found trait {:?} for method '{}'",
                   trait_def_id,
                   name);
            found_traits.push(trait_def_id);
        }

        let mut found_traits = Vec::new();
        let mut search_module = self.current_module.clone();
        loop {
            // Look for the current trait.
            match self.current_trait_ref {
                Some((trait_def_id, _)) => {
                    if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                        add_trait_info(&mut found_traits, trait_def_id, name);
                    }
                }
                None => {} // Nothing to do.
            }

            // Look for trait children.
            build_reduced_graph::populate_module_if_necessary(self, &search_module);

            {
                for (_, child_names) in search_module.children.borrow().iter() {
                    let def = match child_names.type_ns.def() {
                        Some(def) => def,
                        None => continue,
                    };
                    let trait_def_id = match def {
                        DefTrait(trait_def_id) => trait_def_id,
                        _ => continue,
                    };
                    if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                        add_trait_info(&mut found_traits, trait_def_id, name);
                    }
                }
            }

            // Look for imports.
            for (_, import) in search_module.import_resolutions.borrow().iter() {
                let target = match import.type_ns.target {
                    None => continue,
                    Some(ref target) => target,
                };
                let did = match target.binding.def() {
                    Some(DefTrait(trait_def_id)) => trait_def_id,
                    Some(..) | None => continue,
                };
                if self.trait_item_map.contains_key(&(name, did)) {
                    add_trait_info(&mut found_traits, did, name);
                    let id = import.type_ns.id;
                    self.used_imports.insert((id, TypeNS));
                    let trait_name = self.get_trait_name(did);
                    self.record_import_use(id, trait_name);
                    if let Some(DefId{krate: kid, ..}) = target.target_module.def_id() {
                        self.used_crates.insert(kid);
                    }
                }
            }

            match search_module.parent_link.clone() {
                NoParentLink | ModuleParentLink(..) => break,
                BlockParentLink(parent_module, _) => {
                    search_module = parent_module.upgrade().unwrap();
                }
            }
        }

        found_traits
    }

    fn record_def(&mut self, node_id: NodeId, resolution: PathResolution) {
        debug!("(recording def) recording {:?} for {}", resolution, node_id);
        assert!(match resolution.last_private {
                    LastImport{..} => false,
                    _ => true,
                },
                "Import should only be used for `use` directives");

        if let Some(prev_res) = self.def_map.borrow_mut().insert(node_id, resolution) {
            let span = self.ast_map.opt_span(node_id).unwrap_or(codemap::DUMMY_SP);
            self.session.span_bug(span,
                                  &format!("path resolved multiple times ({:?} before, {:?} now)",
                                           prev_res,
                                           resolution));
        }
    }

    fn enforce_default_binding_mode(&mut self,
                                    pat: &Pat,
                                    pat_binding_mode: BindingMode,
                                    descr: &str) {
        match pat_binding_mode {
            BindByValue(_) => {}
            BindByRef(..) => {
                resolve_error(self,
                              pat.span,
                              ResolutionError::CannotUseRefBindingModeWith(descr));
            }
        }
    }

    //
    // Diagnostics
    //
    // Diagnostics are not particularly efficient, because they're rarely
    // hit.
    //

    #[allow(dead_code)]   // useful for debugging
    fn dump_module(&mut self, module_: Rc<Module>) {
        debug!("Dump of module `{}`:", module_to_string(&*module_));

        debug!("Children:");
        build_reduced_graph::populate_module_if_necessary(self, &module_);
        for (&name, _) in module_.children.borrow().iter() {
            debug!("* {}", name);
        }

        debug!("Import resolutions:");
        let import_resolutions = module_.import_resolutions.borrow();
        for (&name, import_resolution) in import_resolutions.iter() {
            let value_repr;
            match import_resolution.value_ns.target {
                None => {
                    value_repr = "".to_string();
                }
                Some(_) => {
                    value_repr = " value:?".to_string();
                    // FIXME #4954
                }
            }

            let type_repr;
            match import_resolution.type_ns.target {
                None => {
                    type_repr = "".to_string();
                }
                Some(_) => {
                    type_repr = " type:?".to_string();
                    // FIXME #4954
                }
            }

            debug!("* {}:{}{}", name, value_repr, type_repr);
        }
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

/// A somewhat inefficient routine to obtain the name of a module.
fn module_to_string(module: &Module) -> String {
    let mut names = Vec::new();

    fn collect_mod(names: &mut Vec<ast::Name>, module: &Module) {
        match module.parent_link {
            NoParentLink => {}
            ModuleParentLink(ref module, name) => {
                names.push(name);
                collect_mod(names, &*module.upgrade().unwrap());
            }
            BlockParentLink(ref module, _) => {
                // danger, shouldn't be ident?
                names.push(special_idents::opaque.name);
                collect_mod(names, &*module.upgrade().unwrap());
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
    PathResolution {
        base_def: DefErr,
        last_private: LastMod(AllPublic),
        depth: 0,
    }
}


pub struct CrateMap {
    pub def_map: RefCell<DefMap>,
    pub freevars: FreevarMap,
    pub export_map: ExportMap,
    pub trait_map: TraitMap,
    pub external_exports: ExternalExports,
    pub glob_map: Option<GlobMap>,
}

#[derive(PartialEq,Copy, Clone)]
pub enum MakeGlobMap {
    Yes,
    No,
}

/// Entry point to crate resolution.
pub fn resolve_crate<'a, 'tcx>(session: &'a Session,
                               ast_map: &'a hir_map::Map<'tcx>,
                               make_glob_map: MakeGlobMap)
                               -> CrateMap {
    let krate = ast_map.krate();
    let mut resolver = create_resolver(session, ast_map, krate, make_glob_map, None);

    resolver.resolve_crate(krate);

    check_unused::check_crate(&mut resolver, krate);

    CrateMap {
        def_map: resolver.def_map,
        freevars: resolver.freevars,
        export_map: resolver.export_map,
        trait_map: resolver.trait_map,
        external_exports: resolver.external_exports,
        glob_map: if resolver.make_glob_map {
            Some(resolver.glob_map)
        } else {
            None
        },
    }
}

/// Builds a name resolution walker to be used within this module,
/// or used externally, with an optional callback function.
///
/// The callback takes a &mut bool which allows callbacks to end a
/// walk when set to true, passing through the rest of the walk, while
/// preserving the ribs + current module. This allows resolve_path
/// calls to be made with the correct scope info. The node in the
/// callback corresponds to the current node in the walk.
pub fn create_resolver<'a, 'tcx>(session: &'a Session,
                                 ast_map: &'a hir_map::Map<'tcx>,
                                 krate: &'a Crate,
                                 make_glob_map: MakeGlobMap,
                                 callback: Option<Box<Fn(hir_map::Node, &mut bool) -> bool>>)
                                 -> Resolver<'a, 'tcx> {
    let mut resolver = Resolver::new(session, ast_map, make_glob_map);

    resolver.callback = callback;

    build_reduced_graph::build_reduced_graph(&mut resolver, krate);
    session.abort_if_errors();

    resolve_imports::resolve_imports(&mut resolver);
    session.abort_if_errors();

    record_exports::record(&mut resolver);
    session.abort_if_errors();

    resolver
}

__build_diagnostic_array! { librustc_resolve, DIAGNOSTICS }
