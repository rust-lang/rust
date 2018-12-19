// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# check.rs

Within the check phase of type check, we check each item one at a time
(bodies of function expressions are checked as part of the containing
function). Inference is used to supply types wherever they are unknown.

By far the most complex case is checking the body of a function. This
can be broken down into several distinct phases:

- gather: creates type variables to represent the type of each local
  variable and pattern binding.

- main: the main pass does the lion's share of the work: it
  determines the types of all expressions, resolves
  methods, checks for most invalid conditions, and so forth.  In
  some cases, where a type is unknown, it may create a type or region
  variable and use that as the type of an expression.

  In the process of checking, various constraints will be placed on
  these type variables through the subtyping relationships requested
  through the `demand` module.  The `infer` module is in charge
  of resolving those constraints.

- regionck: after main is complete, the regionck pass goes over all
  types looking for regions and making sure that they did not escape
  into places they are not in scope.  This may also influence the
  final assignments of the various region variables if there is some
  flexibility.

- vtable: find and records the impls to use for each trait bound that
  appears on a type parameter.

- writeback: writes the final types within a function body, replacing
  type variables with their final inferred types.  These final types
  are written into the `tcx.node_types` table, which should *never* contain
  any reference to a type variable.

## Intermediate types

While type checking a function, the intermediate types for the
expressions, blocks, and so forth contained within the function are
stored in `fcx.node_types` and `fcx.node_substs`.  These types
may contain unresolved type variables.  After type checking is
complete, the functions in the writeback module are used to take the
types from this table, resolve them, and then write them into their
permanent home in the type context `tcx`.

This means that during inferencing you should use `fcx.write_ty()`
and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
nodes within the function.

The types of top-level items, which never contain unbound type
variables, are stored directly into the `tcx` tables.

N.B., a type variable is not the same thing as a type parameter.  A
type variable is rather an "instance" of a type parameter: that is,
given a generic function `fn foo<T>(t: T)`: while checking the
function `foo`, the type `ty_param(0)` refers to the type `T`, which
is treated in abstract.  When `foo()` is called, however, `T` will be
substituted for a fresh type variable `N`.  This variable will
eventually be resolved to some concrete type (which might itself be
type parameter).

*/

mod autoderef;
pub mod dropck;
pub mod _match;
pub mod writeback;
mod regionck;
pub mod coercion;
pub mod demand;
pub mod method;
mod upvar;
mod wfcheck;
mod cast;
mod closure;
mod callee;
mod compare_method;
mod generator_interior;
mod intrinsic;
mod op;

use astconv::AstConv;
use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use rustc::hir::{self, GenericArg, ItemKind, Node, PatKind};
use rustc::hir::def::Def;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use middle::lang_items;
use namespace::Namespace;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::sync::Lrc;
use rustc_target::spec::abi::Abi;
use rustc::infer::{self, InferCtxt, InferOk, RegionVariableOrigin};
use rustc::infer::opaque_types::OpaqueTypeDecl;
use rustc::infer::type_variable::{TypeVariableOrigin};
use rustc::middle::region;
use rustc::mir::interpret::{ConstValue, GlobalId};
use rustc::ty::subst::{CanonicalUserSubsts, UnpackedKind, Subst, Substs,
                       UserSelfTy, UserSubsts};
use rustc::traits::{self, ObligationCause, ObligationCauseCode, TraitEngine};
use rustc::ty::{self, AdtKind, Ty, TyCtxt, GenericParamDefKind, RegionKind, Visibility,
                ToPolyTraitRef, ToPredicate};
use rustc::ty::adjustment::{Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::query::Providers;
use rustc::ty::util::{Representability, IntTypeExt, Discr};
use rustc::ty::layout::VariantIdx;
use syntax_pos::{self, BytePos, Span, MultiSpan};
use syntax::ast;
use syntax::attr;
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax::ptr::P;
use syntax::source_map::{DUMMY_SP, original_sp};
use syntax::symbol::{Symbol, LocalInternedString, keywords};
use syntax::util::lev_distance::find_best_match_for_name;

use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::hash_map::Entry;
use std::cmp;
use std::fmt::Display;
use std::iter;
use std::mem::replace;
use std::ops::{self, Deref};
use std::slice;

use require_c_abi_if_variadic;
use session::{CompileIncomplete, config, Session};
use TypeAndSubsts;
use lint;
use util::captures::Captures;
use util::common::{ErrorReported, indenter};
use util::nodemap::{DefIdMap, DefIdSet, FxHashMap, FxHashSet, NodeMap};

pub use self::Expectation::*;
use self::autoderef::Autoderef;
use self::callee::DeferredCallResolution;
use self::coercion::{CoerceMany, DynamicCoerceMany};
pub use self::compare_method::{compare_impl_method, compare_const_impl};
use self::method::MethodCallee;
use self::TupleArgumentsFlag::*;

/// The type of a local binding, including the revealed type for anon types.
#[derive(Copy, Clone)]
pub struct LocalTy<'tcx> {
    decl_ty: Ty<'tcx>,
    revealed_ty: Ty<'tcx>
}

/// A wrapper for InferCtxt's `in_progress_tables` field.
#[derive(Copy, Clone)]
struct MaybeInProgressTables<'a, 'tcx: 'a> {
    maybe_tables: Option<&'a RefCell<ty::TypeckTables<'tcx>>>,
}

impl<'a, 'tcx> MaybeInProgressTables<'a, 'tcx> {
    fn borrow(self) -> Ref<'a, ty::TypeckTables<'tcx>> {
        match self.maybe_tables {
            Some(tables) => tables.borrow(),
            None => {
                bug!("MaybeInProgressTables: inh/fcx.tables.borrow() with no tables")
            }
        }
    }

    fn borrow_mut(self) -> RefMut<'a, ty::TypeckTables<'tcx>> {
        match self.maybe_tables {
            Some(tables) => tables.borrow_mut(),
            None => {
                bug!("MaybeInProgressTables: inh/fcx.tables.borrow_mut() with no tables")
            }
        }
    }
}

/// closures defined within the function.  For example:
///
///     fn foo() {
///         bar(move|| { ... })
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: InferCtxt<'a, 'gcx, 'tcx>,

    tables: MaybeInProgressTables<'a, 'tcx>,

    locals: RefCell<NodeMap<LocalTy<'tcx>>>,

    fulfillment_cx: RefCell<Box<dyn TraitEngine<'tcx>>>,

    // Some additional `Sized` obligations badly affect type inference.
    // These obligations are added in a later stage of typeck.
    deferred_sized_obligations: RefCell<Vec<(Ty<'tcx>, Span, traits::ObligationCauseCode<'tcx>)>>,

    // When we process a call like `c()` where `c` is a closure type,
    // we may not have decided yet whether `c` is a `Fn`, `FnMut`, or
    // `FnOnce` closure. In that case, we defer full resolution of the
    // call until upvar inference can kick in and make the
    // decision. We keep these deferred resolutions grouped by the
    // def-id of the closure, so that once we decide, we can easily go
    // back and process them.
    deferred_call_resolutions: RefCell<DefIdMap<Vec<DeferredCallResolution<'gcx, 'tcx>>>>,

    deferred_cast_checks: RefCell<Vec<cast::CastCheck<'tcx>>>,

    deferred_generator_interiors: RefCell<Vec<(hir::BodyId, Ty<'tcx>)>>,

    // Opaque types found in explicit return types and their
    // associated fresh inference variable. Writeback resolves these
    // variables to get the concrete type, which can be used to
    // 'de-opaque' OpaqueTypeDecl, after typeck is done with all functions.
    opaque_types: RefCell<DefIdMap<OpaqueTypeDecl<'tcx>>>,

    /// Each type parameter has an implicit region bound that
    /// indicates it must outlive at least the function body (the user
    /// may specify stronger requirements). This field indicates the
    /// region of the callee. If it is `None`, then the parameter
    /// environment is for an item or something where the "callee" is
    /// not clear.
    implicit_region_bound: Option<ty::Region<'tcx>>,

    body_id: Option<hir::BodyId>,
}

impl<'a, 'gcx, 'tcx> Deref for Inherited<'a, 'gcx, 'tcx> {
    type Target = InferCtxt<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

/// When type-checking an expression, we propagate downward
/// whatever type hint we are able in the form of an `Expectation`.
#[derive(Copy, Clone, Debug)]
pub enum Expectation<'tcx> {
    /// We know nothing about what type this expression should have.
    NoExpectation,

    /// This expression is an `if` condition, it must resolve to `bool`.
    ExpectIfCondition,

    /// This expression should have the type given (or some subtype)
    ExpectHasType(Ty<'tcx>),

    /// This expression will be cast to the `Ty`
    ExpectCastableToType(Ty<'tcx>),

    /// This rvalue expression will be wrapped in `&` or `Box` and coerced
    /// to `&Ty` or `Box<Ty>`, respectively. `Ty` is `[A]` or `Trait`.
    ExpectRvalueLikeUnsized(Ty<'tcx>),
}

impl<'a, 'gcx, 'tcx> Expectation<'tcx> {
    // Disregard "castable to" expectations because they
    // can lead us astray. Consider for example `if cond
    // {22} else {c} as u8` -- if we propagate the
    // "castable to u8" constraint to 22, it will pick the
    // type 22u8, which is overly constrained (c might not
    // be a u8). In effect, the problem is that the
    // "castable to" expectation is not the tightest thing
    // we can say, so we want to drop it in this case.
    // The tightest thing we can say is "must unify with
    // else branch". Note that in the case of a "has type"
    // constraint, this limitation does not hold.

    // If the expected type is just a type variable, then don't use
    // an expected type. Otherwise, we might write parts of the type
    // when checking the 'then' block which are incompatible with the
    // 'else' branch.
    fn adjust_for_branches(&self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Expectation<'tcx> {
        match *self {
            ExpectHasType(ety) => {
                let ety = fcx.shallow_resolve(ety);
                if !ety.is_ty_var() {
                    ExpectHasType(ety)
                } else {
                    NoExpectation
                }
            }
            ExpectRvalueLikeUnsized(ety) => {
                ExpectRvalueLikeUnsized(ety)
            }
            _ => NoExpectation
        }
    }

    /// Provide an expectation for an rvalue expression given an *optional*
    /// hint, which is not required for type safety (the resulting type might
    /// be checked higher up, as is the case with `&expr` and `box expr`), but
    /// is useful in determining the concrete type.
    ///
    /// The primary use case is where the expected type is a fat pointer,
    /// like `&[isize]`. For example, consider the following statement:
    ///
    ///    let x: &[isize] = &[1, 2, 3];
    ///
    /// In this case, the expected type for the `&[1, 2, 3]` expression is
    /// `&[isize]`. If however we were to say that `[1, 2, 3]` has the
    /// expectation `ExpectHasType([isize])`, that would be too strong --
    /// `[1, 2, 3]` does not have the type `[isize]` but rather `[isize; 3]`.
    /// It is only the `&[1, 2, 3]` expression as a whole that can be coerced
    /// to the type `&[isize]`. Therefore, we propagate this more limited hint,
    /// which still is useful, because it informs integer literals and the like.
    /// See the test case `test/run-pass/coerce-expect-unsized.rs` and #20169
    /// for examples of where this comes up,.
    fn rvalue_hint(fcx: &FnCtxt<'a, 'gcx, 'tcx>, ty: Ty<'tcx>) -> Expectation<'tcx> {
        match fcx.tcx.struct_tail(ty).sty {
            ty::Slice(_) | ty::Str | ty::Dynamic(..) => {
                ExpectRvalueLikeUnsized(ty)
            }
            _ => ExpectHasType(ty)
        }
    }

    // Resolves `expected` by a single level if it is a variable. If
    // there is no expected type or resolution is not possible (e.g.,
    // no constraints yet present), just returns `None`.
    fn resolve(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Expectation<'tcx> {
        match self {
            NoExpectation => NoExpectation,
            ExpectIfCondition => ExpectIfCondition,
            ExpectCastableToType(t) => {
                ExpectCastableToType(fcx.resolve_type_vars_if_possible(&t))
            }
            ExpectHasType(t) => {
                ExpectHasType(fcx.resolve_type_vars_if_possible(&t))
            }
            ExpectRvalueLikeUnsized(t) => {
                ExpectRvalueLikeUnsized(fcx.resolve_type_vars_if_possible(&t))
            }
        }
    }

    fn to_option(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            NoExpectation => None,
            ExpectIfCondition => Some(fcx.tcx.types.bool),
            ExpectCastableToType(ty) |
            ExpectHasType(ty) |
            ExpectRvalueLikeUnsized(ty) => Some(ty),
        }
    }

    /// It sometimes happens that we want to turn an expectation into
    /// a **hard constraint** (i.e., something that must be satisfied
    /// for the program to type-check). `only_has_type` will return
    /// such a constraint, if it exists.
    fn only_has_type(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            ExpectHasType(ty) => Some(ty),
            ExpectIfCondition => Some(fcx.tcx.types.bool),
            NoExpectation | ExpectCastableToType(_) | ExpectRvalueLikeUnsized(_) => None,
        }
    }

    /// Like `only_has_type`, but instead of returning `None` if no
    /// hard constraint exists, creates a fresh type variable.
    fn coercion_target_type(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>, span: Span) -> Ty<'tcx> {
        self.only_has_type(fcx)
            .unwrap_or_else(|| fcx.next_ty_var(TypeVariableOrigin::MiscVariable(span)))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Needs {
    MutPlace,
    None
}

impl Needs {
    fn maybe_mut_place(m: hir::Mutability) -> Self {
        match m {
            hir::MutMutable => Needs::MutPlace,
            hir::MutImmutable => Needs::None,
        }
    }
}

#[derive(Copy, Clone)]
pub struct UnsafetyState {
    pub def: ast::NodeId,
    pub unsafety: hir::Unsafety,
    pub unsafe_push_count: u32,
    from_fn: bool
}

impl UnsafetyState {
    pub fn function(unsafety: hir::Unsafety, def: ast::NodeId) -> UnsafetyState {
        UnsafetyState { def: def, unsafety: unsafety, unsafe_push_count: 0, from_fn: true }
    }

    pub fn recurse(&mut self, blk: &hir::Block) -> UnsafetyState {
        match self.unsafety {
            // If this unsafe, then if the outer function was already marked as
            // unsafe we shouldn't attribute the unsafe'ness to the block. This
            // way the block can be warned about instead of ignoring this
            // extraneous block (functions are never warned about).
            hir::Unsafety::Unsafe if self.from_fn => *self,

            unsafety => {
                let (unsafety, def, count) = match blk.rules {
                    hir::PushUnsafeBlock(..) =>
                        (unsafety, blk.id, self.unsafe_push_count.checked_add(1).unwrap()),
                    hir::PopUnsafeBlock(..) =>
                        (unsafety, blk.id, self.unsafe_push_count.checked_sub(1).unwrap()),
                    hir::UnsafeBlock(..) =>
                        (hir::Unsafety::Unsafe, blk.id, self.unsafe_push_count),
                    hir::DefaultBlock =>
                        (unsafety, self.def, self.unsafe_push_count),
                };
                UnsafetyState{ def,
                               unsafety,
                               unsafe_push_count: count,
                               from_fn: false }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum PlaceOp {
    Deref,
    Index
}

/// Tracks whether executing a node may exit normally (versus
/// return/break/panic, which "diverge", leaving dead code in their
/// wake). Tracked semi-automatically (through type variables marked
/// as diverging), with some manual adjustments for control-flow
/// primitives (approximating a CFG).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Diverges {
    /// Potentially unknown, some cases converge,
    /// others require a CFG to determine them.
    Maybe,

    /// Definitely known to diverge and therefore
    /// not reach the next sibling or its parent.
    Always,

    /// Same as `Always` but with a reachability
    /// warning already emitted
    WarnedAlways
}

// Convenience impls for combinig `Diverges`.

impl ops::BitAnd for Diverges {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        cmp::min(self, other)
    }
}

impl ops::BitOr for Diverges {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        cmp::max(self, other)
    }
}

impl ops::BitAndAssign for Diverges {
    fn bitand_assign(&mut self, other: Self) {
        *self = *self & other;
    }
}

impl ops::BitOrAssign for Diverges {
    fn bitor_assign(&mut self, other: Self) {
        *self = *self | other;
    }
}

impl Diverges {
    fn always(self) -> bool {
        self >= Diverges::Always
    }
}

pub struct BreakableCtxt<'gcx: 'tcx, 'tcx> {
    may_break: bool,

    // this is `null` for loops where break with a value is illegal,
    // such as `while`, `for`, and `while let`
    coerce: Option<DynamicCoerceMany<'gcx, 'tcx>>,
}

pub struct EnclosingBreakables<'gcx: 'tcx, 'tcx> {
    stack: Vec<BreakableCtxt<'gcx, 'tcx>>,
    by_id: NodeMap<usize>,
}

impl<'gcx, 'tcx> EnclosingBreakables<'gcx, 'tcx> {
    fn find_breakable(&mut self, target_id: ast::NodeId) -> &mut BreakableCtxt<'gcx, 'tcx> {
        let ix = *self.by_id.get(&target_id).unwrap_or_else(|| {
            bug!("could not find enclosing breakable with id {}", target_id);
        });
        &mut self.stack[ix]
    }
}

#[derive(Debug)]
struct PathSeg(DefId, usize);

pub struct FnCtxt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    body_id: ast::NodeId,

    /// The parameter environment used for proving trait obligations
    /// in this function. This can change when we descend into
    /// closures (as they bring new things into scope), hence it is
    /// not part of `Inherited` (as of the time of this writing,
    /// closures do not yet change the environment, but they will
    /// eventually).
    param_env: ty::ParamEnv<'tcx>,

    // Number of errors that had been reported when we started
    // checking this function. On exit, if we find that *more* errors
    // have been reported, we will skip regionck and other work that
    // expects the types within the function to be consistent.
    err_count_on_creation: usize,

    ret_coercion: Option<RefCell<DynamicCoerceMany<'gcx, 'tcx>>>,

    yield_ty: Option<Ty<'tcx>>,

    ps: RefCell<UnsafetyState>,

    /// Whether the last checked node generates a divergence (e.g.,
    /// `return` will set this to Always). In general, when entering
    /// an expression or other node in the tree, the initial value
    /// indicates whether prior parts of the containing expression may
    /// have diverged. It is then typically set to `Maybe` (and the
    /// old value remembered) for processing the subparts of the
    /// current expression. As each subpart is processed, they may set
    /// the flag to `Always` etc.  Finally, at the end, we take the
    /// result and "union" it with the original value, so that when we
    /// return the flag indicates if any subpart of the parent
    /// expression (up to and including this part) has diverged.  So,
    /// if you read it after evaluating a subexpression `X`, the value
    /// you get indicates whether any subexpression that was
    /// evaluating up to and including `X` diverged.
    ///
    /// We currently use this flag only for diagnostic purposes:
    ///
    /// - To warn about unreachable code: if, after processing a
    ///   sub-expression but before we have applied the effects of the
    ///   current node, we see that the flag is set to `Always`, we
    ///   can issue a warning. This corresponds to something like
    ///   `foo(return)`; we warn on the `foo()` expression. (We then
    ///   update the flag to `WarnedAlways` to suppress duplicate
    ///   reports.) Similarly, if we traverse to a fresh statement (or
    ///   tail expression) from a `Always` setting, we will issue a
    ///   warning. This corresponds to something like `{return;
    ///   foo();}` or `{return; 22}`, where we would warn on the
    ///   `foo()` or `22`.
    ///
    /// An expression represents dead-code if, after checking it,
    /// the diverges flag is set to something other than `Maybe`.
    diverges: Cell<Diverges>,

    /// Whether any child nodes have any type errors.
    has_errors: Cell<bool>,

    enclosing_breakables: RefCell<EnclosingBreakables<'gcx, 'tcx>>,

    inh: &'a Inherited<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> Deref for FnCtxt<'a, 'gcx, 'tcx> {
    type Target = Inherited<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.inh
    }
}

/// Helper type of a temporary returned by Inherited::build(...).
/// Necessary because we can't write the following bound:
/// F: for<'b, 'tcx> where 'gcx: 'tcx FnOnce(Inherited<'b, 'gcx, 'tcx>).
pub struct InheritedBuilder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: infer::InferCtxtBuilder<'a, 'gcx, 'tcx>,
    def_id: DefId,
}

impl<'a, 'gcx, 'tcx> Inherited<'a, 'gcx, 'tcx> {
    pub fn build(tcx: TyCtxt<'a, 'gcx, 'gcx>, def_id: DefId)
                 -> InheritedBuilder<'a, 'gcx, 'tcx> {
        let hir_id_root = if def_id.is_local() {
            let node_id = tcx.hir().as_local_node_id(def_id).unwrap();
            let hir_id = tcx.hir().definitions().node_to_hir_id(node_id);
            DefId::local(hir_id.owner)
        } else {
            def_id
        };

        InheritedBuilder {
            infcx: tcx.infer_ctxt().with_fresh_in_progress_tables(hir_id_root),
            def_id,
        }
    }
}

impl<'a, 'gcx, 'tcx> InheritedBuilder<'a, 'gcx, 'tcx> {
    fn enter<F, R>(&'tcx mut self, f: F) -> R
        where F: for<'b> FnOnce(Inherited<'b, 'gcx, 'tcx>) -> R
    {
        let def_id = self.def_id;
        self.infcx.enter(|infcx| f(Inherited::new(infcx, def_id)))
    }
}

impl<'a, 'gcx, 'tcx> Inherited<'a, 'gcx, 'tcx> {
    fn new(infcx: InferCtxt<'a, 'gcx, 'tcx>, def_id: DefId) -> Self {
        let tcx = infcx.tcx;
        let item_id = tcx.hir().as_local_node_id(def_id);
        let body_id = item_id.and_then(|id| tcx.hir().maybe_body_owned_by(id));
        let implicit_region_bound = body_id.map(|body_id| {
            let body = tcx.hir().body(body_id);
            tcx.mk_region(ty::ReScope(region::Scope {
                id: body.value.hir_id.local_id,
                data: region::ScopeData::CallSite
            }))
        });

        Inherited {
            tables: MaybeInProgressTables {
                maybe_tables: infcx.in_progress_tables,
            },
            infcx,
            fulfillment_cx: RefCell::new(TraitEngine::new(tcx)),
            locals: RefCell::new(Default::default()),
            deferred_sized_obligations: RefCell::new(Vec::new()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_generator_interiors: RefCell::new(Vec::new()),
            opaque_types: RefCell::new(Default::default()),
            implicit_region_bound,
            body_id,
        }
    }

    fn register_predicate(&self, obligation: traits::PredicateObligation<'tcx>) {
        debug!("register_predicate({:?})", obligation);
        if obligation.has_escaping_bound_vars() {
            span_bug!(obligation.cause.span, "escaping bound vars in predicate {:?}",
                      obligation);
        }
        self.fulfillment_cx
            .borrow_mut()
            .register_predicate_obligation(self, obligation);
    }

    fn register_predicates<I>(&self, obligations: I)
        where I: IntoIterator<Item = traits::PredicateObligation<'tcx>>
    {
        for obligation in obligations {
            self.register_predicate(obligation);
        }
    }

    fn register_infer_ok_obligations<T>(&self, infer_ok: InferOk<'tcx, T>) -> T {
        self.register_predicates(infer_ok.obligations);
        infer_ok.value
    }

    fn normalize_associated_types_in<T>(&self,
                                        span: Span,
                                        body_id: ast::NodeId,
                                        param_env: ty::ParamEnv<'tcx>,
                                        value: &T) -> T
        where T : TypeFoldable<'tcx>
    {
        let ok = self.partially_normalize_associated_types_in(span, body_id, param_env, value);
        self.register_infer_ok_obligations(ok)
    }
}

struct CheckItemTypesVisitor<'a, 'tcx: 'a> { tcx: TyCtxt<'a, 'tcx, 'tcx> }

impl<'a, 'tcx> ItemLikeVisitor<'tcx> for CheckItemTypesVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &'tcx hir::Item) {
        check_item_type(self.tcx, i);
    }
    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem) { }
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem) { }
}

pub fn check_wf_new<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Result<(), ErrorReported> {
    tcx.sess.track_errors(|| {
        let mut visit = wfcheck::CheckTypeWellFormedVisitor::new(tcx);
        tcx.hir().krate().visit_all_item_likes(&mut visit.as_deep_visitor());
    })
}

pub fn check_item_types<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Result<(), ErrorReported> {
    tcx.sess.track_errors(|| {
        tcx.hir().krate().visit_all_item_likes(&mut CheckItemTypesVisitor { tcx });
    })
}

pub fn check_item_bodies<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Result<(), CompileIncomplete> {
    tcx.typeck_item_bodies(LOCAL_CRATE)
}

fn typeck_item_bodies<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, crate_num: CrateNum)
                                -> Result<(), CompileIncomplete>
{
    debug_assert!(crate_num == LOCAL_CRATE);
    Ok(tcx.sess.track_errors(|| {
        tcx.par_body_owners(|body_owner_def_id| {
            ty::query::queries::typeck_tables_of::ensure(tcx, body_owner_def_id);
        });
    })?)
}

fn check_item_well_formed<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    wfcheck::check_item_well_formed(tcx, def_id);
}

fn check_trait_item_well_formed<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    wfcheck::check_trait_item(tcx, def_id);
}

fn check_impl_item_well_formed<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    wfcheck::check_impl_item(tcx, def_id);
}

pub fn provide(providers: &mut Providers) {
    method::provide(providers);
    *providers = Providers {
        typeck_item_bodies,
        typeck_tables_of,
        has_typeck_tables,
        adt_destructor,
        used_trait_imports,
        check_item_well_formed,
        check_trait_item_well_formed,
        check_impl_item_well_formed,
        ..*providers
    };
}

fn adt_destructor<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            def_id: DefId)
                            -> Option<ty::Destructor> {
    tcx.calculate_dtor(def_id, &mut dropck::check_drop_impl)
}

/// If this def-id is a "primary tables entry", returns `Some((body_id, decl))`
/// with information about it's body-id and fn-decl (if any). Otherwise,
/// returns `None`.
///
/// If this function returns "some", then `typeck_tables(def_id)` will
/// succeed; if it returns `None`, then `typeck_tables(def_id)` may or
/// may not succeed.  In some cases where this function returns `None`
/// (notably closures), `typeck_tables(def_id)` would wind up
/// redirecting to the owning function.
fn primary_body_of<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             id: ast::NodeId)
                             -> Option<(hir::BodyId, Option<&'tcx hir::FnDecl>)>
{
    match tcx.hir().get(id) {
        Node::Item(item) => {
            match item.node {
                hir::ItemKind::Const(_, body) |
                hir::ItemKind::Static(_, _, body) =>
                    Some((body, None)),
                hir::ItemKind::Fn(ref decl, .., body) =>
                    Some((body, Some(decl))),
                _ =>
                    None,
            }
        }
        Node::TraitItem(item) => {
            match item.node {
                hir::TraitItemKind::Const(_, Some(body)) =>
                    Some((body, None)),
                hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Provided(body)) =>
                    Some((body, Some(&sig.decl))),
                _ =>
                    None,
            }
        }
        Node::ImplItem(item) => {
            match item.node {
                hir::ImplItemKind::Const(_, body) =>
                    Some((body, None)),
                hir::ImplItemKind::Method(ref sig, body) =>
                    Some((body, Some(&sig.decl))),
                _ =>
                    None,
            }
        }
        Node::AnonConst(constant) => Some((constant.body, None)),
        _ => None,
    }
}

fn has_typeck_tables<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               def_id: DefId)
                               -> bool {
    // Closures' tables come from their outermost function,
    // as they are part of the same "inference environment".
    let outer_def_id = tcx.closure_base_def_id(def_id);
    if outer_def_id != def_id {
        return tcx.has_typeck_tables(outer_def_id);
    }

    let id = tcx.hir().as_local_node_id(def_id).unwrap();
    primary_body_of(tcx, id).is_some()
}

fn used_trait_imports<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                def_id: DefId)
                                -> Lrc<DefIdSet> {
    tcx.typeck_tables_of(def_id).used_trait_imports.clone()
}

fn typeck_tables_of<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              def_id: DefId)
                              -> &'tcx ty::TypeckTables<'tcx> {
    // Closures' tables come from their outermost function,
    // as they are part of the same "inference environment".
    let outer_def_id = tcx.closure_base_def_id(def_id);
    if outer_def_id != def_id {
        return tcx.typeck_tables_of(outer_def_id);
    }

    let id = tcx.hir().as_local_node_id(def_id).unwrap();
    let span = tcx.hir().span(id);

    // Figure out what primary body this item has.
    let (body_id, fn_decl) = primary_body_of(tcx, id).unwrap_or_else(|| {
        span_bug!(span, "can't type-check body of {:?}", def_id);
    });
    let body = tcx.hir().body(body_id);

    let tables = Inherited::build(tcx, def_id).enter(|inh| {
        let param_env = tcx.param_env(def_id);
        let fcx = if let Some(decl) = fn_decl {
            let fn_sig = tcx.fn_sig(def_id);

            check_abi(tcx, span, fn_sig.abi());

            // Compute the fty from point of view of inside the fn.
            let fn_sig =
                tcx.liberate_late_bound_regions(def_id, &fn_sig);
            let fn_sig =
                inh.normalize_associated_types_in(body.value.span,
                                                  body_id.node_id,
                                                  param_env,
                                                  &fn_sig);

            let fcx = check_fn(&inh, param_env, fn_sig, decl, id, body, None).0;
            fcx
        } else {
            let fcx = FnCtxt::new(&inh, param_env, body.value.id);
            let expected_type = tcx.type_of(def_id);
            let expected_type = fcx.normalize_associated_types_in(body.value.span, &expected_type);
            fcx.require_type_is_sized(expected_type, body.value.span, traits::ConstSized);

            let revealed_ty = if tcx.features().impl_trait_in_bindings {
                fcx.instantiate_opaque_types_from_value(
                    id,
                    &expected_type
                )
            } else {
                expected_type
            };

            // Gather locals in statics (because of block expressions).
            GatherLocalsVisitor { fcx: &fcx, parent_id: id, }.visit_body(body);

            fcx.check_expr_coercable_to_type(&body.value, revealed_ty);

            fcx
        };

        // All type checking constraints were added, try to fallback unsolved variables.
        fcx.select_obligations_where_possible(false);
        let mut fallback_has_occurred = false;
        for ty in &fcx.unsolved_variables() {
            fallback_has_occurred |= fcx.fallback_if_possible(ty);
        }
        fcx.select_obligations_where_possible(fallback_has_occurred);

        // Even though coercion casts provide type hints, we check casts after fallback for
        // backwards compatibility. This makes fallback a stronger type hint than a cast coercion.
        fcx.check_casts();

        // Closure and generator analysis may run after fallback
        // because they don't constrain other type variables.
        fcx.closure_analyze(body);
        assert!(fcx.deferred_call_resolutions.borrow().is_empty());
        fcx.resolve_generator_interiors(def_id);

        for (ty, span, code) in fcx.deferred_sized_obligations.borrow_mut().drain(..) {
            let ty = fcx.normalize_ty(span, ty);
            fcx.require_type_is_sized(ty, span, code);
        }
        fcx.select_all_obligations_or_error();

        if fn_decl.is_some() {
            fcx.regionck_fn(id, body);
        } else {
            fcx.regionck_expr(body);
        }

        fcx.resolve_type_vars_in_body(body)
    });

    // Consistency check our TypeckTables instance can hold all ItemLocalIds
    // it will need to hold.
    assert_eq!(tables.local_id_root,
               Some(DefId::local(tcx.hir().definitions().node_to_hir_id(id).owner)));
    tables
}

fn check_abi<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, span: Span, abi: Abi) {
    if !tcx.sess.target.target.is_abi_supported(abi) {
        struct_span_err!(tcx.sess, span, E0570,
            "The ABI `{}` is not supported for the current target", abi).emit()
    }
}

struct GatherLocalsVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    parent_id: ast::NodeId,
}

impl<'a, 'gcx, 'tcx> GatherLocalsVisitor<'a, 'gcx, 'tcx> {
    fn assign(&mut self, span: Span, nid: ast::NodeId, ty_opt: Option<LocalTy<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // infer the variable's type
                let var_ty = self.fcx.next_ty_var(TypeVariableOrigin::TypeInference(span));
                self.fcx.locals.borrow_mut().insert(nid, LocalTy {
                    decl_ty: var_ty,
                    revealed_ty: var_ty
                });
                var_ty
            }
            Some(typ) => {
                // take type that the user specified
                self.fcx.locals.borrow_mut().insert(nid, typ);
                typ.revealed_ty
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for GatherLocalsVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'gcx hir::Local) {
        let local_ty = match local.ty {
            Some(ref ty) => {
                let o_ty = self.fcx.to_ty(&ty);

                let revealed_ty = if self.fcx.tcx.features().impl_trait_in_bindings {
                    self.fcx.instantiate_opaque_types_from_value(
                        self.parent_id,
                        &o_ty
                    )
                } else {
                    o_ty
                };

                let c_ty = self.fcx.inh.infcx.canonicalize_user_type_annotation(&revealed_ty);
                debug!("visit_local: ty.hir_id={:?} o_ty={:?} revealed_ty={:?} c_ty={:?}",
                       ty.hir_id, o_ty, revealed_ty, c_ty);
                self.fcx.tables.borrow_mut().user_provided_tys_mut().insert(ty.hir_id, c_ty);

                Some(LocalTy { decl_ty: o_ty, revealed_ty })
            },
            None => None,
        };
        self.assign(local.span, local.id, local_ty);

        debug!("Local variable {:?} is assigned type {}",
               local.pat,
               self.fcx.ty_to_string(
                   self.fcx.locals.borrow().get(&local.id).unwrap().clone().decl_ty));
        intravisit::walk_local(self, local);
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'gcx hir::Pat) {
        if let PatKind::Binding(_, _, ident, _) = p.node {
            let var_ty = self.assign(p.span, p.id, None);

            if !self.fcx.tcx.features().unsized_locals {
                self.fcx.require_type_is_sized(var_ty, p.span,
                                               traits::VariableType(p.id));
            }

            debug!("Pattern binding {} is assigned to {} with type {:?}",
                   ident,
                   self.fcx.ty_to_string(
                       self.fcx.locals.borrow().get(&p.id).unwrap().clone().decl_ty),
                   var_ty);
        }
        intravisit::walk_pat(self, p);
    }

    // Don't descend into the bodies of nested closures
    fn visit_fn(&mut self, _: intravisit::FnKind<'gcx>, _: &'gcx hir::FnDecl,
                _: hir::BodyId, _: Span, _: ast::NodeId) { }
}

/// When `check_fn` is invoked on a generator (i.e., a body that
/// includes yield), it returns back some information about the yield
/// points.
struct GeneratorTypes<'tcx> {
    /// Type of value that is yielded.
    yield_ty: ty::Ty<'tcx>,

    /// Types that are captured (see `GeneratorInterior` for more).
    interior: ty::Ty<'tcx>,

    /// Indicates if the generator is movable or static (immovable)
    movability: hir::GeneratorMovability,
}

/// Helper used for fns and closures. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
fn check_fn<'a, 'gcx, 'tcx>(inherited: &'a Inherited<'a, 'gcx, 'tcx>,
                            param_env: ty::ParamEnv<'tcx>,
                            fn_sig: ty::FnSig<'tcx>,
                            decl: &'gcx hir::FnDecl,
                            fn_id: ast::NodeId,
                            body: &'gcx hir::Body,
                            can_be_generator: Option<hir::GeneratorMovability>)
                            -> (FnCtxt<'a, 'gcx, 'tcx>, Option<GeneratorTypes<'tcx>>)
{
    let mut fn_sig = fn_sig.clone();

    debug!("check_fn(sig={:?}, fn_id={}, param_env={:?})", fn_sig, fn_id, param_env);

    // Create the function context.  This is either derived from scratch or,
    // in the case of closures, based on the outer context.
    let mut fcx = FnCtxt::new(inherited, param_env, body.value.id);
    *fcx.ps.borrow_mut() = UnsafetyState::function(fn_sig.unsafety, fn_id);

    let declared_ret_ty = fn_sig.output();
    fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
    let revealed_ret_ty = fcx.instantiate_opaque_types_from_value(fn_id, &declared_ret_ty);
    fcx.ret_coercion = Some(RefCell::new(CoerceMany::new(revealed_ret_ty)));
    fn_sig = fcx.tcx.mk_fn_sig(
        fn_sig.inputs().iter().cloned(),
        revealed_ret_ty,
        fn_sig.variadic,
        fn_sig.unsafety,
        fn_sig.abi
    );

    let span = body.value.span;

    if body.is_generator && can_be_generator.is_some() {
        let yield_ty = fcx.next_ty_var(TypeVariableOrigin::TypeInference(span));
        fcx.require_type_is_sized(yield_ty, span, traits::SizedYieldType);
        fcx.yield_ty = Some(yield_ty);
    }

    let outer_def_id = fcx.tcx.closure_base_def_id(fcx.tcx.hir().local_def_id(fn_id));
    let outer_node_id = fcx.tcx.hir().as_local_node_id(outer_def_id).unwrap();
    GatherLocalsVisitor { fcx: &fcx, parent_id: outer_node_id, }.visit_body(body);

    // Add formal parameters.
    for (arg_ty, arg) in fn_sig.inputs().iter().zip(&body.arguments) {
        // Check the pattern.
        fcx.check_pat_walk(&arg.pat, arg_ty,
            ty::BindingMode::BindByValue(hir::Mutability::MutImmutable), true);

        // Check that argument is Sized.
        // The check for a non-trivial pattern is a hack to avoid duplicate warnings
        // for simple cases like `fn foo(x: Trait)`,
        // where we would error once on the parameter as a whole, and once on the binding `x`.
        if arg.pat.simple_ident().is_none() && !fcx.tcx.features().unsized_locals {
            fcx.require_type_is_sized(arg_ty, decl.output.span(), traits::SizedArgumentType);
        }

        fcx.write_ty(arg.hir_id, arg_ty);
    }

    let fn_hir_id = fcx.tcx.hir().node_to_hir_id(fn_id);
    inherited.tables.borrow_mut().liberated_fn_sigs_mut().insert(fn_hir_id, fn_sig);

    fcx.check_return_expr(&body.value);

    // We insert the deferred_generator_interiors entry after visiting the body.
    // This ensures that all nested generators appear before the entry of this generator.
    // resolve_generator_interiors relies on this property.
    let gen_ty = if can_be_generator.is_some() && body.is_generator {
        let interior = fcx.next_ty_var(TypeVariableOrigin::MiscVariable(span));
        fcx.deferred_generator_interiors.borrow_mut().push((body.id(), interior));
        Some(GeneratorTypes {
            yield_ty: fcx.yield_ty.unwrap(),
            interior,
            movability: can_be_generator.unwrap(),
        })
    } else {
        None
    };

    // Finalize the return check by taking the LUB of the return types
    // we saw and assigning it to the expected return type. This isn't
    // really expected to fail, since the coercions would have failed
    // earlier when trying to find a LUB.
    //
    // However, the behavior around `!` is sort of complex. In the
    // event that the `actual_return_ty` comes back as `!`, that
    // indicates that the fn either does not return or "returns" only
    // values of type `!`. In this case, if there is an expected
    // return type that is *not* `!`, that should be ok. But if the
    // return type is being inferred, we want to "fallback" to `!`:
    //
    //     let x = move || panic!();
    //
    // To allow for that, I am creating a type variable with diverging
    // fallback. This was deemed ever so slightly better than unifying
    // the return value with `!` because it allows for the caller to
    // make more assumptions about the return type (e.g., they could do
    //
    //     let y: Option<u32> = Some(x());
    //
    // which would then cause this return type to become `u32`, not
    // `!`).
    let coercion = fcx.ret_coercion.take().unwrap().into_inner();
    let mut actual_return_ty = coercion.complete(&fcx);
    if actual_return_ty.is_never() {
        actual_return_ty = fcx.next_diverging_ty_var(
            TypeVariableOrigin::DivergingFn(span));
    }
    fcx.demand_suptype(span, revealed_ret_ty, actual_return_ty);

    // Check that the main return type implements the termination trait.
    if let Some(term_id) = fcx.tcx.lang_items().termination() {
        if let Some((id, _, entry_type)) = *fcx.tcx.sess.entry_fn.borrow() {
            if id == fn_id {
                if let config::EntryFnType::Main = entry_type {
                    let substs = fcx.tcx.mk_substs_trait(declared_ret_ty, &[]);
                    let trait_ref = ty::TraitRef::new(term_id, substs);
                    let return_ty_span = decl.output.span();
                    let cause = traits::ObligationCause::new(
                        return_ty_span, fn_id, ObligationCauseCode::MainFunctionType);

                    inherited.register_predicate(
                        traits::Obligation::new(
                            cause, param_env, trait_ref.to_predicate()));
                }
            }
        }
    }

    // Check that a function marked as `#[panic_handler]` has signature `fn(&PanicInfo) -> !`
    if let Some(panic_impl_did) = fcx.tcx.lang_items().panic_impl() {
        if panic_impl_did == fcx.tcx.hir().local_def_id(fn_id) {
            if let Some(panic_info_did) = fcx.tcx.lang_items().panic_info() {
                // at this point we don't care if there are duplicate handlers or if the handler has
                // the wrong signature as this value we'll be used when writing metadata and that
                // only happens if compilation succeeded
                fcx.tcx.sess.has_panic_handler.try_set_same(true);

                if declared_ret_ty.sty != ty::Never {
                    fcx.tcx.sess.span_err(
                        decl.output.span(),
                        "return type should be `!`",
                    );
                }

                let inputs = fn_sig.inputs();
                let span = fcx.tcx.hir().span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_panic_info = match inputs[0].sty {
                        ty::Ref(region, ty, mutbl) => match ty.sty {
                            ty::Adt(ref adt, _) => {
                                adt.did == panic_info_did &&
                                    mutbl == hir::Mutability::MutImmutable &&
                                    *region != RegionKind::ReStatic
                            },
                            _ => false,
                        },
                        _ => false,
                    };

                    if !arg_is_panic_info {
                        fcx.tcx.sess.span_err(
                            decl.inputs[0].span,
                            "argument should be `&PanicInfo`",
                        );
                    }

                    if let Node::Item(item) = fcx.tcx.hir().get(fn_id) {
                        if let ItemKind::Fn(_, _, ref generics, _) = item.node {
                            if !generics.params.is_empty() {
                                fcx.tcx.sess.span_err(
                                    span,
                                    "should have no type parameters",
                                );
                            }
                        }
                    }
                } else {
                    let span = fcx.tcx.sess.source_map().def_span(span);
                    fcx.tcx.sess.span_err(span, "function should have one argument");
                }
            } else {
                fcx.tcx.sess.err("language item required, but not found: `panic_info`");
            }
        }
    }

    // Check that a function marked as `#[alloc_error_handler]` has signature `fn(Layout) -> !`
    if let Some(alloc_error_handler_did) = fcx.tcx.lang_items().oom() {
        if alloc_error_handler_did == fcx.tcx.hir().local_def_id(fn_id) {
            if let Some(alloc_layout_did) = fcx.tcx.lang_items().alloc_layout() {
                if declared_ret_ty.sty != ty::Never {
                    fcx.tcx.sess.span_err(
                        decl.output.span(),
                        "return type should be `!`",
                    );
                }

                let inputs = fn_sig.inputs();
                let span = fcx.tcx.hir().span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_alloc_layout = match inputs[0].sty {
                        ty::Adt(ref adt, _) => {
                            adt.did == alloc_layout_did
                        },
                        _ => false,
                    };

                    if !arg_is_alloc_layout {
                        fcx.tcx.sess.span_err(
                            decl.inputs[0].span,
                            "argument should be `Layout`",
                        );
                    }

                    if let Node::Item(item) = fcx.tcx.hir().get(fn_id) {
                        if let ItemKind::Fn(_, _, ref generics, _) = item.node {
                            if !generics.params.is_empty() {
                                fcx.tcx.sess.span_err(
                                    span,
                                    "`#[alloc_error_handler]` function should have no type \
                                     parameters",
                                );
                            }
                        }
                    }
                } else {
                    let span = fcx.tcx.sess.source_map().def_span(span);
                    fcx.tcx.sess.span_err(span, "function should have one argument");
                }
            } else {
                fcx.tcx.sess.err("language item required, but not found: `alloc_layout`");
            }
        }
    }

    (fcx, gen_ty)
}

fn check_struct<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          id: ast::NodeId,
                          span: Span) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);

    if def.repr.simd() {
        check_simd(tcx, span, def_id);
    }

    check_transparent(tcx, span, def_id);
    check_packed(tcx, span, def_id);
}

fn check_union<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         id: ast::NodeId,
                         span: Span) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);

    check_packed(tcx, span, def_id);
}

pub fn check_item_type<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, it: &'tcx hir::Item) {
    debug!(
        "check_item_type(it.id={}, it.name={})",
        it.id,
        tcx.item_path_str(tcx.hir().local_def_id(it.id))
    );
    let _indenter = indenter();
    match it.node {
        // Consts can play a role in type-checking, so they are included here.
        hir::ItemKind::Static(..) => {
            let def_id = tcx.hir().local_def_id(it.id);
            tcx.typeck_tables_of(def_id);
            maybe_check_static_with_link_section(tcx, def_id, it.span);
        }
        hir::ItemKind::Const(..) => {
            tcx.typeck_tables_of(tcx.hir().local_def_id(it.id));
        }
        hir::ItemKind::Enum(ref enum_definition, _) => {
            check_enum(tcx, it.span, &enum_definition.variants, it.id);
        }
        hir::ItemKind::Fn(..) => {} // entirely within check_item_body
        hir::ItemKind::Impl(.., ref impl_item_refs) => {
            debug!("ItemKind::Impl {} with id {}", it.name, it.id);
            let impl_def_id = tcx.hir().local_def_id(it.id);
            if let Some(impl_trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                check_impl_items_against_trait(
                    tcx,
                    it.span,
                    impl_def_id,
                    impl_trait_ref,
                    impl_item_refs,
                );
                let trait_def_id = impl_trait_ref.def_id;
                check_on_unimplemented(tcx, trait_def_id, it);
            }
        }
        hir::ItemKind::Trait(..) => {
            let def_id = tcx.hir().local_def_id(it.id);
            check_on_unimplemented(tcx, def_id, it);
        }
        hir::ItemKind::Struct(..) => {
            check_struct(tcx, it.id, it.span);
        }
        hir::ItemKind::Union(..) => {
            check_union(tcx, it.id, it.span);
        }
        hir::ItemKind::Existential(..) | hir::ItemKind::Ty(..) => {
            let def_id = tcx.hir().local_def_id(it.id);
            let pty_ty = tcx.type_of(def_id);
            let generics = tcx.generics_of(def_id);
            check_bounds_are_used(tcx, &generics, pty_ty);
        }
        hir::ItemKind::ForeignMod(ref m) => {
            check_abi(tcx, it.span, m.abi);

            if m.abi == Abi::RustIntrinsic {
                for item in &m.items {
                    intrinsic::check_intrinsic_type(tcx, item);
                }
            } else if m.abi == Abi::PlatformIntrinsic {
                for item in &m.items {
                    intrinsic::check_platform_intrinsic_type(tcx, item);
                }
            } else {
                for item in &m.items {
                    let generics = tcx.generics_of(tcx.hir().local_def_id(item.id));
                    if generics.params.len() - generics.own_counts().lifetimes != 0 {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            item.span,
                            E0044,
                            "foreign items may not have type parameters"
                        );
                        err.span_label(item.span, "can't have type parameters");
                        // FIXME: once we start storing spans for type arguments, turn this into a
                        // suggestion.
                        err.help(
                            "use specialization instead of type parameters by replacing them \
                             with concrete types like `u32`",
                        );
                        err.emit();
                    }

                    if let hir::ForeignItemKind::Fn(ref fn_decl, _, _) = item.node {
                        require_c_abi_if_variadic(tcx, fn_decl, m.abi, item.span);
                    }
                }
            }
        }
        _ => { /* nothing to do */ }
    }
}

fn maybe_check_static_with_link_section(tcx: TyCtxt, id: DefId, span: Span) {
    // Only restricted on wasm32 target for now
    if !tcx.sess.opts.target_triple.triple().starts_with("wasm32") {
        return
    }

    // If `#[link_section]` is missing, then nothing to verify
    let attrs = tcx.codegen_fn_attrs(id);
    if attrs.link_section.is_none() {
        return
    }

    // For the wasm32 target statics with #[link_section] are placed into custom
    // sections of the final output file, but this isn't link custom sections of
    // other executable formats. Namely we can only embed a list of bytes,
    // nothing with pointers to anything else or relocations. If any relocation
    // show up, reject them here.
    let instance = ty::Instance::mono(tcx, id);
    let cid = GlobalId {
        instance,
        promoted: None
    };
    let param_env = ty::ParamEnv::reveal_all();
    if let Ok(static_) = tcx.const_eval(param_env.and(cid)) {
        let alloc = if let ConstValue::ByRef(_, allocation, _) = static_.val {
            allocation
        } else {
            bug!("Matching on non-ByRef static")
        };
        if alloc.relocations.len() != 0 {
            let msg = "statics with a custom `#[link_section]` must be a \
                       simple list of bytes on the wasm target with no \
                       extra levels of indirection such as references";
            tcx.sess.span_err(span, msg);
        }
    }
}

fn check_on_unimplemented<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    trait_def_id: DefId,
                                    item: &hir::Item) {
    let item_def_id = tcx.hir().local_def_id(item.id);
    // an error would be reported if this fails.
    let _ = traits::OnUnimplementedDirective::of_item(tcx, trait_def_id, item_def_id);
}

fn report_forbidden_specialization<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                             impl_item: &hir::ImplItem,
                                             parent_impl: DefId)
{
    let mut err = struct_span_err!(
        tcx.sess, impl_item.span, E0520,
        "`{}` specializes an item from a parent `impl`, but \
         that item is not marked `default`",
        impl_item.ident);
    err.span_label(impl_item.span, format!("cannot specialize default item `{}`",
                                            impl_item.ident));

    match tcx.span_of_impl(parent_impl) {
        Ok(span) => {
            err.span_label(span, "parent `impl` is here");
            err.note(&format!("to specialize, `{}` in the parent `impl` must be marked `default`",
                              impl_item.ident));
        }
        Err(cname) => {
            err.note(&format!("parent implementation is in crate `{}`", cname));
        }
    }

    err.emit();
}

fn check_specialization_validity<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                           trait_def: &ty::TraitDef,
                                           trait_item: &ty::AssociatedItem,
                                           impl_id: DefId,
                                           impl_item: &hir::ImplItem)
{
    let ancestors = trait_def.ancestors(tcx, impl_id);

    let kind = match impl_item.node {
        hir::ImplItemKind::Const(..) => ty::AssociatedKind::Const,
        hir::ImplItemKind::Method(..) => ty::AssociatedKind::Method,
        hir::ImplItemKind::Existential(..) => ty::AssociatedKind::Existential,
        hir::ImplItemKind::Type(_) => ty::AssociatedKind::Type
    };

    let parent = ancestors.defs(tcx, trait_item.ident, kind, trait_def.def_id).nth(1)
        .map(|node_item| node_item.map(|parent| parent.defaultness));

    if let Some(parent) = parent {
        if tcx.impl_item_is_final(&parent) {
            report_forbidden_specialization(tcx, impl_item, parent.node.def_id());
        }
    }

}

fn check_impl_items_against_trait<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                            impl_span: Span,
                                            impl_id: DefId,
                                            impl_trait_ref: ty::TraitRef<'tcx>,
                                            impl_item_refs: &[hir::ImplItemRef]) {
    let impl_span = tcx.sess.source_map().def_span(impl_span);

    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() { return; }

    // Locate trait definition and items
    let trait_def = tcx.trait_def(impl_trait_ref.def_id);
    let mut overridden_associated_type = None;

    let impl_items = || impl_item_refs.iter().map(|iiref| tcx.hir().impl_item(iiref.id));

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items() {
        let ty_impl_item = tcx.associated_item(tcx.hir().local_def_id(impl_item.id));
        let ty_trait_item = tcx.associated_items(impl_trait_ref.def_id)
            .find(|ac| Namespace::from(&impl_item.node) == Namespace::from(ac.kind) &&
                       tcx.hygienic_eq(ty_impl_item.ident, ac.ident, impl_trait_ref.def_id))
            .or_else(|| {
                // Not compatible, but needed for the error message
                tcx.associated_items(impl_trait_ref.def_id)
                   .find(|ac| tcx.hygienic_eq(ty_impl_item.ident, ac.ident, impl_trait_ref.def_id))
            });

        // Check that impl definition matches trait definition
        if let Some(ty_trait_item) = ty_trait_item {
            match impl_item.node {
                hir::ImplItemKind::Const(..) => {
                    // Find associated const definition.
                    if ty_trait_item.kind == ty::AssociatedKind::Const {
                        compare_const_impl(tcx,
                                           &ty_impl_item,
                                           impl_item.span,
                                           &ty_trait_item,
                                           impl_trait_ref);
                    } else {
                         let mut err = struct_span_err!(tcx.sess, impl_item.span, E0323,
                             "item `{}` is an associated const, \
                              which doesn't match its trait `{}`",
                             ty_impl_item.ident,
                             impl_trait_ref);
                         err.span_label(impl_item.span, "does not match trait");
                         // We can only get the spans from local trait definition
                         // Same for E0324 and E0325
                         if let Some(trait_span) = tcx.hir().span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, "item in trait");
                         }
                         err.emit()
                    }
                }
                hir::ImplItemKind::Method(..) => {
                    let trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssociatedKind::Method {
                        compare_impl_method(tcx,
                                            &ty_impl_item,
                                            impl_item.span,
                                            &ty_trait_item,
                                            impl_trait_ref,
                                            trait_span);
                    } else {
                        let mut err = struct_span_err!(tcx.sess, impl_item.span, E0324,
                            "item `{}` is an associated method, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref);
                         err.span_label(impl_item.span, "does not match trait");
                         if let Some(trait_span) = tcx.hir().span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, "item in trait");
                         }
                         err.emit()
                    }
                }
                hir::ImplItemKind::Existential(..) |
                hir::ImplItemKind::Type(_) => {
                    if ty_trait_item.kind == ty::AssociatedKind::Type {
                        if ty_trait_item.defaultness.has_value() {
                            overridden_associated_type = Some(impl_item);
                        }
                    } else {
                        let mut err = struct_span_err!(tcx.sess, impl_item.span, E0325,
                            "item `{}` is an associated type, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref);
                         err.span_label(impl_item.span, "does not match trait");
                         if let Some(trait_span) = tcx.hir().span_if_local(ty_trait_item.def_id) {
                            err.span_label(trait_span, "item in trait");
                         }
                         err.emit()
                    }
                }
            }

            check_specialization_validity(tcx, trait_def, &ty_trait_item, impl_id, impl_item);
        }
    }

    // Check for missing items from trait
    let mut missing_items = Vec::new();
    let mut invalidated_items = Vec::new();
    let associated_type_overridden = overridden_associated_type.is_some();
    for trait_item in tcx.associated_items(impl_trait_ref.def_id) {
        let is_implemented = trait_def.ancestors(tcx, impl_id)
            .defs(tcx, trait_item.ident, trait_item.kind, impl_trait_ref.def_id)
            .next()
            .map(|node_item| !node_item.node.is_from_trait())
            .unwrap_or(false);

        if !is_implemented && !tcx.impl_is_default(impl_id) {
            if !trait_item.defaultness.has_value() {
                missing_items.push(trait_item);
            } else if associated_type_overridden {
                invalidated_items.push(trait_item.ident);
            }
        }
    }

    if !missing_items.is_empty() {
        let mut err = struct_span_err!(tcx.sess, impl_span, E0046,
            "not all trait items implemented, missing: `{}`",
            missing_items.iter()
                .map(|trait_item| trait_item.ident.to_string())
                .collect::<Vec<_>>().join("`, `"));
        err.span_label(impl_span, format!("missing `{}` in implementation",
                missing_items.iter()
                    .map(|trait_item| trait_item.ident.to_string())
                    .collect::<Vec<_>>().join("`, `")));
        for trait_item in missing_items {
            if let Some(span) = tcx.hir().span_if_local(trait_item.def_id) {
                err.span_label(span, format!("`{}` from trait", trait_item.ident));
            } else {
                err.note_trait_signature(trait_item.ident.to_string(),
                                         trait_item.signature(&tcx));
            }
        }
        err.emit();
    }

    if !invalidated_items.is_empty() {
        let invalidator = overridden_associated_type.unwrap();
        span_err!(tcx.sess, invalidator.span, E0399,
                  "the following trait items need to be reimplemented \
                   as `{}` was overridden: `{}`",
                  invalidator.ident,
                  invalidated_items.iter()
                                   .map(|name| name.to_string())
                                   .collect::<Vec<_>>().join("`, `"))
    }
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
fn check_representable<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 sp: Span,
                                 item_def_id: DefId)
                                 -> bool {
    let rty = tcx.type_of(item_def_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match rty.is_representable(tcx, sp) {
        Representability::SelfRecursive(spans) => {
            let mut err = tcx.recursive_type_with_infinite_size_error(item_def_id);
            for span in spans {
                err.span_label(span, "recursive without indirection");
            }
            err.emit();
            return false
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    return true
}

pub fn check_simd<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span, def_id: DefId) {
    let t = tcx.type_of(def_id);
    if let ty::Adt(def, substs) = t.sty {
        if def.is_struct() {
            let fields = &def.non_enum_variant().fields;
            if fields.is_empty() {
                span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty");
                return;
            }
            let e = fields[0].ty(tcx, substs);
            if !fields.iter().all(|f| f.ty(tcx, substs) == e) {
                struct_span_err!(tcx.sess, sp, E0076, "SIMD vector should be homogeneous")
                                .span_label(sp, "SIMD elements must have the same type")
                                .emit();
                return;
            }
            match e.sty {
                ty::Param(_) => { /* struct<T>(T, T, T, T) is ok */ }
                _ if e.is_machine() => { /* struct(u8, u8, u8, u8) is ok */ }
                _ => {
                    span_err!(tcx.sess, sp, E0077,
                              "SIMD vector element type should be machine type");
                    return;
                }
            }
        }
    }
}

fn check_packed<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span, def_id: DefId) {
    let repr = tcx.adt_def(def_id).repr;
    if repr.packed() {
        for attr in tcx.get_attrs(def_id).iter() {
            for r in attr::find_repr_attrs(&tcx.sess.parse_sess, attr) {
                if let attr::ReprPacked(pack) = r {
                    if pack != repr.pack {
                        struct_span_err!(tcx.sess, sp, E0634,
                                         "type has conflicting packed representation hints").emit();
                    }
                }
            }
        }
        if repr.align > 0 {
            struct_span_err!(tcx.sess, sp, E0587,
                             "type has conflicting packed and align representation hints").emit();
        }
        else if check_packed_inner(tcx, def_id, &mut Vec::new()) {
            struct_span_err!(tcx.sess, sp, E0588,
                "packed type cannot transitively contain a `[repr(align)]` type").emit();
        }
    }
}

fn check_packed_inner<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                def_id: DefId,
                                stack: &mut Vec<DefId>) -> bool {
    let t = tcx.type_of(def_id);
    if stack.contains(&def_id) {
        debug!("check_packed_inner: {:?} is recursive", t);
        return false;
    }
    if let ty::Adt(def, substs) = t.sty {
        if def.is_struct() || def.is_union() {
            if tcx.adt_def(def.did).repr.align > 0 {
                return true;
            }
            // push struct def_id before checking fields
            stack.push(def_id);
            for field in &def.non_enum_variant().fields {
                let f = field.ty(tcx, substs);
                if let ty::Adt(def, _) = f.sty {
                    if check_packed_inner(tcx, def.did, stack) {
                        return true;
                    }
                }
            }
            // only need to pop if not early out
            stack.pop();
        }
    }
    false
}

fn check_transparent<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span, def_id: DefId) {
    let adt = tcx.adt_def(def_id);
    if !adt.repr.transparent() {
        return;
    }

    // For each field, figure out if it's known to be a ZST and align(1)
    let field_infos = adt.non_enum_variant().fields.iter().map(|field| {
        let ty = field.ty(tcx, Substs::identity_for_item(tcx, field.did));
        let param_env = tcx.param_env(field.did);
        let layout = tcx.layout_of(param_env.and(ty));
        // We are currently checking the type this field came from, so it must be local
        let span = tcx.hir().span_if_local(field.did).unwrap();
        let zst = layout.map(|layout| layout.is_zst()).unwrap_or(false);
        let align1 = layout.map(|layout| layout.align.abi.bytes() == 1).unwrap_or(false);
        (span, zst, align1)
    });

    let non_zst_fields = field_infos.clone().filter(|(_span, zst, _align1)| !*zst);
    let non_zst_count = non_zst_fields.clone().count();
    if non_zst_count != 1 {
        let field_spans: Vec<_> = non_zst_fields.map(|(span, _zst, _align1)| span).collect();
        struct_span_err!(tcx.sess, sp, E0690,
                         "transparent struct needs exactly one non-zero-sized field, but has {}",
                         non_zst_count)
        .span_note(field_spans, "non-zero-sized field")
        .emit();
    }
    for (span, zst, align1) in field_infos {
        if zst && !align1 {
            span_err!(tcx.sess, span, E0691,
                      "zero-sized field in transparent struct has alignment larger than 1");
        }
    }
}

#[allow(trivial_numeric_casts)]
pub fn check_enum<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            sp: Span,
                            vs: &'tcx [hir::Variant],
                            id: ast::NodeId) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated

    if vs.is_empty() {
        let attributes = tcx.get_attrs(def_id);
        if let Some(attr) = attr::find_by_name(&attributes, "repr") {
            struct_span_err!(
                tcx.sess, attr.span, E0084,
                "unsupported representation for zero-variant enum")
                .span_label(sp, "zero-variant enum")
                .emit();
        }
    }

    let repr_type_ty = def.repr.discr_type().to_ty(tcx);
    if repr_type_ty == tcx.types.i128 || repr_type_ty == tcx.types.u128 {
        if !tcx.features().repr128 {
            emit_feature_err(&tcx.sess.parse_sess,
                             "repr128",
                             sp,
                             GateIssue::Language,
                             "repr with 128-bit type is unstable");
        }
    }

    for v in vs {
        if let Some(ref e) = v.node.disr_expr {
            tcx.typeck_tables_of(tcx.hir().local_def_id(e.id));
        }
    }

    let mut disr_vals: Vec<Discr<'tcx>> = Vec::with_capacity(vs.len());
    for ((_, discr), v) in def.discriminants(tcx).zip(vs) {
        // Check for duplicate discriminant values
        if let Some(i) = disr_vals.iter().position(|&x| x.val == discr.val) {
            let variant_did = def.variants[VariantIdx::new(i)].did;
            let variant_i_node_id = tcx.hir().as_local_node_id(variant_did).unwrap();
            let variant_i = tcx.hir().expect_variant(variant_i_node_id);
            let i_span = match variant_i.node.disr_expr {
                Some(ref expr) => tcx.hir().span(expr.id),
                None => tcx.hir().span(variant_i_node_id)
            };
            let span = match v.node.disr_expr {
                Some(ref expr) => tcx.hir().span(expr.id),
                None => v.span
            };
            struct_span_err!(tcx.sess, span, E0081,
                             "discriminant value `{}` already exists", disr_vals[i])
                .span_label(i_span, format!("first use of `{}`", disr_vals[i]))
                .span_label(span , format!("enum already has `{}`", disr_vals[i]))
                .emit();
        }
        disr_vals.push(discr);
    }

    check_representable(tcx, sp, def_id);
}

impl<'a, 'gcx, 'tcx> AstConv<'gcx, 'tcx> for FnCtxt<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn get_type_parameter_bounds(&self, _: Span, def_id: DefId)
                                 -> Lrc<ty::GenericPredicates<'tcx>>
    {
        let tcx = self.tcx;
        let node_id = tcx.hir().as_local_node_id(def_id).unwrap();
        let item_id = tcx.hir().ty_param_owner(node_id);
        let item_def_id = tcx.hir().local_def_id(item_id);
        let generics = tcx.generics_of(item_def_id);
        let index = generics.param_def_id_to_index[&def_id];
        Lrc::new(ty::GenericPredicates {
            parent: None,
            predicates: self.param_env.caller_bounds.iter().filter_map(|&predicate| {
                match predicate {
                    ty::Predicate::Trait(ref data)
                    if data.skip_binder().self_ty().is_param(index) => {
                        // HACK(eddyb) should get the original `Span`.
                        let span = tcx.def_span(def_id);
                        Some((predicate, span))
                    }
                    _ => None
                }
            }).collect()
        })
    }

    fn re_infer(&self, span: Span, def: Option<&ty::GenericParamDef>)
                -> Option<ty::Region<'tcx>> {
        let v = match def {
            Some(def) => infer::EarlyBoundRegion(span, def.name),
            None => infer::MiscVariable(span)
        };
        Some(self.next_region_var(v))
    }

    fn ty_infer(&self, span: Span) -> Ty<'tcx> {
        self.next_ty_var(TypeVariableOrigin::TypeInference(span))
    }

    fn ty_infer_for_def(&self,
                        ty_param_def: &ty::GenericParamDef,
                        span: Span) -> Ty<'tcx> {
        if let UnpackedKind::Type(ty) = self.var_for_def(span, ty_param_def).unpack() {
            return ty;
        }
        unreachable!()
    }

    fn projected_ty_from_poly_trait_ref(&self,
                                        span: Span,
                                        item_def_id: DefId,
                                        poly_trait_ref: ty::PolyTraitRef<'tcx>)
                                        -> Ty<'tcx>
    {
        let (trait_ref, _) = self.replace_bound_vars_with_fresh_vars(
            span,
            infer::LateBoundRegionConversionTime::AssocTypeProjection(item_def_id),
            &poly_trait_ref
        );

        self.tcx().mk_projection(item_def_id, trait_ref.substs)
    }

    fn normalize_ty(&self, span: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.has_escaping_bound_vars() {
            ty // FIXME: normalization and escaping regions
        } else {
            self.normalize_associated_types_in(span, &ty)
        }
    }

    fn set_tainted_by_errors(&self) {
        self.infcx.set_tainted_by_errors()
    }

    fn record_ty(&self, hir_id: hir::HirId, ty: Ty<'tcx>, _span: Span) {
        self.write_ty(hir_id, ty)
    }
}

/// Controls whether the arguments are tupled. This is used for the call
/// operator.
///
/// Tupling means that all call-side arguments are packed into a tuple and
/// passed as a single parameter. For example, if tupling is enabled, this
/// function:
///
///     fn f(x: (isize, isize))
///
/// Can be called as:
///
///     f(1, 2);
///
/// Instead of:
///
///     f((1, 2));
#[derive(Clone, Eq, PartialEq)]
enum TupleArgumentsFlag {
    DontTupleArguments,
    TupleArguments,
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn new(inh: &'a Inherited<'a, 'gcx, 'tcx>,
               param_env: ty::ParamEnv<'tcx>,
               body_id: ast::NodeId)
               -> FnCtxt<'a, 'gcx, 'tcx> {
        FnCtxt {
            body_id,
            param_env,
            err_count_on_creation: inh.tcx.sess.err_count(),
            ret_coercion: None,
            yield_ty: None,
            ps: RefCell::new(UnsafetyState::function(hir::Unsafety::Normal,
                                                     ast::CRATE_NODE_ID)),
            diverges: Cell::new(Diverges::Maybe),
            has_errors: Cell::new(false),
            enclosing_breakables: RefCell::new(EnclosingBreakables {
                stack: Vec::new(),
                by_id: Default::default(),
            }),
            inh,
        }
    }

    pub fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    pub fn err_count_since_creation(&self) -> usize {
        self.tcx.sess.err_count() - self.err_count_on_creation
    }

    /// Produce warning on the given node, if the current point in the
    /// function is unreachable, and there hasn't been another warning.
    fn warn_if_unreachable(&self, id: ast::NodeId, span: Span, kind: &str) {
        if self.diverges.get() == Diverges::Always {
            self.diverges.set(Diverges::WarnedAlways);

            debug!("warn_if_unreachable: id={:?} span={:?} kind={}", id, span, kind);

            self.tcx().lint_node(
                lint::builtin::UNREACHABLE_CODE,
                id, span,
                &format!("unreachable {}", kind));
        }
    }

    pub fn cause(&self,
                 span: Span,
                 code: ObligationCauseCode<'tcx>)
                 -> ObligationCause<'tcx> {
        ObligationCause::new(span, self.body_id, code)
    }

    pub fn misc(&self, span: Span) -> ObligationCause<'tcx> {
        self.cause(span, ObligationCauseCode::MiscObligation)
    }

    /// Resolves type variables in `ty` if possible. Unlike the infcx
    /// version (resolve_type_vars_if_possible), this version will
    /// also select obligations if it seems useful, in an effort
    /// to get more type information.
    fn resolve_type_vars_with_obligations(&self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        debug!("resolve_type_vars_with_obligations(ty={:?})", ty);

        // No Infer()? Nothing needs doing.
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_with_obligations: ty={:?}", ty);
            return ty;
        }

        // If `ty` is a type variable, see whether we already know what it is.
        ty = self.resolve_type_vars_if_possible(&ty);
        if !ty.has_infer_types() {
            debug!("resolve_type_vars_with_obligations: ty={:?}", ty);
            return ty;
        }

        // If not, try resolving pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible(false);
        ty = self.resolve_type_vars_if_possible(&ty);

        debug!("resolve_type_vars_with_obligations: ty={:?}", ty);
        ty
    }

    fn record_deferred_call_resolution(&self,
                                       closure_def_id: DefId,
                                       r: DeferredCallResolution<'gcx, 'tcx>) {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.entry(closure_def_id).or_default().push(r);
    }

    fn remove_deferred_call_resolutions(&self,
                                        closure_def_id: DefId)
                                        -> Vec<DeferredCallResolution<'gcx, 'tcx>>
    {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.remove(&closure_def_id).unwrap_or(vec![])
    }

    pub fn tag(&self) -> String {
        let self_ptr: *const FnCtxt = self;
        format!("{:?}", self_ptr)
    }

    pub fn local_ty(&self, span: Span, nid: ast::NodeId) -> LocalTy<'tcx> {
        self.locals.borrow().get(&nid).cloned().unwrap_or_else(||
            span_bug!(span, "no type for local variable {}",
                      self.tcx.hir().node_to_string(nid))
        )
    }

    #[inline]
    pub fn write_ty(&self, id: hir::HirId, ty: Ty<'tcx>) {
        debug!("write_ty({:?}, {:?}) in fcx {}",
               id, self.resolve_type_vars_if_possible(&ty), self.tag());
        self.tables.borrow_mut().node_types_mut().insert(id, ty);

        if ty.references_error() {
            self.has_errors.set(true);
            self.set_tainted_by_errors();
        }
    }

    pub fn write_field_index(&self, node_id: ast::NodeId, index: usize) {
        let hir_id = self.tcx.hir().node_to_hir_id(node_id);
        self.tables.borrow_mut().field_indices_mut().insert(hir_id, index);
    }

    // The NodeId and the ItemLocalId must identify the same item. We just pass
    // both of them for consistency checking.
    pub fn write_method_call(&self,
                             hir_id: hir::HirId,
                             method: MethodCallee<'tcx>) {
        debug!("write_method_call(hir_id={:?}, method={:?})", hir_id, method);
        self.tables
            .borrow_mut()
            .type_dependent_defs_mut()
            .insert(hir_id, Def::Method(method.def_id));

        self.write_substs(hir_id, method.substs);

        // When the method is confirmed, the `method.substs` includes
        // parameters from not just the method, but also the impl of
        // the method -- in particular, the `Self` type will be fully
        // resolved. However, those are not something that the "user
        // specified" -- i.e., those types come from the inferred type
        // of the receiver, not something the user wrote. So when we
        // create the user-substs, we want to replace those earlier
        // types with just the types that the user actually wrote --
        // that is, those that appear on the *method itself*.
        //
        // As an example, if the user wrote something like
        // `foo.bar::<u32>(...)` -- the `Self` type here will be the
        // type of `foo` (possibly adjusted), but we don't want to
        // include that. We want just the `[_, u32]` part.
        if !method.substs.is_noop() {
            let method_generics = self.tcx.generics_of(method.def_id);
            if !method_generics.params.is_empty() {
                let user_substs = self.infcx.probe(|_| {
                    let just_method_substs = Substs::for_item(self.tcx, method.def_id, |param, _| {
                        let i = param.index as usize;
                        if i < method_generics.parent_count {
                            self.infcx.var_for_def(DUMMY_SP, param)
                        } else {
                            method.substs[i]
                        }
                    });
                    self.infcx.canonicalize_user_type_annotation(&UserSubsts {
                        substs: just_method_substs,
                        user_self_ty: None, // not relevant here
                    })
                });

                debug!("write_method_call: user_substs = {:?}", user_substs);
                self.write_user_substs(hir_id, user_substs);
            }
        }
    }

    pub fn write_substs(&self, node_id: hir::HirId, substs: &'tcx Substs<'tcx>) {
        if !substs.is_noop() {
            debug!("write_substs({:?}, {:?}) in fcx {}",
                   node_id,
                   substs,
                   self.tag());

            self.tables.borrow_mut().node_substs_mut().insert(node_id, substs);
        }
    }

    /// Given the substs that we just converted from the HIR, try to
    /// canonicalize them and store them as user-given substitutions
    /// (i.e., substitutions that must be respected by the NLL check).
    ///
    /// This should be invoked **before any unifications have
    /// occurred**, so that annotations like `Vec<_>` are preserved
    /// properly.
    pub fn write_user_substs_from_substs(
        &self,
        hir_id: hir::HirId,
        substs: &'tcx Substs<'tcx>,
        user_self_ty: Option<UserSelfTy<'tcx>>,
    ) {
        debug!(
            "write_user_substs_from_substs({:?}, {:?}) in fcx {}",
            hir_id,
            substs,
            self.tag(),
        );

        if !substs.is_noop() {
            let user_substs = self.infcx.canonicalize_user_type_annotation(&UserSubsts {
                substs,
                user_self_ty,
            });
            debug!("instantiate_value_path: user_substs = {:?}", user_substs);
            self.write_user_substs(hir_id, user_substs);
        }
    }

    pub fn write_user_substs(&self, hir_id: hir::HirId, substs: CanonicalUserSubsts<'tcx>) {
        debug!(
            "write_user_substs({:?}, {:?}) in fcx {}",
            hir_id,
            substs,
            self.tag(),
        );

        if !substs.is_identity() {
            self.tables.borrow_mut().user_substs_mut().insert(hir_id, substs);
        } else {
            debug!("write_user_substs: skipping identity substs");
        }
    }

    pub fn apply_adjustments(&self, expr: &hir::Expr, adj: Vec<Adjustment<'tcx>>) {
        debug!("apply_adjustments(expr={:?}, adj={:?})", expr, adj);

        if adj.is_empty() {
            return;
        }

        match self.tables.borrow_mut().adjustments_mut().entry(expr.hir_id) {
            Entry::Vacant(entry) => { entry.insert(adj); },
            Entry::Occupied(mut entry) => {
                debug!(" - composing on top of {:?}", entry.get());
                match (&entry.get()[..], &adj[..]) {
                    // Applying any adjustment on top of a NeverToAny
                    // is a valid NeverToAny adjustment, because it can't
                    // be reached.
                    (&[Adjustment { kind: Adjust::NeverToAny, .. }], _) => return,
                    (&[
                        Adjustment { kind: Adjust::Deref(_), .. },
                        Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), .. },
                    ], &[
                        Adjustment { kind: Adjust::Deref(_), .. },
                        .. // Any following adjustments are allowed.
                    ]) => {
                        // A reborrow has no effect before a dereference.
                    }
                    // FIXME: currently we never try to compose autoderefs
                    // and ReifyFnPointer/UnsafeFnPointer, but we could.
                    _ =>
                        bug!("while adjusting {:?}, can't compose {:?} and {:?}",
                             expr, entry.get(), adj)
                };
                *entry.get_mut() = adj;
            }
        }
    }

    /// Basically whenever we are converting from a type scheme into
    /// the fn body space, we always want to normalize associated
    /// types as well. This function combines the two.
    fn instantiate_type_scheme<T>(&self,
                                  span: Span,
                                  substs: &Substs<'tcx>,
                                  value: &T)
                                  -> T
        where T : TypeFoldable<'tcx>
    {
        let value = value.subst(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, &value);
        debug!("instantiate_type_scheme(value={:?}, substs={:?}) = {:?}",
               value,
               substs,
               result);
        result
    }

    /// As `instantiate_type_scheme`, but for the bounds found in a
    /// generic type scheme.
    fn instantiate_bounds(&self, span: Span, def_id: DefId, substs: &Substs<'tcx>)
                          -> ty::InstantiatedPredicates<'tcx> {
        let bounds = self.tcx.predicates_of(def_id);
        let result = bounds.instantiate(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, &result);
        debug!("instantiate_bounds(bounds={:?}, substs={:?}) = {:?}",
               bounds,
               substs,
               result);
        result
    }

    /// Replace the opaque types from the given value with type variables,
    /// and records the `OpaqueTypeMap` for later use during writeback. See
    /// `InferCtxt::instantiate_opaque_types` for more details.
    fn instantiate_opaque_types_from_value<T: TypeFoldable<'tcx>>(
        &self,
        parent_id: ast::NodeId,
        value: &T,
    ) -> T {
        let parent_def_id = self.tcx.hir().local_def_id(parent_id);
        debug!("instantiate_opaque_types_from_value(parent_def_id={:?}, value={:?})",
               parent_def_id,
               value);

        let (value, opaque_type_map) = self.register_infer_ok_obligations(
            self.instantiate_opaque_types(
                parent_def_id,
                self.body_id,
                self.param_env,
                value,
            )
        );

        let mut opaque_types = self.opaque_types.borrow_mut();
        for (ty, decl) in opaque_type_map {
            let old_value = opaque_types.insert(ty, decl);
            assert!(old_value.is_none(), "instantiated twice: {:?}/{:?}", ty, decl);
        }

        value
    }

    fn normalize_associated_types_in<T>(&self, span: Span, value: &T) -> T
        where T : TypeFoldable<'tcx>
    {
        self.inh.normalize_associated_types_in(span, self.body_id, self.param_env, value)
    }

    fn normalize_associated_types_in_as_infer_ok<T>(&self, span: Span, value: &T)
                                                    -> InferOk<'tcx, T>
        where T : TypeFoldable<'tcx>
    {
        self.inh.partially_normalize_associated_types_in(span,
                                                         self.body_id,
                                                         self.param_env,
                                                         value)
    }

    pub fn require_type_meets(&self,
                              ty: Ty<'tcx>,
                              span: Span,
                              code: traits::ObligationCauseCode<'tcx>,
                              def_id: DefId)
    {
        self.register_bound(
            ty,
            def_id,
            traits::ObligationCause::new(span, self.body_id, code));
    }

    pub fn require_type_is_sized(&self,
                                 ty: Ty<'tcx>,
                                 span: Span,
                                 code: traits::ObligationCauseCode<'tcx>)
    {
        let lang_item = self.tcx.require_lang_item(lang_items::SizedTraitLangItem);
        self.require_type_meets(ty, span, code, lang_item);
    }

    pub fn require_type_is_sized_deferred(&self,
                                          ty: Ty<'tcx>,
                                          span: Span,
                                          code: traits::ObligationCauseCode<'tcx>)
    {
        self.deferred_sized_obligations.borrow_mut().push((ty, span, code));
    }

    pub fn register_bound(&self,
                          ty: Ty<'tcx>,
                          def_id: DefId,
                          cause: traits::ObligationCause<'tcx>)
    {
        self.fulfillment_cx.borrow_mut()
                           .register_bound(self, self.param_env, ty, def_id, cause);
    }

    pub fn to_ty(&self, ast_t: &hir::Ty) -> Ty<'tcx> {
        let t = AstConv::ast_ty_to_ty(self, ast_t);
        self.register_wf_obligation(t, ast_t.span, traits::MiscObligation);
        t
    }

    pub fn to_ty_saving_user_provided_ty(&self, ast_ty: &hir::Ty) -> Ty<'tcx> {
        let ty = self.to_ty(ast_ty);

        // If the type given by the user has free regions, save it for
        // later, since NLL would like to enforce those. Also pass in
        // types that involve projections, since those can resolve to
        // `'static` bounds (modulo #54940, which hopefully will be
        // fixed by the time you see this comment, dear reader,
        // although I have my doubts). Other sorts of things are
        // already sufficiently enforced with erased regions. =)
        if ty.has_free_regions() || ty.has_projections() {
            let c_ty = self.infcx.canonicalize_response(&ty);
            self.tables.borrow_mut().user_provided_tys_mut().insert(ast_ty.hir_id, c_ty);
        }

        ty
    }

    pub fn node_ty(&self, id: hir::HirId) -> Ty<'tcx> {
        match self.tables.borrow().node_types().get(id) {
            Some(&t) => t,
            None if self.is_tainted_by_errors() => self.tcx.types.err,
            None => {
                let node_id = self.tcx.hir().hir_to_node_id(id);
                bug!("no type for node {}: {} in fcx {}",
                     node_id, self.tcx.hir().node_to_string(node_id),
                     self.tag());
            }
        }
    }

    /// Registers an obligation for checking later, during regionck, that the type `ty` must
    /// outlive the region `r`.
    pub fn register_wf_obligation(&self,
                                  ty: Ty<'tcx>,
                                  span: Span,
                                  code: traits::ObligationCauseCode<'tcx>)
    {
        // WF obligations never themselves fail, so no real need to give a detailed cause:
        let cause = traits::ObligationCause::new(span, self.body_id, code);
        self.register_predicate(traits::Obligation::new(cause,
                                                        self.param_env,
                                                        ty::Predicate::WellFormed(ty)));
    }

    /// Registers obligations that all types appearing in `substs` are well-formed.
    pub fn add_wf_bounds(&self, substs: &Substs<'tcx>, expr: &hir::Expr) {
        for ty in substs.types() {
            self.register_wf_obligation(ty, expr.span, traits::MiscObligation);
        }
    }

    /// Given a fully substituted set of bounds (`generic_bounds`), and the values with which each
    /// type/region parameter was instantiated (`substs`), creates and registers suitable
    /// trait/region obligations.
    ///
    /// For example, if there is a function:
    ///
    /// ```
    /// fn foo<'a,T:'a>(...)
    /// ```
    ///
    /// and a reference:
    ///
    /// ```
    /// let f = foo;
    /// ```
    ///
    /// Then we will create a fresh region variable `'$0` and a fresh type variable `$1` for `'a`
    /// and `T`. This routine will add a region obligation `$1:'$0` and register it locally.
    pub fn add_obligations_for_parameters(&self,
                                          cause: traits::ObligationCause<'tcx>,
                                          predicates: &ty::InstantiatedPredicates<'tcx>)
    {
        assert!(!predicates.has_escaping_bound_vars());

        debug!("add_obligations_for_parameters(predicates={:?})",
               predicates);

        for obligation in traits::predicates_for_generics(cause, self.param_env, predicates) {
            self.register_predicate(obligation);
        }
    }

    // FIXME(arielb1): use this instead of field.ty everywhere
    // Only for fields! Returns <none> for methods>
    // Indifferent to privacy flags
    pub fn field_ty(&self,
                    span: Span,
                    field: &'tcx ty::FieldDef,
                    substs: &Substs<'tcx>)
                    -> Ty<'tcx>
    {
        self.normalize_associated_types_in(span, &field.ty(self.tcx, substs))
    }

    fn check_casts(&self) {
        let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
        for cast in deferred_cast_checks.drain(..) {
            cast.check(self);
        }
    }

    fn resolve_generator_interiors(&self, def_id: DefId) {
        let mut generators = self.deferred_generator_interiors.borrow_mut();
        for (body_id, interior) in generators.drain(..) {
            self.select_obligations_where_possible(false);
            generator_interior::resolve_interior(self, def_id, body_id, interior);
        }
    }

    // Tries to apply a fallback to `ty` if it is an unsolved variable.
    // Non-numerics get replaced with ! or () (depending on whether
    // feature(never_type) is enabled, unconstrained ints with i32,
    // unconstrained floats with f64.
    // Fallback becomes very dubious if we have encountered type-checking errors.
    // In that case, fallback to Error.
    // The return value indicates whether fallback has occurred.
    fn fallback_if_possible(&self, ty: Ty<'tcx>) -> bool {
        use rustc::ty::error::UnconstrainedNumeric::Neither;
        use rustc::ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};

        assert!(ty.is_ty_infer());
        let fallback = match self.type_is_unconstrained_numeric(ty) {
            _ if self.is_tainted_by_errors() => self.tcx().types.err,
            UnconstrainedInt => self.tcx.types.i32,
            UnconstrainedFloat => self.tcx.types.f64,
            Neither if self.type_var_diverges(ty) => self.tcx.mk_diverging_default(),
            Neither => return false,
        };
        debug!("default_type_parameters: defaulting `{:?}` to `{:?}`", ty, fallback);
        self.demand_eqtype(syntax_pos::DUMMY_SP, ty, fallback);
        true
    }

    fn select_all_obligations_or_error(&self) {
        debug!("select_all_obligations_or_error");
        if let Err(errors) = self.fulfillment_cx.borrow_mut().select_all_or_error(&self) {
            self.report_fulfillment_errors(&errors, self.inh.body_id, false);
        }
    }

    /// Select as many obligations as we can at present.
    fn select_obligations_where_possible(&self, fallback_has_occurred: bool) {
        if let Err(errors) = self.fulfillment_cx.borrow_mut().select_where_possible(self) {
            self.report_fulfillment_errors(&errors, self.inh.body_id, fallback_has_occurred);
        }
    }

    /// For the overloaded place expressions (`*x`, `x[3]`), the trait
    /// returns a type of `&T`, but the actual type we assign to the
    /// *expression* is `T`. So this function just peels off the return
    /// type by one layer to yield `T`.
    fn make_overloaded_place_return_type(&self,
                                          method: MethodCallee<'tcx>)
                                          -> ty::TypeAndMut<'tcx>
    {
        // extract method return type, which will be &T;
        let ret_ty = method.sig.output();

        // method returns &T, but the type as visible to user is T, so deref
        ret_ty.builtin_deref(true).unwrap()
    }

    fn lookup_indexing(&self,
                       expr: &hir::Expr,
                       base_expr: &'gcx hir::Expr,
                       base_ty: Ty<'tcx>,
                       idx_ty: Ty<'tcx>,
                       needs: Needs)
                       -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)>
    {
        // FIXME(#18741) -- this is almost but not quite the same as the
        // autoderef that normal method probing does. They could likely be
        // consolidated.

        let mut autoderef = self.autoderef(base_expr.span, base_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_index_step(expr, base_expr, &autoderef, needs, idx_ty);
        }
        autoderef.finalize();
        result
    }

    /// To type-check `base_expr[index_expr]`, we progressively autoderef
    /// (and otherwise adjust) `base_expr`, looking for a type which either
    /// supports builtin indexing or overloaded indexing.
    /// This loop implements one step in that search; the autoderef loop
    /// is implemented by `lookup_indexing`.
    fn try_index_step(&self,
                      expr: &hir::Expr,
                      base_expr: &hir::Expr,
                      autoderef: &Autoderef<'a, 'gcx, 'tcx>,
                      needs: Needs,
                      index_ty: Ty<'tcx>)
                      -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)>
    {
        let adjusted_ty = autoderef.unambiguous_final_ty();
        debug!("try_index_step(expr={:?}, base_expr={:?}, adjusted_ty={:?}, \
                               index_ty={:?})",
               expr,
               base_expr,
               adjusted_ty,
               index_ty);

        for &unsize in &[false, true] {
            let mut self_ty = adjusted_ty;
            if unsize {
                // We only unsize arrays here.
                if let ty::Array(element_ty, _) = adjusted_ty.sty {
                    self_ty = self.tcx.mk_slice(element_ty);
                } else {
                    continue;
                }
            }

            // If some lookup succeeds, write callee into table and extract index/element
            // type from the method signature.
            // If some lookup succeeded, install method in table
            let input_ty = self.next_ty_var(TypeVariableOrigin::AutoDeref(base_expr.span));
            let method = self.try_overloaded_place_op(
                expr.span, self_ty, &[input_ty], needs, PlaceOp::Index);

            let result = method.map(|ok| {
                debug!("try_index_step: success, using overloaded indexing");
                let method = self.register_infer_ok_obligations(ok);

                let mut adjustments = autoderef.adjust_steps(needs);
                if let ty::Ref(region, _, r_mutbl) = method.sig.inputs()[0].sty {
                    let mutbl = match r_mutbl {
                        hir::MutImmutable => AutoBorrowMutability::Immutable,
                        hir::MutMutable => AutoBorrowMutability::Mutable {
                            // Indexing can be desugared to a method call,
                            // so maybe we could use two-phase here.
                            // See the documentation of AllowTwoPhase for why that's
                            // not the case today.
                            allow_two_phase_borrow: AllowTwoPhase::No,
                        }
                    };
                    adjustments.push(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                        target: self.tcx.mk_ref(region, ty::TypeAndMut {
                            mutbl: r_mutbl,
                            ty: adjusted_ty
                        })
                    });
                }
                if unsize {
                    adjustments.push(Adjustment {
                        kind: Adjust::Unsize,
                        target: method.sig.inputs()[0]
                    });
                }
                self.apply_adjustments(base_expr, adjustments);

                self.write_method_call(expr.hir_id, method);
                (input_ty, self.make_overloaded_place_return_type(method).ty)
            });
            if result.is_some() {
                return result;
            }
        }

        None
    }

    fn resolve_place_op(&self, op: PlaceOp, is_mut: bool) -> (Option<DefId>, ast::Ident) {
        let (tr, name) = match (op, is_mut) {
            (PlaceOp::Deref, false) =>
                (self.tcx.lang_items().deref_trait(), "deref"),
            (PlaceOp::Deref, true) =>
                (self.tcx.lang_items().deref_mut_trait(), "deref_mut"),
            (PlaceOp::Index, false) =>
                (self.tcx.lang_items().index_trait(), "index"),
            (PlaceOp::Index, true) =>
                (self.tcx.lang_items().index_mut_trait(), "index_mut"),
        };
        (tr, ast::Ident::from_str(name))
    }

    fn try_overloaded_place_op(&self,
                                span: Span,
                                base_ty: Ty<'tcx>,
                                arg_tys: &[Ty<'tcx>],
                                needs: Needs,
                                op: PlaceOp)
                                -> Option<InferOk<'tcx, MethodCallee<'tcx>>>
    {
        debug!("try_overloaded_place_op({:?},{:?},{:?},{:?})",
               span,
               base_ty,
               needs,
               op);

        // Try Mut first, if needed.
        let (mut_tr, mut_op) = self.resolve_place_op(op, true);
        let method = match (needs, mut_tr) {
            (Needs::MutPlace, Some(trait_did)) => {
                self.lookup_method_in_trait(span, mut_op, trait_did, base_ty, Some(arg_tys))
            }
            _ => None,
        };

        // Otherwise, fall back to the immutable version.
        let (imm_tr, imm_op) = self.resolve_place_op(op, false);
        let method = match (method, imm_tr) {
            (None, Some(trait_did)) => {
                self.lookup_method_in_trait(span, imm_op, trait_did, base_ty, Some(arg_tys))
            }
            (method, _) => method,
        };

        method
    }

    fn check_method_argument_types(&self,
                                   sp: Span,
                                   expr_sp: Span,
                                   method: Result<MethodCallee<'tcx>, ()>,
                                   args_no_rcvr: &'gcx [hir::Expr],
                                   tuple_arguments: TupleArgumentsFlag,
                                   expected: Expectation<'tcx>)
                                   -> Ty<'tcx> {
        let has_error = match method {
            Ok(method) => {
                method.substs.references_error() || method.sig.references_error()
            }
            Err(_) => true
        };
        if has_error {
            let err_inputs = self.err_args(args_no_rcvr.len());

            let err_inputs = match tuple_arguments {
                DontTupleArguments => err_inputs,
                TupleArguments => vec![self.tcx.intern_tup(&err_inputs[..])],
            };

            self.check_argument_types(sp, expr_sp, &err_inputs[..], &[], args_no_rcvr,
                                      false, tuple_arguments, None);
            return self.tcx.types.err;
        }

        let method = method.unwrap();
        // HACK(eddyb) ignore self in the definition (see above).
        let expected_arg_tys = self.expected_inputs_for_expected_output(
            sp,
            expected,
            method.sig.output(),
            &method.sig.inputs()[1..]
        );
        self.check_argument_types(sp, expr_sp, &method.sig.inputs()[1..], &expected_arg_tys[..],
                                  args_no_rcvr, method.sig.variadic, tuple_arguments,
                                  self.tcx.hir().span_if_local(method.def_id));
        method.sig.output()
    }

    fn self_type_matches_expected_vid(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        expected_vid: ty::TyVid,
    ) -> bool {
        let self_ty = self.shallow_resolve(trait_ref.self_ty());
        debug!(
            "self_type_matches_expected_vid(trait_ref={:?}, self_ty={:?}, expected_vid={:?})",
            trait_ref, self_ty, expected_vid
        );
        match self_ty.sty {
            ty::Infer(ty::TyVar(found_vid)) => {
                // FIXME: consider using `sub_root_var` here so we
                // can see through subtyping.
                let found_vid = self.root_var(found_vid);
                debug!("self_type_matches_expected_vid - found_vid={:?}", found_vid);
                expected_vid == found_vid
            }
            _ => false
        }
    }

    fn obligations_for_self_ty<'b>(&'b self, self_ty: ty::TyVid)
        -> impl Iterator<Item=(ty::PolyTraitRef<'tcx>, traits::PredicateObligation<'tcx>)>
           + Captures<'gcx> + 'b
    {
        // FIXME: consider using `sub_root_var` here so we
        // can see through subtyping.
        let ty_var_root = self.root_var(self_ty);
        debug!("obligations_for_self_ty: self_ty={:?} ty_var_root={:?} pending_obligations={:?}",
               self_ty, ty_var_root,
               self.fulfillment_cx.borrow().pending_obligations());

        self.fulfillment_cx
            .borrow()
            .pending_obligations()
            .into_iter()
            .filter_map(move |obligation| match obligation.predicate {
                ty::Predicate::Projection(ref data) =>
                    Some((data.to_poly_trait_ref(self.tcx), obligation)),
                ty::Predicate::Trait(ref data) =>
                    Some((data.to_poly_trait_ref(), obligation)),
                ty::Predicate::Subtype(..) => None,
                ty::Predicate::RegionOutlives(..) => None,
                ty::Predicate::TypeOutlives(..) => None,
                ty::Predicate::WellFormed(..) => None,
                ty::Predicate::ObjectSafe(..) => None,
                ty::Predicate::ConstEvaluatable(..) => None,
                // N.B., this predicate is created by breaking down a
                // `ClosureType: FnFoo()` predicate, where
                // `ClosureType` represents some `Closure`. It can't
                // possibly be referring to the current closure,
                // because we haven't produced the `Closure` for
                // this closure yet; this is exactly why the other
                // code is looking for a self type of a unresolved
                // inference variable.
                ty::Predicate::ClosureKind(..) => None,
            }).filter(move |(tr, _)| self.self_type_matches_expected_vid(*tr, ty_var_root))
    }

    fn type_var_is_sized(&self, self_ty: ty::TyVid) -> bool {
        self.obligations_for_self_ty(self_ty).any(|(tr, _)| {
            Some(tr.def_id()) == self.tcx.lang_items().sized_trait()
        })
    }

    /// Generic function that factors out common logic from function calls,
    /// method calls and overloaded operators.
    fn check_argument_types(&self,
                            sp: Span,
                            expr_sp: Span,
                            fn_inputs: &[Ty<'tcx>],
                            mut expected_arg_tys: &[Ty<'tcx>],
                            args: &'gcx [hir::Expr],
                            variadic: bool,
                            tuple_arguments: TupleArgumentsFlag,
                            def_span: Option<Span>) {
        let tcx = self.tcx;

        // Grab the argument types, supplying fresh type variables
        // if the wrong number of arguments were supplied
        let supplied_arg_count = if tuple_arguments == DontTupleArguments {
            args.len()
        } else {
            1
        };

        // All the input types from the fn signature must outlive the call
        // so as to validate implied bounds.
        for &fn_input_ty in fn_inputs {
            self.register_wf_obligation(fn_input_ty, sp, traits::MiscObligation);
        }

        let expected_arg_count = fn_inputs.len();

        let param_count_error = |expected_count: usize,
                                 arg_count: usize,
                                 error_code: &str,
                                 variadic: bool,
                                 sugg_unit: bool| {
            let mut err = tcx.sess.struct_span_err_with_code(sp,
                &format!("this function takes {}{} but {} {} supplied",
                    if variadic {"at least "} else {""},
                    potentially_plural_count(expected_count, "parameter"),
                    potentially_plural_count(arg_count, "parameter"),
                    if arg_count == 1 {"was"} else {"were"}),
                DiagnosticId::Error(error_code.to_owned()));

            if let Some(def_s) = def_span.map(|sp| tcx.sess.source_map().def_span(sp)) {
                err.span_label(def_s, "defined here");
            }
            if sugg_unit {
                let sugg_span = tcx.sess.source_map().end_point(expr_sp);
                // remove closing `)` from the span
                let sugg_span = sugg_span.shrink_to_lo();
                err.span_suggestion_with_applicability(
                    sugg_span,
                    "expected the unit value `()`; create it with empty parentheses",
                    String::from("()"),
                    Applicability::MachineApplicable);
            } else {
                err.span_label(sp, format!("expected {}{}",
                                           if variadic {"at least "} else {""},
                                           potentially_plural_count(expected_count, "parameter")));
            }
            err.emit();
        };

        let formal_tys = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolved_type(sp, fn_inputs[0]);
            match tuple_type.sty {
                ty::Tuple(arg_types) if arg_types.len() != args.len() => {
                    param_count_error(arg_types.len(), args.len(), "E0057", false, false);
                    expected_arg_tys = &[];
                    self.err_args(args.len())
                }
                ty::Tuple(arg_types) => {
                    expected_arg_tys = match expected_arg_tys.get(0) {
                        Some(&ty) => match ty.sty {
                            ty::Tuple(ref tys) => &tys,
                            _ => &[]
                        },
                        None => &[]
                    };
                    arg_types.to_vec()
                }
                _ => {
                    span_err!(tcx.sess, sp, E0059,
                        "cannot use call notation; the first type parameter \
                         for the function trait is neither a tuple nor unit");
                    expected_arg_tys = &[];
                    self.err_args(args.len())
                }
            }
        } else if expected_arg_count == supplied_arg_count {
            fn_inputs.to_vec()
        } else if variadic {
            if supplied_arg_count >= expected_arg_count {
                fn_inputs.to_vec()
            } else {
                param_count_error(expected_arg_count, supplied_arg_count, "E0060", true, false);
                expected_arg_tys = &[];
                self.err_args(supplied_arg_count)
            }
        } else {
            // is the missing argument of type `()`?
            let sugg_unit = if expected_arg_tys.len() == 1 && supplied_arg_count == 0 {
                self.resolve_type_vars_if_possible(&expected_arg_tys[0]).is_unit()
            } else if fn_inputs.len() == 1 && supplied_arg_count == 0 {
                self.resolve_type_vars_if_possible(&fn_inputs[0]).is_unit()
            } else {
                false
            };
            param_count_error(expected_arg_count, supplied_arg_count, "E0061", false, sugg_unit);

            expected_arg_tys = &[];
            self.err_args(supplied_arg_count)
        };
        // If there is no expectation, expect formal_tys.
        let expected_arg_tys = if !expected_arg_tys.is_empty() {
            expected_arg_tys
        } else {
            &formal_tys
        };

        debug!("check_argument_types: formal_tys={:?}",
               formal_tys.iter().map(|t| self.ty_to_string(*t)).collect::<Vec<String>>());

        // Check the arguments.
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for &check_closures in &[false, true] {
            debug!("check_closures={}", check_closures);

            // More awful hacks: before we check argument types, try to do
            // an "opportunistic" vtable resolution of any trait bounds on
            // the call. This helps coercions.
            if check_closures {
                self.select_obligations_where_possible(false);
            }

            // For variadic functions, we don't have a declared type for all of
            // the arguments hence we only do our usual type checking with
            // the arguments who's types we do know.
            let t = if variadic {
                expected_arg_count
            } else if tuple_arguments == TupleArguments {
                args.len()
            } else {
                supplied_arg_count
            };
            for (i, arg) in args.iter().take(t).enumerate() {
                // Warn only for the first loop (the "no closures" one).
                // Closure arguments themselves can't be diverging, but
                // a previous argument can, e.g., `foo(panic!(), || {})`.
                if !check_closures {
                    self.warn_if_unreachable(arg.id, arg.span, "expression");
                }

                let is_closure = match arg.node {
                    hir::ExprKind::Closure(..) => true,
                    _ => false
                };

                if is_closure != check_closures {
                    continue;
                }

                debug!("checking the argument");
                let formal_ty = formal_tys[i];

                // The special-cased logic below has three functions:
                // 1. Provide as good of an expected type as possible.
                let expected = Expectation::rvalue_hint(self, expected_arg_tys[i]);

                let checked_ty = self.check_expr_with_expectation(&arg, expected);

                // 2. Coerce to the most detailed type that could be coerced
                //    to, which is `expected_ty` if `rvalue_hint` returns an
                //    `ExpectHasType(expected_ty)`, or the `formal_ty` otherwise.
                let coerce_ty = expected.only_has_type(self).unwrap_or(formal_ty);
                // We're processing function arguments so we definitely want to use
                // two-phase borrows.
                self.demand_coerce(&arg, checked_ty, coerce_ty, AllowTwoPhase::Yes);

                // 3. Relate the expected type and the formal one,
                //    if the expected type was used for the coercion.
                self.demand_suptype(arg.span, formal_ty, coerce_ty);
            }
        }

        // We also need to make sure we at least write the ty of the other
        // arguments which we skipped above.
        if variadic {
            fn variadic_error<'tcx>(s: &Session, span: Span, t: Ty<'tcx>, cast_ty: &str) {
                use structured_errors::{VariadicError, StructuredDiagnostic};
                VariadicError::new(s, span, t, cast_ty).diagnostic().emit();
            }

            for arg in args.iter().skip(expected_arg_count) {
                let arg_ty = self.check_expr(&arg);

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolved_type(arg.span, arg_ty);
                match arg_ty.sty {
                    ty::Float(ast::FloatTy::F32) => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_double");
                    }
                    ty::Int(ast::IntTy::I8) | ty::Int(ast::IntTy::I16) | ty::Bool => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_int");
                    }
                    ty::Uint(ast::UintTy::U8) | ty::Uint(ast::UintTy::U16) => {
                        variadic_error(tcx.sess, arg.span, arg_ty, "c_uint");
                    }
                    ty::FnDef(..) => {
                        let ptr_ty = self.tcx.mk_fn_ptr(arg_ty.fn_sig(self.tcx));
                        let ptr_ty = self.resolve_type_vars_if_possible(&ptr_ty);
                        variadic_error(tcx.sess, arg.span, arg_ty, &ptr_ty.to_string());
                    }
                    _ => {}
                }
            }
        }
    }

    fn err_args(&self, len: usize) -> Vec<Ty<'tcx>> {
        vec![self.tcx.types.err; len]
    }

    // AST fragment checking
    fn check_lit(&self,
                 lit: &ast::Lit,
                 expected: Expectation<'tcx>)
                 -> Ty<'tcx>
    {
        let tcx = self.tcx;

        match lit.node {
            ast::LitKind::Str(..) => tcx.mk_static_str(),
            ast::LitKind::ByteStr(ref v) => {
                tcx.mk_imm_ref(tcx.types.re_static,
                               tcx.mk_array(tcx.types.u8, v.len() as u64))
            }
            ast::LitKind::Byte(_) => tcx.types.u8,
            ast::LitKind::Char(_) => tcx.types.char,
            ast::LitKind::Int(_, ast::LitIntType::Signed(t)) => tcx.mk_mach_int(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsigned(t)) => tcx.mk_mach_uint(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| {
                    match ty.sty {
                        ty::Int(_) | ty::Uint(_) => Some(ty),
                        ty::Char => Some(tcx.types.u8),
                        ty::RawPtr(..) => Some(tcx.types.usize),
                        ty::FnDef(..) | ty::FnPtr(_) => Some(tcx.types.usize),
                        _ => None
                    }
                });
                opt_ty.unwrap_or_else(
                    || tcx.mk_int_var(self.next_int_var_id()))
            }
            ast::LitKind::Float(_, t) => tcx.mk_mach_float(t),
            ast::LitKind::FloatUnsuffixed(_) => {
                let opt_ty = expected.to_option(self).and_then(|ty| {
                    match ty.sty {
                        ty::Float(_) => Some(ty),
                        _ => None
                    }
                });
                opt_ty.unwrap_or_else(
                    || tcx.mk_float_var(self.next_float_var_id()))
            }
            ast::LitKind::Bool(_) => tcx.types.bool
        }
    }

    fn check_expr_eq_type(&self,
                          expr: &'gcx hir::Expr,
                          expected: Ty<'tcx>) {
        let ty = self.check_expr_with_hint(expr, expected);
        self.demand_eqtype(expr.span, expected, ty);
    }

    pub fn check_expr_has_type_or_error(&self,
                                        expr: &'gcx hir::Expr,
                                        expected: Ty<'tcx>) -> Ty<'tcx> {
        self.check_expr_meets_expectation_or_error(expr, ExpectHasType(expected))
    }

    fn check_expr_meets_expectation_or_error(&self,
                                             expr: &'gcx hir::Expr,
                                             expected: Expectation<'tcx>) -> Ty<'tcx> {
        let expected_ty = expected.to_option(&self).unwrap_or(self.tcx.types.bool);
        let mut ty = self.check_expr_with_expectation(expr, expected);

        // While we don't allow *arbitrary* coercions here, we *do* allow
        // coercions from ! to `expected`.
        if ty.is_never() {
            assert!(!self.tables.borrow().adjustments().contains_key(expr.hir_id),
                    "expression with never type wound up being adjusted");
            let adj_ty = self.next_diverging_ty_var(
                TypeVariableOrigin::AdjustmentType(expr.span));
            self.apply_adjustments(expr, vec![Adjustment {
                kind: Adjust::NeverToAny,
                target: adj_ty
            }]);
            ty = adj_ty;
        }

        if let Some(mut err) = self.demand_suptype_diag(expr.span, expected_ty, ty) {
            // Add help to type error if this is an `if` condition with an assignment
            if let (ExpectIfCondition, &hir::ExprKind::Assign(ref lhs, ref rhs))
                 = (expected, &expr.node)
            {
                let msg = "try comparing for equality";
                if let (Ok(left), Ok(right)) = (
                    self.tcx.sess.source_map().span_to_snippet(lhs.span),
                    self.tcx.sess.source_map().span_to_snippet(rhs.span))
                {
                    err.span_suggestion_with_applicability(
                        expr.span,
                        msg,
                        format!("{} == {}", left, right),
                        Applicability::MaybeIncorrect);
                } else {
                    err.help(msg);
                }
            }
            err.emit();
        }
        ty
    }

    fn check_expr_coercable_to_type(&self,
                                    expr: &'gcx hir::Expr,
                                    expected: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.check_expr_with_hint(expr, expected);
        // checks don't need two phase
        self.demand_coerce(expr, ty, expected, AllowTwoPhase::No)
    }

    fn check_expr_with_hint(&self,
                            expr: &'gcx hir::Expr,
                            expected: Ty<'tcx>) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, ExpectHasType(expected))
    }

    fn check_expr_with_expectation(&self,
                                   expr: &'gcx hir::Expr,
                                   expected: Expectation<'tcx>) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_needs(expr, expected, Needs::None)
    }

    fn check_expr(&self, expr: &'gcx hir::Expr) -> Ty<'tcx> {
        self.check_expr_with_expectation(expr, NoExpectation)
    }

    fn check_expr_with_needs(&self, expr: &'gcx hir::Expr, needs: Needs) -> Ty<'tcx> {
        self.check_expr_with_expectation_and_needs(expr, NoExpectation, needs)
    }

    // Determine the `Self` type, using fresh variables for all variables
    // declared on the impl declaration e.g., `impl<A,B> for Vec<(A,B)>`
    // would return `($0, $1)` where `$0` and `$1` are freshly instantiated type
    // variables.
    pub fn impl_self_ty(&self,
                        span: Span, // (potential) receiver for this impl
                        did: DefId)
                        -> TypeAndSubsts<'tcx> {
        let ity = self.tcx.type_of(did);
        debug!("impl_self_ty: ity={:?}", ity);

        let substs = self.fresh_substs_for_item(span, did);
        let substd_ty = self.instantiate_type_scheme(span, &substs, &ity);

        TypeAndSubsts { substs: substs, ty: substd_ty }
    }

    /// Unifies the output type with the expected type early, for more coercions
    /// and forward type information on the input expressions.
    fn expected_inputs_for_expected_output(&self,
                                           call_span: Span,
                                           expected_ret: Expectation<'tcx>,
                                           formal_ret: Ty<'tcx>,
                                           formal_args: &[Ty<'tcx>])
                                           -> Vec<Ty<'tcx>> {
        let formal_ret = self.resolve_type_vars_with_obligations(formal_ret);
        let ret_ty = match expected_ret.only_has_type(self) {
            Some(ret) => ret,
            None => return Vec::new()
        };
        let expect_args = self.fudge_regions_if_ok(&RegionVariableOrigin::Coercion(call_span), || {
            // Attempt to apply a subtyping relationship between the formal
            // return type (likely containing type variables if the function
            // is polymorphic) and the expected return type.
            // No argument expectations are produced if unification fails.
            let origin = self.misc(call_span);
            let ures = self.at(&origin, self.param_env).sup(ret_ty, &formal_ret);

            // FIXME(#27336) can't use ? here, Try::from_error doesn't default
            // to identity so the resulting type is not constrained.
            match ures {
                Ok(ok) => {
                    // Process any obligations locally as much as
                    // we can.  We don't care if some things turn
                    // out unconstrained or ambiguous, as we're
                    // just trying to get hints here.
                    self.save_and_restore_in_snapshot_flag(|_| {
                        let mut fulfill = TraitEngine::new(self.tcx);
                        for obligation in ok.obligations {
                            fulfill.register_predicate_obligation(self, obligation);
                        }
                        fulfill.select_where_possible(self)
                    }).map_err(|_| ())?;
                }
                Err(_) => return Err(()),
            }

            // Record all the argument types, with the substitutions
            // produced from the above subtyping unification.
            Ok(formal_args.iter().map(|ty| {
                self.resolve_type_vars_if_possible(ty)
            }).collect())
        }).unwrap_or_default();
        debug!("expected_inputs_for_expected_output(formal={:?} -> {:?}, expected={:?} -> {:?})",
               formal_args, formal_ret,
               expect_args, expected_ret);
        expect_args
    }

    // Checks a method call.
    fn check_method_call(&self,
                         expr: &'gcx hir::Expr,
                         segment: &hir::PathSegment,
                         span: Span,
                         args: &'gcx [hir::Expr],
                         expected: Expectation<'tcx>,
                         needs: Needs) -> Ty<'tcx> {
        let rcvr = &args[0];
        let rcvr_t = self.check_expr_with_needs(&rcvr, needs);
        // no need to check for bot/err -- callee does that
        let rcvr_t = self.structurally_resolved_type(args[0].span, rcvr_t);

        let method = match self.lookup_method(rcvr_t,
                                              segment,
                                              span,
                                              expr,
                                              rcvr) {
            Ok(method) => {
                self.write_method_call(expr.hir_id, method);
                Ok(method)
            }
            Err(error) => {
                if segment.ident.name != keywords::Invalid.name() {
                    self.report_method_error(span,
                                             rcvr_t,
                                             segment.ident,
                                             Some(rcvr),
                                             error,
                                             Some(args));
                }
                Err(())
            }
        };

        // Call the generic checker.
        self.check_method_argument_types(span,
                                         expr.span,
                                         method,
                                         &args[1..],
                                         DontTupleArguments,
                                         expected)
    }

    fn check_return_expr(&self, return_expr: &'gcx hir::Expr) {
        let ret_coercion =
            self.ret_coercion
                .as_ref()
                .unwrap_or_else(|| span_bug!(return_expr.span,
                                             "check_return_expr called outside fn body"));

        let ret_ty = ret_coercion.borrow().expected_ty();
        let return_expr_ty = self.check_expr_with_hint(return_expr, ret_ty.clone());
        ret_coercion.borrow_mut()
                    .coerce(self,
                            &self.cause(return_expr.span,
                                        ObligationCauseCode::ReturnType(return_expr.id)),
                            return_expr,
                            return_expr_ty);
    }

    // A generic function for checking the 'then' and 'else' clauses in an 'if'
    // or 'if-else' expression.
    fn check_then_else(&self,
                       cond_expr: &'gcx hir::Expr,
                       then_expr: &'gcx hir::Expr,
                       opt_else_expr: Option<&'gcx hir::Expr>,
                       sp: Span,
                       expected: Expectation<'tcx>) -> Ty<'tcx> {
        let cond_ty = self.check_expr_meets_expectation_or_error(cond_expr, ExpectIfCondition);
        let cond_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        let expected = expected.adjust_for_branches(self);
        let then_ty = self.check_expr_with_expectation(then_expr, expected);
        let then_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);

        // We've already taken the expected type's preferences
        // into account when typing the `then` branch. To figure
        // out the initial shot at a LUB, we thus only consider
        // `expected` if it represents a *hard* constraint
        // (`only_has_type`); otherwise, we just go with a
        // fresh type variable.
        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce: DynamicCoerceMany = CoerceMany::new(coerce_to_ty);

        let if_cause = self.cause(sp, ObligationCauseCode::IfExpression);
        coerce.coerce(self, &if_cause, then_expr, then_ty);

        if let Some(else_expr) = opt_else_expr {
            let else_ty = self.check_expr_with_expectation(else_expr, expected);
            let else_diverges = self.diverges.get();

            coerce.coerce(self, &if_cause, else_expr, else_ty);

            // We won't diverge unless both branches do (or the condition does).
            self.diverges.set(cond_diverges | then_diverges & else_diverges);
        } else {
            let else_cause = self.cause(sp, ObligationCauseCode::IfExpressionWithNoElse);
            coerce.coerce_forced_unit(self, &else_cause, &mut |_| (), true);

            // If the condition is false we can't diverge.
            self.diverges.set(cond_diverges);
        }

        let result_ty = coerce.complete(self);
        if cond_ty.references_error() {
            self.tcx.types.err
        } else {
            result_ty
        }
    }

    // Check field access expressions
    fn check_field(&self,
                   expr: &'gcx hir::Expr,
                   needs: Needs,
                   base: &'gcx hir::Expr,
                   field: ast::Ident) -> Ty<'tcx> {
        let expr_t = self.check_expr_with_needs(base, needs);
        let expr_t = self.structurally_resolved_type(base.span,
                                                     expr_t);
        let mut private_candidate = None;
        let mut autoderef = self.autoderef(expr.span, expr_t);
        while let Some((base_t, _)) = autoderef.next() {
            match base_t.sty {
                ty::Adt(base_def, substs) if !base_def.is_enum() => {
                    debug!("struct named {:?}",  base_t);
                    let (ident, def_scope) =
                        self.tcx.adjust_ident(field, base_def.did, self.body_id);
                    let fields = &base_def.non_enum_variant().fields;
                    if let Some(index) = fields.iter().position(|f| f.ident.modern() == ident) {
                        let field = &fields[index];
                        let field_ty = self.field_ty(expr.span, field, substs);
                        // Save the index of all fields regardless of their visibility in case
                        // of error recovery.
                        self.write_field_index(expr.id, index);
                        if field.vis.is_accessible_from(def_scope, self.tcx) {
                            let adjustments = autoderef.adjust_steps(needs);
                            self.apply_adjustments(base, adjustments);
                            autoderef.finalize();

                            self.tcx.check_stability(field.did, Some(expr.id), expr.span);
                            return field_ty;
                        }
                        private_candidate = Some((base_def.did, field_ty));
                    }
                }
                ty::Tuple(ref tys) => {
                    let fstr = field.as_str();
                    if let Ok(index) = fstr.parse::<usize>() {
                        if fstr == index.to_string() {
                            if let Some(field_ty) = tys.get(index) {
                                let adjustments = autoderef.adjust_steps(needs);
                                self.apply_adjustments(base, adjustments);
                                autoderef.finalize();

                                self.write_field_index(expr.id, index);
                                return field_ty;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        autoderef.unambiguous_final_ty();

        if let Some((did, field_ty)) = private_candidate {
            let struct_path = self.tcx().item_path_str(did);
            let mut err = struct_span_err!(self.tcx().sess, expr.span, E0616,
                                           "field `{}` of struct `{}` is private",
                                           field, struct_path);
            // Also check if an accessible method exists, which is often what is meant.
            if self.method_exists(field, expr_t, expr.id, false) {
                err.note(&format!("a method `{}` also exists, perhaps you wish to call it", field));
            }
            err.emit();
            field_ty
        } else if field.name == keywords::Invalid.name() {
            self.tcx().types.err
        } else if self.method_exists(field, expr_t, expr.id, true) {
            type_error_struct!(self.tcx().sess, field.span, expr_t, E0615,
                               "attempted to take value of method `{}` on type `{}`",
                               field, expr_t)
                .help("maybe a `()` to call it is missing?")
                .emit();
            self.tcx().types.err
        } else {
            if !expr_t.is_primitive_ty() {
                let mut err = self.no_such_field_err(field.span, field, expr_t);

                match expr_t.sty {
                    ty::Adt(def, _) if !def.is_enum() => {
                        if let Some(suggested_field_name) =
                            Self::suggest_field_name(def.non_enum_variant(),
                                                     &field.as_str(), vec![]) {
                                err.span_label(field.span,
                                               format!("did you mean `{}`?", suggested_field_name));
                            } else {
                                err.span_label(field.span, "unknown field");
                                let struct_variant_def = def.non_enum_variant();
                                let field_names = self.available_field_names(struct_variant_def);
                                if !field_names.is_empty() {
                                    err.note(&format!("available fields are: {}",
                                                      self.name_series_display(field_names)));
                                }
                            };
                    }
                    ty::Array(_, len) => {
                        if let (Some(len), Ok(user_index)) = (
                            len.assert_usize(self.tcx),
                            field.as_str().parse::<u64>()
                        ) {
                            let base = self.tcx.hir().node_to_pretty_string(base.id);
                            let help = "instead of using tuple indexing, use array indexing";
                            let suggestion = format!("{}[{}]", base, field);
                            let applicability = if len < user_index {
                                Applicability::MachineApplicable
                            } else {
                                Applicability::MaybeIncorrect
                            };
                            err.span_suggestion_with_applicability(
                                expr.span, help, suggestion, applicability
                            );
                        }
                    }
                    ty::RawPtr(..) => {
                        let base = self.tcx.hir().node_to_pretty_string(base.id);
                        let msg = format!("`{}` is a native pointer; try dereferencing it", base);
                        let suggestion = format!("(*{}).{}", base, field);
                        err.span_suggestion_with_applicability(
                            field.span,
                            &msg,
                            suggestion,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    _ => {}
                }
                err
            } else {
                type_error_struct!(self.tcx().sess, field.span, expr_t, E0610,
                                   "`{}` is a primitive type and therefore doesn't have fields",
                                   expr_t)
            }.emit();
            self.tcx().types.err
        }
    }

    // Return an hint about the closest match in field names
    fn suggest_field_name(variant: &'tcx ty::VariantDef,
                          field: &str,
                          skip: Vec<LocalInternedString>)
                          -> Option<Symbol> {
        let names = variant.fields.iter().filter_map(|field| {
            // ignore already set fields and private fields from non-local crates
            if skip.iter().any(|x| *x == field.ident.as_str()) ||
               (variant.did.krate != LOCAL_CRATE && field.vis != Visibility::Public) {
                None
            } else {
                Some(&field.ident.name)
            }
        });

        find_best_match_for_name(names, field, None)
    }

    fn available_field_names(&self, variant: &'tcx ty::VariantDef) -> Vec<ast::Name> {
        variant.fields.iter().filter(|field| {
            let def_scope = self.tcx.adjust_ident(field.ident, variant.did, self.body_id).1;
            field.vis.is_accessible_from(def_scope, self.tcx)
        })
        .map(|field| field.ident.name)
        .collect()
    }

    fn name_series_display(&self, names: Vec<ast::Name>) -> String {
        // dynamic limit, to never omit just one field
        let limit = if names.len() == 6 { 6 } else { 5 };
        let mut display = names.iter().take(limit)
            .map(|n| format!("`{}`", n)).collect::<Vec<_>>().join(", ");
        if names.len() > limit {
            display = format!("{} ... and {} others", display, names.len() - limit);
        }
        display
    }

    fn no_such_field_err<T: Display>(&self, span: Span, field: T, expr_t: &ty::TyS)
        -> DiagnosticBuilder {
        type_error_struct!(self.tcx().sess, span, expr_t, E0609,
                           "no field `{}` on type `{}`",
                           field, expr_t)
    }

    fn report_unknown_field(&self,
                            ty: Ty<'tcx>,
                            variant: &'tcx ty::VariantDef,
                            field: &hir::Field,
                            skip_fields: &[hir::Field],
                            kind_name: &str) {
        let mut err = self.type_error_struct_with_diag(
            field.ident.span,
            |actual| match ty.sty {
                ty::Adt(adt, ..) if adt.is_enum() => {
                    struct_span_err!(self.tcx.sess, field.ident.span, E0559,
                                     "{} `{}::{}` has no field named `{}`",
                                     kind_name, actual, variant.name, field.ident)
                }
                _ => {
                    struct_span_err!(self.tcx.sess, field.ident.span, E0560,
                                     "{} `{}` has no field named `{}`",
                                     kind_name, actual, field.ident)
                }
            },
            ty);
        // prevent all specified fields from being suggested
        let skip_fields = skip_fields.iter().map(|ref x| x.ident.as_str());
        if let Some(field_name) = Self::suggest_field_name(variant,
                                                           &field.ident.as_str(),
                                                           skip_fields.collect()) {
            err.span_label(field.ident.span,
                           format!("field does not exist - did you mean `{}`?", field_name));
        } else {
            match ty.sty {
                ty::Adt(adt, ..) => {
                    if adt.is_enum() {
                        err.span_label(field.ident.span,
                                       format!("`{}::{}` does not have this field",
                                               ty, variant.name));
                    } else {
                        err.span_label(field.ident.span,
                                       format!("`{}` does not have this field", ty));
                    }
                    let available_field_names = self.available_field_names(variant);
                    if !available_field_names.is_empty() {
                        err.note(&format!("available fields are: {}",
                                          self.name_series_display(available_field_names)));
                    }
                }
                _ => bug!("non-ADT passed to report_unknown_field")
            }
        };
        err.emit();
    }

    fn check_expr_struct_fields(&self,
                                adt_ty: Ty<'tcx>,
                                expected: Expectation<'tcx>,
                                expr_id: ast::NodeId,
                                span: Span,
                                variant: &'tcx ty::VariantDef,
                                ast_fields: &'gcx [hir::Field],
                                check_completeness: bool) -> bool {
        let tcx = self.tcx;

        let adt_ty_hint =
            self.expected_inputs_for_expected_output(span, expected, adt_ty, &[adt_ty])
                .get(0).cloned().unwrap_or(adt_ty);
        // re-link the regions that EIfEO can erase.
        self.demand_eqtype(span, adt_ty_hint, adt_ty);

        let (substs, adt_kind, kind_name) = match &adt_ty.sty {
            &ty::Adt(adt, substs) => {
                (substs, adt.adt_kind(), adt.variant_descr())
            }
            _ => span_bug!(span, "non-ADT passed to check_expr_struct_fields")
        };

        let mut remaining_fields = variant.fields.iter().enumerate().map(|(i, field)|
            (field.ident.modern(), (i, field))
        ).collect::<FxHashMap<_, _>>();

        let mut seen_fields = FxHashMap::default();

        let mut error_happened = false;

        // Type-check each field.
        for field in ast_fields {
            let ident = tcx.adjust_ident(field.ident, variant.did, self.body_id).0;
            let field_type = if let Some((i, v_field)) = remaining_fields.remove(&ident) {
                seen_fields.insert(ident, field.span);
                self.write_field_index(field.id, i);

                // We don't look at stability attributes on
                // struct-like enums (yet...), but it's definitely not
                // a bug to have constructed one.
                if adt_kind != AdtKind::Enum {
                    tcx.check_stability(v_field.did, Some(expr_id), field.span);
                }

                self.field_ty(field.span, v_field, substs)
            } else {
                error_happened = true;
                if let Some(prev_span) = seen_fields.get(&ident) {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                   field.ident.span,
                                                   E0062,
                                                   "field `{}` specified more than once",
                                                   ident);

                    err.span_label(field.ident.span, "used more than once");
                    err.span_label(*prev_span, format!("first use of `{}`", ident));

                    err.emit();
                } else {
                    self.report_unknown_field(adt_ty, variant, field, ast_fields, kind_name);
                }

                tcx.types.err
            };

            // Make sure to give a type to the field even if there's
            // an error, so we can continue type-checking.
            self.check_expr_coercable_to_type(&field.expr, field_type);
        }

        // Make sure the programmer specified correct number of fields.
        if kind_name == "union" {
            if ast_fields.len() != 1 {
                tcx.sess.span_err(span, "union expressions should have exactly one field");
            }
        } else if check_completeness && !error_happened && !remaining_fields.is_empty() {
            let len = remaining_fields.len();

            let mut displayable_field_names = remaining_fields
                                              .keys()
                                              .map(|ident| ident.as_str())
                                              .collect::<Vec<_>>();

            displayable_field_names.sort();

            let truncated_fields_error = if len <= 3 {
                String::new()
            } else {
                format!(" and {} other field{}", (len - 3), if len - 3 == 1 {""} else {"s"})
            };

            let remaining_fields_names = displayable_field_names.iter().take(3)
                                        .map(|n| format!("`{}`", n))
                                        .collect::<Vec<_>>()
                                        .join(", ");

            struct_span_err!(tcx.sess, span, E0063,
                             "missing field{} {}{} in initializer of `{}`",
                             if remaining_fields.len() == 1 { "" } else { "s" },
                             remaining_fields_names,
                             truncated_fields_error,
                             adt_ty)
                .span_label(span, format!("missing {}{}",
                                          remaining_fields_names,
                                          truncated_fields_error))
                .emit();
        }
        error_happened
    }

    fn check_struct_fields_on_error(&self,
                                    fields: &'gcx [hir::Field],
                                    base_expr: &'gcx Option<P<hir::Expr>>) {
        for field in fields {
            self.check_expr(&field.expr);
        }
        if let Some(ref base) = *base_expr {
            self.check_expr(&base);
        }
    }

    pub fn check_struct_path(&self,
                             qpath: &hir::QPath,
                             node_id: ast::NodeId)
                             -> Option<(&'tcx ty::VariantDef,  Ty<'tcx>)> {
        let path_span = match *qpath {
            hir::QPath::Resolved(_, ref path) => path.span,
            hir::QPath::TypeRelative(ref qself, _) => qself.span
        };
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, node_id);
        let variant = match def {
            Def::Err => {
                self.set_tainted_by_errors();
                return None;
            }
            Def::Variant(..) => {
                match ty.sty {
                    ty::Adt(adt, substs) => {
                        Some((adt.variant_of_def(def), adt.did, substs))
                    }
                    _ => bug!("unexpected type: {:?}", ty.sty)
                }
            }
            Def::Struct(..) | Def::Union(..) | Def::TyAlias(..) |
            Def::AssociatedTy(..) | Def::SelfTy(..) => {
                match ty.sty {
                    ty::Adt(adt, substs) if !adt.is_enum() => {
                        Some((adt.non_enum_variant(), adt.did, substs))
                    }
                    _ => None,
                }
            }
            _ => bug!("unexpected definition: {:?}", def)
        };

        if let Some((variant, did, substs)) = variant {
            debug!("check_struct_path: did={:?} substs={:?}", did, substs);
            let hir_id = self.tcx.hir().node_to_hir_id(node_id);
            self.write_user_substs_from_substs(hir_id, substs, None);

            // Check bounds on type arguments used in the path.
            let bounds = self.instantiate_bounds(path_span, did, substs);
            let cause = traits::ObligationCause::new(path_span, self.body_id,
                                                     traits::ItemObligation(did));
            self.add_obligations_for_parameters(cause, &bounds);

            Some((variant, ty))
        } else {
            struct_span_err!(self.tcx.sess, path_span, E0071,
                             "expected struct, variant or union type, found {}",
                             ty.sort_string(self.tcx))
                .span_label(path_span, "not a struct")
                .emit();
            None
        }
    }

    fn check_expr_struct(&self,
                         expr: &hir::Expr,
                         expected: Expectation<'tcx>,
                         qpath: &hir::QPath,
                         fields: &'gcx [hir::Field],
                         base_expr: &'gcx Option<P<hir::Expr>>) -> Ty<'tcx>
    {
        // Find the relevant variant
        let (variant, adt_ty) =
            if let Some(variant_ty) = self.check_struct_path(qpath, expr.id) {
                variant_ty
            } else {
                self.check_struct_fields_on_error(fields, base_expr);
                return self.tcx.types.err;
            };

        let path_span = match *qpath {
            hir::QPath::Resolved(_, ref path) => path.span,
            hir::QPath::TypeRelative(ref qself, _) => qself.span
        };

        // Prohibit struct expressions when non-exhaustive flag is set.
        let adt = adt_ty.ty_adt_def().expect("`check_struct_path` returned non-ADT type");
        if !adt.did.is_local() && variant.is_field_list_non_exhaustive() {
            span_err!(self.tcx.sess, expr.span, E0639,
                      "cannot create non-exhaustive {} using struct expression",
                      adt.variant_descr());
        }

        let error_happened = self.check_expr_struct_fields(adt_ty, expected, expr.id, path_span,
                                                           variant, fields, base_expr.is_none());
        if let &Some(ref base_expr) = base_expr {
            // If check_expr_struct_fields hit an error, do not attempt to populate
            // the fields with the base_expr. This could cause us to hit errors later
            // when certain fields are assumed to exist that in fact do not.
            if !error_happened {
                self.check_expr_has_type_or_error(base_expr, adt_ty);
                match adt_ty.sty {
                    ty::Adt(adt, substs) if adt.is_struct() => {
                        let fru_field_types = adt.non_enum_variant().fields.iter().map(|f| {
                            self.normalize_associated_types_in(expr.span, &f.ty(self.tcx, substs))
                        }).collect();

                        self.tables
                            .borrow_mut()
                            .fru_field_types_mut()
                            .insert(expr.hir_id, fru_field_types);
                    }
                    _ => {
                        span_err!(self.tcx.sess, base_expr.span, E0436,
                                  "functional record update syntax requires a struct");
                    }
                }
            }
        }
        self.require_type_is_sized(adt_ty, expr.span, traits::StructInitializerSized);
        adt_ty
    }


    /// Invariant:
    /// If an expression has any sub-expressions that result in a type error,
    /// inspecting that expression's type with `ty.references_error()` will return
    /// true. Likewise, if an expression is known to diverge, inspecting its
    /// type with `ty::type_is_bot` will return true (n.b.: since Rust is
    /// strict, _|_ can appear in the type of an expression that does not,
    /// itself, diverge: for example, fn() -> _|_.)
    /// Note that inspecting a type's structure *directly* may expose the fact
    /// that there are actually multiple representations for `Error`, so avoid
    /// that when err needs to be handled differently.
    fn check_expr_with_expectation_and_needs(&self,
                                             expr: &'gcx hir::Expr,
                                             expected: Expectation<'tcx>,
                                             needs: Needs) -> Ty<'tcx> {
        debug!(">> type-checking: expr={:?} expected={:?}",
               expr, expected);

        // Warn for expressions after diverging siblings.
        self.warn_if_unreachable(expr.id, expr.span, "expression");

        // Hide the outer diverging and has_errors flags.
        let old_diverges = self.diverges.get();
        let old_has_errors = self.has_errors.get();
        self.diverges.set(Diverges::Maybe);
        self.has_errors.set(false);

        let ty = self.check_expr_kind(expr, expected, needs);

        // Warn for non-block expressions with diverging children.
        match expr.node {
            hir::ExprKind::Block(..) |
            hir::ExprKind::Loop(..) | hir::ExprKind::While(..) |
            hir::ExprKind::If(..) | hir::ExprKind::Match(..) => {}

            _ => self.warn_if_unreachable(expr.id, expr.span, "expression")
        }

        // Any expression that produces a value of type `!` must have diverged
        if ty.is_never() {
            self.diverges.set(self.diverges.get() | Diverges::Always);
        }

        // Record the type, which applies it effects.
        // We need to do this after the warning above, so that
        // we don't warn for the diverging expression itself.
        self.write_ty(expr.hir_id, ty);

        // Combine the diverging and has_error flags.
        self.diverges.set(self.diverges.get() | old_diverges);
        self.has_errors.set(self.has_errors.get() | old_has_errors);

        debug!("type of {} is...", self.tcx.hir().node_to_string(expr.id));
        debug!("... {:?}, expected is {:?}", ty, expected);

        ty
    }

    fn check_expr_kind(
        &self,
        expr: &'gcx hir::Expr,
        expected: Expectation<'tcx>,
        needs: Needs
    ) -> Ty<'tcx> {
        debug!(
            "check_expr_kind(expr={:?}, expected={:?}, needs={:?})",
            expr,
            expected,
            needs,
        );

        let tcx = self.tcx;
        let id = expr.id;
        match expr.node {
            hir::ExprKind::Box(ref subexpr) => {
                let expected_inner = expected.to_option(self).map_or(NoExpectation, |ty| {
                    match ty.sty {
                        ty::Adt(def, _) if def.is_box()
                            => Expectation::rvalue_hint(self, ty.boxed_ty()),
                        _ => NoExpectation
                    }
                });
                let referent_ty = self.check_expr_with_expectation(subexpr, expected_inner);
                tcx.mk_box(referent_ty)
            }

            hir::ExprKind::Lit(ref lit) => {
                self.check_lit(&lit, expected)
            }
            hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.check_binop(expr, op, lhs, rhs)
            }
            hir::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                self.check_binop_assign(expr, op, lhs, rhs)
            }
            hir::ExprKind::Unary(unop, ref oprnd) => {
                let expected_inner = match unop {
                    hir::UnNot | hir::UnNeg => {
                        expected
                    }
                    hir::UnDeref => {
                        NoExpectation
                    }
                };
                let needs = match unop {
                    hir::UnDeref => needs,
                    _ => Needs::None
                };
                let mut oprnd_t = self.check_expr_with_expectation_and_needs(&oprnd,
                                                                             expected_inner,
                                                                             needs);

                if !oprnd_t.references_error() {
                    oprnd_t = self.structurally_resolved_type(expr.span, oprnd_t);
                    match unop {
                        hir::UnDeref => {
                            if let Some(mt) = oprnd_t.builtin_deref(true) {
                                oprnd_t = mt.ty;
                            } else if let Some(ok) = self.try_overloaded_deref(
                                    expr.span, oprnd_t, needs) {
                                let method = self.register_infer_ok_obligations(ok);
                                if let ty::Ref(region, _, mutbl) = method.sig.inputs()[0].sty {
                                    let mutbl = match mutbl {
                                        hir::MutImmutable => AutoBorrowMutability::Immutable,
                                        hir::MutMutable => AutoBorrowMutability::Mutable {
                                            // (It shouldn't actually matter for unary ops whether
                                            // we enable two-phase borrows or not, since a unary
                                            // op has no additional operands.)
                                            allow_two_phase_borrow: AllowTwoPhase::No,
                                        }
                                    };
                                    self.apply_adjustments(oprnd, vec![Adjustment {
                                        kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                                        target: method.sig.inputs()[0]
                                    }]);
                                }
                                oprnd_t = self.make_overloaded_place_return_type(method).ty;
                                self.write_method_call(expr.hir_id, method);
                            } else {
                                type_error_struct!(tcx.sess, expr.span, oprnd_t, E0614,
                                                   "type `{}` cannot be dereferenced",
                                                   oprnd_t).emit();
                                oprnd_t = tcx.types.err;
                            }
                        }
                        hir::UnNot => {
                            let result = self.check_user_unop(expr, oprnd_t, unop);
                            // If it's builtin, we can reuse the type, this helps inference.
                            if !(oprnd_t.is_integral() || oprnd_t.sty == ty::Bool) {
                                oprnd_t = result;
                            }
                        }
                        hir::UnNeg => {
                            let result = self.check_user_unop(expr, oprnd_t, unop);
                            // If it's builtin, we can reuse the type, this helps inference.
                            if !(oprnd_t.is_integral() || oprnd_t.is_fp()) {
                                oprnd_t = result;
                            }
                        }
                    }
                }
                oprnd_t
            }
            hir::ExprKind::AddrOf(mutbl, ref oprnd) => {
                let hint = expected.only_has_type(self).map_or(NoExpectation, |ty| {
                    match ty.sty {
                        ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                            if oprnd.is_place_expr() {
                                // Places may legitimately have unsized types.
                                // For example, dereferences of a fat pointer and
                                // the last field of a struct can be unsized.
                                ExpectHasType(ty)
                            } else {
                                Expectation::rvalue_hint(self, ty)
                            }
                        }
                        _ => NoExpectation
                    }
                });
                let needs = Needs::maybe_mut_place(mutbl);
                let ty = self.check_expr_with_expectation_and_needs(&oprnd, hint, needs);

                let tm = ty::TypeAndMut { ty: ty, mutbl: mutbl };
                if tm.ty.references_error() {
                    tcx.types.err
                } else {
                    // Note: at this point, we cannot say what the best lifetime
                    // is to use for resulting pointer.  We want to use the
                    // shortest lifetime possible so as to avoid spurious borrowck
                    // errors.  Moreover, the longest lifetime will depend on the
                    // precise details of the value whose address is being taken
                    // (and how long it is valid), which we don't know yet until type
                    // inference is complete.
                    //
                    // Therefore, here we simply generate a region variable.  The
                    // region inferencer will then select the ultimate value.
                    // Finally, borrowck is charged with guaranteeing that the
                    // value whose address was taken can actually be made to live
                    // as long as it needs to live.
                    let region = self.next_region_var(infer::AddrOfRegion(expr.span));
                    tcx.mk_ref(region, tm)
                }
            }
            hir::ExprKind::Path(ref qpath) => {
                let (def, opt_ty, segs) = self.resolve_ty_and_def_ufcs(qpath, expr.id, expr.span);
                let ty = if def != Def::Err {
                    self.instantiate_value_path(segs, opt_ty, def, expr.span, id).0
                } else {
                    self.set_tainted_by_errors();
                    tcx.types.err
                };

                if let ty::FnDef(..) = ty.sty {
                    let fn_sig = ty.fn_sig(tcx);
                    if !tcx.features().unsized_locals {
                        // We want to remove some Sized bounds from std functions,
                        // but don't want to expose the removal to stable Rust.
                        // i.e., we don't want to allow
                        //
                        // ```rust
                        // drop as fn(str);
                        // ```
                        //
                        // to work in stable even if the Sized bound on `drop` is relaxed.
                        for i in 0..fn_sig.inputs().skip_binder().len() {
                            // We just want to check sizedness, so instead of introducing
                            // placeholder lifetimes with probing, we just replace higher lifetimes
                            // with fresh vars.
                            let input = self.replace_bound_vars_with_fresh_vars(
                                expr.span,
                                infer::LateBoundRegionConversionTime::FnCall,
                                &fn_sig.input(i)).0;
                            self.require_type_is_sized_deferred(input, expr.span,
                                                                traits::SizedArgumentType);
                        }
                    }
                    // Here we want to prevent struct constructors from returning unsized types.
                    // There were two cases this happened: fn pointer coercion in stable
                    // and usual function call in presense of unsized_locals.
                    // Also, as we just want to check sizedness, instead of introducing
                    // placeholder lifetimes with probing, we just replace higher lifetimes
                    // with fresh vars.
                    let output = self.replace_bound_vars_with_fresh_vars(
                        expr.span,
                        infer::LateBoundRegionConversionTime::FnCall,
                        &fn_sig.output()).0;
                    self.require_type_is_sized_deferred(output, expr.span, traits::SizedReturnType);
                }

                // We always require that the type provided as the value for
                // a type parameter outlives the moment of instantiation.
                let substs = self.tables.borrow().node_substs(expr.hir_id);
                self.add_wf_bounds(substs, expr);

                ty
            }
            hir::ExprKind::InlineAsm(_, ref outputs, ref inputs) => {
                for expr in outputs.iter().chain(inputs.iter()) {
                    self.check_expr(expr);
                }
                tcx.mk_unit()
            }
            hir::ExprKind::Break(destination, ref expr_opt) => {
                if let Ok(target_id) = destination.target_id {
                    let (e_ty, cause);
                    if let Some(ref e) = *expr_opt {
                        // If this is a break with a value, we need to type-check
                        // the expression. Get an expected type from the loop context.
                        let opt_coerce_to = {
                            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
                            enclosing_breakables.find_breakable(target_id)
                                                .coerce
                                                .as_ref()
                                                .map(|coerce| coerce.expected_ty())
                        };

                        // If the loop context is not a `loop { }`, then break with
                        // a value is illegal, and `opt_coerce_to` will be `None`.
                        // Just set expectation to error in that case.
                        let coerce_to = opt_coerce_to.unwrap_or(tcx.types.err);

                        // Recurse without `enclosing_breakables` borrowed.
                        e_ty = self.check_expr_with_hint(e, coerce_to);
                        cause = self.misc(e.span);
                    } else {
                        // Otherwise, this is a break *without* a value. That's
                        // always legal, and is equivalent to `break ()`.
                        e_ty = tcx.mk_unit();
                        cause = self.misc(expr.span);
                    }

                    // Now that we have type-checked `expr_opt`, borrow
                    // the `enclosing_loops` field and let's coerce the
                    // type of `expr_opt` into what is expected.
                    let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
                    let ctxt = enclosing_breakables.find_breakable(target_id);
                    if let Some(ref mut coerce) = ctxt.coerce {
                        if let Some(ref e) = *expr_opt {
                            coerce.coerce(self, &cause, e, e_ty);
                        } else {
                            assert!(e_ty.is_unit());
                            coerce.coerce_forced_unit(self, &cause, &mut |_| (), true);
                        }
                    } else {
                        // If `ctxt.coerce` is `None`, we can just ignore
                        // the type of the expresison.  This is because
                        // either this was a break *without* a value, in
                        // which case it is always a legal type (`()`), or
                        // else an error would have been flagged by the
                        // `loops` pass for using break with an expression
                        // where you are not supposed to.
                        assert!(expr_opt.is_none() || self.tcx.sess.err_count() > 0);
                    }

                    ctxt.may_break = true;

                    // the type of a `break` is always `!`, since it diverges
                    tcx.types.never
                } else {
                    // Otherwise, we failed to find the enclosing loop;
                    // this can only happen if the `break` was not
                    // inside a loop at all, which is caught by the
                    // loop-checking pass.
                    if self.tcx.sess.err_count() == 0 {
                        self.tcx.sess.delay_span_bug(expr.span,
                            "break was outside loop, but no error was emitted");
                    }

                    // We still need to assign a type to the inner expression to
                    // prevent the ICE in #43162.
                    if let Some(ref e) = *expr_opt {
                        self.check_expr_with_hint(e, tcx.types.err);

                        // ... except when we try to 'break rust;'.
                        // ICE this expression in particular (see #43162).
                        if let hir::ExprKind::Path(hir::QPath::Resolved(_, ref path)) = e.node {
                            if path.segments.len() == 1 && path.segments[0].ident.name == "rust" {
                                fatally_break_rust(self.tcx.sess);
                            }
                        }
                    }
                    // There was an error; make type-check fail.
                    tcx.types.err
                }

            }
            hir::ExprKind::Continue(destination) => {
                if destination.target_id.is_ok() {
                    tcx.types.never
                } else {
                    // There was an error; make type-check fail.
                    tcx.types.err
                }
            }
            hir::ExprKind::Ret(ref expr_opt) => {
                if self.ret_coercion.is_none() {
                    struct_span_err!(self.tcx.sess, expr.span, E0572,
                                     "return statement outside of function body").emit();
                } else if let Some(ref e) = *expr_opt {
                    self.check_return_expr(e);
                } else {
                    let mut coercion = self.ret_coercion.as_ref().unwrap().borrow_mut();
                    let cause = self.cause(expr.span, ObligationCauseCode::ReturnNoExpression);
                    coercion.coerce_forced_unit(self, &cause, &mut |_| (), true);
                }
                tcx.types.never
            }
            hir::ExprKind::Assign(ref lhs, ref rhs) => {
                let lhs_ty = self.check_expr_with_needs(&lhs, Needs::MutPlace);

                let rhs_ty = self.check_expr_coercable_to_type(&rhs, lhs_ty);

                match expected {
                    ExpectIfCondition => {
                        self.tcx.sess.delay_span_bug(lhs.span, "invalid lhs expression in if;\
                                                                expected error elsehwere");
                    }
                    _ => {
                        // Only check this if not in an `if` condition, as the
                        // mistyped comparison help is more appropriate.
                        if !lhs.is_place_expr() {
                            struct_span_err!(self.tcx.sess, expr.span, E0070,
                                                "invalid left-hand side expression")
                                .span_label(expr.span, "left-hand of expression not valid")
                                .emit();
                        }
                    }
                }

                self.require_type_is_sized(lhs_ty, lhs.span, traits::AssignmentLhsSized);

                if lhs_ty.references_error() || rhs_ty.references_error() {
                    tcx.types.err
                } else {
                    tcx.mk_unit()
                }
            }
            hir::ExprKind::If(ref cond, ref then_expr, ref opt_else_expr) => {
                self.check_then_else(&cond, then_expr, opt_else_expr.as_ref().map(|e| &**e),
                                     expr.span, expected)
            }
            hir::ExprKind::While(ref cond, ref body, _) => {
                let ctxt = BreakableCtxt {
                    // cannot use break with a value from a while loop
                    coerce: None,
                    may_break: false,  // Will get updated if/when we find a `break`.
                };

                let (ctxt, ()) = self.with_breakable_ctxt(expr.id, ctxt, || {
                    self.check_expr_has_type_or_error(&cond, tcx.types.bool);
                    let cond_diverging = self.diverges.get();
                    self.check_block_no_value(&body);

                    // We may never reach the body so it diverging means nothing.
                    self.diverges.set(cond_diverging);
                });

                if ctxt.may_break {
                    // No way to know whether it's diverging because
                    // of a `break` or an outer `break` or `return`.
                    self.diverges.set(Diverges::Maybe);
                }

                self.tcx.mk_unit()
            }
            hir::ExprKind::Loop(ref body, _, source) => {
                let coerce = match source {
                    // you can only use break with a value from a normal `loop { }`
                    hir::LoopSource::Loop => {
                        let coerce_to = expected.coercion_target_type(self, body.span);
                        Some(CoerceMany::new(coerce_to))
                    }

                    hir::LoopSource::WhileLet |
                    hir::LoopSource::ForLoop => {
                        None
                    }
                };

                let ctxt = BreakableCtxt {
                    coerce,
                    may_break: false, // Will get updated if/when we find a `break`.
                };

                let (ctxt, ()) = self.with_breakable_ctxt(expr.id, ctxt, || {
                    self.check_block_no_value(&body);
                });

                if ctxt.may_break {
                    // No way to know whether it's diverging because
                    // of a `break` or an outer `break` or `return`.
                    self.diverges.set(Diverges::Maybe);
                }

                // If we permit break with a value, then result type is
                // the LUB of the breaks (possibly ! if none); else, it
                // is nil. This makes sense because infinite loops
                // (which would have type !) are only possible iff we
                // permit break with a value [1].
                if ctxt.coerce.is_none() && !ctxt.may_break {
                    // [1]
                    self.tcx.sess.delay_span_bug(body.span, "no coercion, but loop may not break");
                }
                ctxt.coerce.map(|c| c.complete(self)).unwrap_or_else(|| self.tcx.mk_unit())
            }
            hir::ExprKind::Match(ref discrim, ref arms, match_src) => {
                self.check_match(expr, &discrim, arms, expected, match_src)
            }
            hir::ExprKind::Closure(capture, ref decl, body_id, _, gen) => {
                self.check_expr_closure(expr, capture, &decl, body_id, gen, expected)
            }
            hir::ExprKind::Block(ref body, _) => {
                self.check_block_with_expected(&body, expected)
            }
            hir::ExprKind::Call(ref callee, ref args) => {
                self.check_call(expr, &callee, args, expected)
            }
            hir::ExprKind::MethodCall(ref segment, span, ref args) => {
                self.check_method_call(expr, segment, span, args, expected, needs)
            }
            hir::ExprKind::Cast(ref e, ref t) => {
                // Find the type of `e`. Supply hints based on the type we are casting to,
                // if appropriate.
                let t_cast = self.to_ty_saving_user_provided_ty(t);
                let t_cast = self.resolve_type_vars_if_possible(&t_cast);
                let t_expr = self.check_expr_with_expectation(e, ExpectCastableToType(t_cast));
                let t_cast = self.resolve_type_vars_if_possible(&t_cast);

                // Eagerly check for some obvious errors.
                if t_expr.references_error() || t_cast.references_error() {
                    tcx.types.err
                } else {
                    // Defer other checks until we're done type checking.
                    let mut deferred_cast_checks = self.deferred_cast_checks.borrow_mut();
                    match cast::CastCheck::new(self, e, t_expr, t_cast, t.span, expr.span) {
                        Ok(cast_check) => {
                            deferred_cast_checks.push(cast_check);
                            t_cast
                        }
                        Err(ErrorReported) => {
                            tcx.types.err
                        }
                    }
                }
            }
            hir::ExprKind::Type(ref e, ref t) => {
                let ty = self.to_ty_saving_user_provided_ty(&t);
                self.check_expr_eq_type(&e, ty);
                ty
            }
            hir::ExprKind::Array(ref args) => {
                let uty = expected.to_option(self).and_then(|uty| {
                    match uty.sty {
                        ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                        _ => None
                    }
                });

                let element_ty = if !args.is_empty() {
                    let coerce_to = uty.unwrap_or_else(
                        || self.next_ty_var(TypeVariableOrigin::TypeInference(expr.span)));
                    let mut coerce = CoerceMany::with_coercion_sites(coerce_to, args);
                    assert_eq!(self.diverges.get(), Diverges::Maybe);
                    for e in args {
                        let e_ty = self.check_expr_with_hint(e, coerce_to);
                        let cause = self.misc(e.span);
                        coerce.coerce(self, &cause, e, e_ty);
                    }
                    coerce.complete(self)
                } else {
                    self.next_ty_var(TypeVariableOrigin::TypeInference(expr.span))
                };
                tcx.mk_array(element_ty, args.len() as u64)
            }
            hir::ExprKind::Repeat(ref element, ref count) => {
                let count_def_id = tcx.hir().local_def_id(count.id);
                let param_env = ty::ParamEnv::empty();
                let substs = Substs::identity_for_item(tcx.global_tcx(), count_def_id);
                let instance = ty::Instance::resolve(
                    tcx.global_tcx(),
                    param_env,
                    count_def_id,
                    substs,
                ).unwrap();
                let global_id = GlobalId {
                    instance,
                    promoted: None
                };
                let count = tcx.const_eval(param_env.and(global_id));

                let uty = match expected {
                    ExpectHasType(uty) => {
                        match uty.sty {
                            ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                            _ => None
                        }
                    }
                    _ => None
                };

                let (element_ty, t) = match uty {
                    Some(uty) => {
                        self.check_expr_coercable_to_type(&element, uty);
                        (uty, uty)
                    }
                    None => {
                        let ty = self.next_ty_var(TypeVariableOrigin::MiscVariable(element.span));
                        let element_ty = self.check_expr_has_type_or_error(&element, ty);
                        (element_ty, ty)
                    }
                };

                if let Ok(count) = count {
                    let zero_or_one = count.assert_usize(tcx).map_or(false, |count| count <= 1);
                    if !zero_or_one {
                        // For [foo, ..n] where n > 1, `foo` must have
                        // Copy type:
                        let lang_item = self.tcx.require_lang_item(lang_items::CopyTraitLangItem);
                        self.require_type_meets(t, expr.span, traits::RepeatVec, lang_item);
                    }
                }

                if element_ty.references_error() {
                    tcx.types.err
                } else if let Ok(count) = count {
                    tcx.mk_ty(ty::Array(t, count))
                } else {
                    tcx.types.err
                }
            }
            hir::ExprKind::Tup(ref elts) => {
                let flds = expected.only_has_type(self).and_then(|ty| {
                    let ty = self.resolve_type_vars_with_obligations(ty);
                    match ty.sty {
                        ty::Tuple(ref flds) => Some(&flds[..]),
                        _ => None
                    }
                });

                let elt_ts_iter = elts.iter().enumerate().map(|(i, e)| {
                    let t = match flds {
                        Some(ref fs) if i < fs.len() => {
                            let ety = fs[i];
                            self.check_expr_coercable_to_type(&e, ety);
                            ety
                        }
                        _ => {
                            self.check_expr_with_expectation(&e, NoExpectation)
                        }
                    };
                    t
                });
                let tuple = tcx.mk_tup(elt_ts_iter);
                if tuple.references_error() {
                    tcx.types.err
                } else {
                    self.require_type_is_sized(tuple, expr.span, traits::TupleInitializerSized);
                    tuple
                }
            }
            hir::ExprKind::Struct(ref qpath, ref fields, ref base_expr) => {
                self.check_expr_struct(expr, expected, qpath, fields, base_expr)
            }
            hir::ExprKind::Field(ref base, field) => {
                self.check_field(expr, needs, &base, field)
            }
            hir::ExprKind::Index(ref base, ref idx) => {
                let base_t = self.check_expr_with_needs(&base, needs);
                let idx_t = self.check_expr(&idx);

                if base_t.references_error() {
                    base_t
                } else if idx_t.references_error() {
                    idx_t
                } else {
                    let base_t = self.structurally_resolved_type(base.span, base_t);
                    match self.lookup_indexing(expr, base, base_t, idx_t, needs) {
                        Some((index_ty, element_ty)) => {
                            // two-phase not needed because index_ty is never mutable
                            self.demand_coerce(idx, idx_t, index_ty, AllowTwoPhase::No);
                            element_ty
                        }
                        None => {
                            let mut err =
                                type_error_struct!(tcx.sess, expr.span, base_t, E0608,
                                                   "cannot index into a value of type `{}`",
                                                   base_t);
                            // Try to give some advice about indexing tuples.
                            if let ty::Tuple(..) = base_t.sty {
                                let mut needs_note = true;
                                // If the index is an integer, we can show the actual
                                // fixed expression:
                                if let hir::ExprKind::Lit(ref lit) = idx.node {
                                    if let ast::LitKind::Int(i,
                                            ast::LitIntType::Unsuffixed) = lit.node {
                                        let snip = tcx.sess.source_map().span_to_snippet(base.span);
                                        if let Ok(snip) = snip {
                                            err.span_suggestion_with_applicability(
                                                expr.span,
                                                "to access tuple elements, use",
                                                format!("{}.{}", snip, i),
                                                Applicability::MachineApplicable);
                                            needs_note = false;
                                        }
                                    }
                                }
                                if needs_note {
                                    err.help("to access tuple elements, use tuple indexing \
                                              syntax (e.g., `tuple.0`)");
                                }
                            }
                            err.emit();
                            self.tcx.types.err
                        }
                    }
                }
            }
            hir::ExprKind::Yield(ref value) => {
                match self.yield_ty {
                    Some(ty) => {
                        self.check_expr_coercable_to_type(&value, ty);
                    }
                    None => {
                        struct_span_err!(self.tcx.sess, expr.span, E0627,
                                         "yield statement outside of generator literal").emit();
                    }
                }
                tcx.mk_unit()
            }
        }
    }

    // Finish resolving a path in a struct expression or pattern `S::A { .. }` if necessary.
    // The newly resolved definition is written into `type_dependent_defs`.
    fn finish_resolving_struct_path(&self,
                                    qpath: &hir::QPath,
                                    path_span: Span,
                                    node_id: ast::NodeId)
                                    -> (Def, Ty<'tcx>)
    {
        match *qpath {
            hir::QPath::Resolved(ref maybe_qself, ref path) => {
                let self_ty = maybe_qself.as_ref().map(|qself| self.to_ty(qself));
                let ty = AstConv::def_to_ty(self, self_ty, path, true);
                (path.def, ty)
            }
            hir::QPath::TypeRelative(ref qself, ref segment) => {
                let ty = self.to_ty(qself);

                let def = if let hir::TyKind::Path(hir::QPath::Resolved(_, ref path)) = qself.node {
                    path.def
                } else {
                    Def::Err
                };
                let (ty, def) = AstConv::associated_path_def_to_ty(self, node_id, path_span,
                                                                   ty, def, segment);

                // Write back the new resolution.
                let hir_id = self.tcx.hir().node_to_hir_id(node_id);
                self.tables.borrow_mut().type_dependent_defs_mut().insert(hir_id, def);

                (def, ty)
            }
        }
    }

    // Resolve associated value path into a base type and associated constant or method definition.
    // The newly resolved definition is written into `type_dependent_defs`.
    pub fn resolve_ty_and_def_ufcs<'b>(&self,
                                       qpath: &'b hir::QPath,
                                       node_id: ast::NodeId,
                                       span: Span)
                                       -> (Def, Option<Ty<'tcx>>, &'b [hir::PathSegment])
    {
        let (ty, item_segment) = match *qpath {
            hir::QPath::Resolved(ref opt_qself, ref path) => {
                return (path.def,
                        opt_qself.as_ref().map(|qself| self.to_ty(qself)),
                        &path.segments[..]);
            }
            hir::QPath::TypeRelative(ref qself, ref segment) => {
                (self.to_ty(qself), segment)
            }
        };
        let hir_id = self.tcx.hir().node_to_hir_id(node_id);
        if let Some(cached_def) = self.tables.borrow().type_dependent_defs().get(hir_id) {
            // Return directly on cache hit. This is useful to avoid doubly reporting
            // errors with default match binding modes. See #44614.
            return (*cached_def, Some(ty), slice::from_ref(&**item_segment))
        }
        let item_name = item_segment.ident;
        let def = match self.resolve_ufcs(span, item_name, ty, node_id) {
            Ok(def) => def,
            Err(error) => {
                let def = match error {
                    method::MethodError::PrivateMatch(def, _) => def,
                    _ => Def::Err,
                };
                if item_name.name != keywords::Invalid.name() {
                    self.report_method_error(span, ty, item_name, None, error, None);
                }
                def
            }
        };

        // Write back the new resolution.
        self.tables.borrow_mut().type_dependent_defs_mut().insert(hir_id, def);
        (def, Some(ty), slice::from_ref(&**item_segment))
    }

    pub fn check_decl_initializer(&self,
                                  local: &'gcx hir::Local,
                                  init: &'gcx hir::Expr) -> Ty<'tcx>
    {
        // FIXME(tschottdorf): contains_explicit_ref_binding() must be removed
        // for #42640 (default match binding modes).
        //
        // See #44848.
        let ref_bindings = local.pat.contains_explicit_ref_binding();

        let local_ty = self.local_ty(init.span, local.id).revealed_ty;
        if let Some(m) = ref_bindings {
            // Somewhat subtle: if we have a `ref` binding in the pattern,
            // we want to avoid introducing coercions for the RHS. This is
            // both because it helps preserve sanity and, in the case of
            // ref mut, for soundness (issue #23116). In particular, in
            // the latter case, we need to be clear that the type of the
            // referent for the reference that results is *equal to* the
            // type of the place it is referencing, and not some
            // supertype thereof.
            let init_ty = self.check_expr_with_needs(init, Needs::maybe_mut_place(m));
            self.demand_eqtype(init.span, local_ty, init_ty);
            init_ty
        } else {
            self.check_expr_coercable_to_type(init, local_ty)
        }
    }

    pub fn check_decl_local(&self, local: &'gcx hir::Local) {
        let t = self.local_ty(local.span, local.id).decl_ty;
        self.write_ty(local.hir_id, t);

        if let Some(ref init) = local.init {
            let init_ty = self.check_decl_initializer(local, &init);
            if init_ty.references_error() {
                self.write_ty(local.hir_id, init_ty);
            }
        }

        self.check_pat_walk(&local.pat, t,
                            ty::BindingMode::BindByValue(hir::Mutability::MutImmutable),
                            true);
        let pat_ty = self.node_ty(local.pat.hir_id);
        if pat_ty.references_error() {
            self.write_ty(local.hir_id, pat_ty);
        }
    }

    pub fn check_stmt(&self, stmt: &'gcx hir::Stmt) {
        // Don't do all the complex logic below for DeclItem.
        match stmt.node {
            hir::StmtKind::Decl(ref decl, _) => {
                if let hir::DeclKind::Item(_) = decl.node {
                    return
                }
            }
            hir::StmtKind::Expr(..) | hir::StmtKind::Semi(..) => {}
        }

        self.warn_if_unreachable(stmt.node.id(), stmt.span, "statement");

        // Hide the outer diverging and has_errors flags.
        let old_diverges = self.diverges.get();
        let old_has_errors = self.has_errors.get();
        self.diverges.set(Diverges::Maybe);
        self.has_errors.set(false);

        match stmt.node {
            hir::StmtKind::Decl(ref decl, _) => {
                match decl.node {
                    hir::DeclKind::Local(ref l) => {
                        self.check_decl_local(&l);
                    }
                    hir::DeclKind::Item(_) => {/* ignore for now */}
                }
            }
            hir::StmtKind::Expr(ref expr, _) => {
                // Check with expected type of ()
                self.check_expr_has_type_or_error(&expr, self.tcx.mk_unit());
            }
            hir::StmtKind::Semi(ref expr, _) => {
                self.check_expr(&expr);
            }
        }

        // Combine the diverging and has_error flags.
        self.diverges.set(self.diverges.get() | old_diverges);
        self.has_errors.set(self.has_errors.get() | old_has_errors);
    }

    pub fn check_block_no_value(&self, blk: &'gcx hir::Block)  {
        let unit = self.tcx.mk_unit();
        let ty = self.check_block_with_expected(blk, ExpectHasType(unit));

        // if the block produces a `!` value, that can always be
        // (effectively) coerced to unit.
        if !ty.is_never() {
            self.demand_suptype(blk.span, unit, ty);
        }
    }

    fn check_block_with_expected(&self,
                                 blk: &'gcx hir::Block,
                                 expected: Expectation<'tcx>) -> Ty<'tcx> {
        let prev = {
            let mut fcx_ps = self.ps.borrow_mut();
            let unsafety_state = fcx_ps.recurse(blk);
            replace(&mut *fcx_ps, unsafety_state)
        };

        // In some cases, blocks have just one exit, but other blocks
        // can be targeted by multiple breaks. This can happen both
        // with labeled blocks as well as when we desugar
        // a `try { ... }` expression.
        //
        // Example 1:
        //
        //    'a: { if true { break 'a Err(()); } Ok(()) }
        //
        // Here we would wind up with two coercions, one from
        // `Err(())` and the other from the tail expression
        // `Ok(())`. If the tail expression is omitted, that's a
        // "forced unit" -- unless the block diverges, in which
        // case we can ignore the tail expression (e.g., `'a: {
        // break 'a 22; }` would not force the type of the block
        // to be `()`).
        let tail_expr = blk.expr.as_ref();
        let coerce_to_ty = expected.coercion_target_type(self, blk.span);
        let coerce = if blk.targeted_by_break {
            CoerceMany::new(coerce_to_ty)
        } else {
            let tail_expr: &[P<hir::Expr>] = match tail_expr {
                Some(e) => slice::from_ref(e),
                None => &[],
            };
            CoerceMany::with_coercion_sites(coerce_to_ty, tail_expr)
        };

        let prev_diverges = self.diverges.get();
        let ctxt = BreakableCtxt {
            coerce: Some(coerce),
            may_break: false,
        };

        let (ctxt, ()) = self.with_breakable_ctxt(blk.id, ctxt, || {
            for s in &blk.stmts {
                self.check_stmt(s);
            }

            // check the tail expression **without** holding the
            // `enclosing_breakables` lock below.
            let tail_expr_ty = tail_expr.map(|t| self.check_expr_with_expectation(t, expected));

            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            let ctxt = enclosing_breakables.find_breakable(blk.id);
            let coerce = ctxt.coerce.as_mut().unwrap();
            if let Some(tail_expr_ty) = tail_expr_ty {
                let tail_expr = tail_expr.unwrap();
                let cause = self.cause(tail_expr.span,
                                       ObligationCauseCode::BlockTailExpression(blk.id));
                coerce.coerce(self,
                              &cause,
                              tail_expr,
                              tail_expr_ty);
            } else {
                // Subtle: if there is no explicit tail expression,
                // that is typically equivalent to a tail expression
                // of `()` -- except if the block diverges. In that
                // case, there is no value supplied from the tail
                // expression (assuming there are no other breaks,
                // this implies that the type of the block will be
                // `!`).
                //
                // #41425 -- label the implicit `()` as being the
                // "found type" here, rather than the "expected type".
                //
                // #44579 -- if the block was recovered during parsing,
                // the type would be nonsensical and it is not worth it
                // to perform the type check, so we avoid generating the
                // diagnostic output.
                if !self.diverges.get().always() && !blk.recovered {
                    coerce.coerce_forced_unit(self, &self.misc(blk.span), &mut |err| {
                        if let Some(expected_ty) = expected.only_has_type(self) {
                            self.consider_hint_about_removing_semicolon(blk,
                                                                        expected_ty,
                                                                        err);
                        }
                    }, false);
                }
            }
        });

        if ctxt.may_break {
            // If we can break from the block, then the block's exit is always reachable
            // (... as long as the entry is reachable) - regardless of the tail of the block.
            self.diverges.set(prev_diverges);
        }

        let mut ty = ctxt.coerce.unwrap().complete(self);

        if self.has_errors.get() || ty.references_error() {
            ty = self.tcx.types.err
        }

        self.write_ty(blk.hir_id, ty);

        *self.ps.borrow_mut() = prev;
        ty
    }

    /// Given a `NodeId`, return the `FnDecl` of the method it is enclosed by and whether a
    /// suggestion can be made, `None` otherwise.
    pub fn get_fn_decl(&self, blk_id: ast::NodeId) -> Option<(hir::FnDecl, bool)> {
        // Get enclosing Fn, if it is a function or a trait method, unless there's a `loop` or
        // `while` before reaching it, as block tail returns are not available in them.
        if let Some(fn_id) = self.tcx.hir().get_return_block(blk_id) {
            let parent = self.tcx.hir().get(fn_id);

            if let Node::Item(&hir::Item {
                name, node: hir::ItemKind::Fn(ref decl, ..), ..
            }) = parent {
                decl.clone().and_then(|decl| {
                    // This is less than ideal, it will not suggest a return type span on any
                    // method called `main`, regardless of whether it is actually the entry point,
                    // but it will still present it as the reason for the expected type.
                    Some((decl, name != Symbol::intern("main")))
                })
            } else if let Node::TraitItem(&hir::TraitItem {
                node: hir::TraitItemKind::Method(hir::MethodSig {
                    ref decl, ..
                }, ..), ..
            }) = parent {
                decl.clone().and_then(|decl| {
                    Some((decl, true))
                })
            } else if let Node::ImplItem(&hir::ImplItem {
                node: hir::ImplItemKind::Method(hir::MethodSig {
                    ref decl, ..
                }, ..), ..
            }) = parent {
                decl.clone().and_then(|decl| {
                    Some((decl, false))
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// On implicit return expressions with mismatched types, provide the following suggestions:
    ///
    ///  - Point out the method's return type as the reason for the expected type
    ///  - Possible missing semicolon
    ///  - Possible missing return type if the return type is the default, and not `fn main()`
    pub fn suggest_mismatched_types_on_tail(&self,
                                            err: &mut DiagnosticBuilder<'tcx>,
                                            expression: &'gcx hir::Expr,
                                            expected: Ty<'tcx>,
                                            found: Ty<'tcx>,
                                            cause_span: Span,
                                            blk_id: ast::NodeId) {
        self.suggest_missing_semicolon(err, expression, expected, cause_span);
        if let Some((fn_decl, can_suggest)) = self.get_fn_decl(blk_id) {
            self.suggest_missing_return_type(err, &fn_decl, expected, found, can_suggest);
        }
        self.suggest_ref_or_into(err, expression, expected, found);
    }

    pub fn suggest_ref_or_into(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        expr: &hir::Expr,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if let Some((sp, msg, suggestion)) = self.check_ref(expr, found, expected) {
            err.span_suggestion_with_applicability(
                sp,
                msg,
                suggestion,
                Applicability::MachineApplicable,
            );
        } else if !self.check_for_cast(err, expr, found, expected) {
            let methods = self.get_conversion_methods(expr.span, expected, found);
            if let Ok(expr_text) = self.sess().source_map().span_to_snippet(expr.span) {
                let mut suggestions = iter::repeat(&expr_text).zip(methods.iter())
                    .filter_map(|(receiver, method)| {
                        let method_call = format!(".{}()", method.ident);
                        if receiver.ends_with(&method_call) {
                            None  // do not suggest code that is already there (#53348)
                        } else {
                            let method_call_list = [".to_vec()", ".to_string()"];
                            if receiver.ends_with(".clone()")
                                    && method_call_list.contains(&method_call.as_str()) {
                                let max_len = receiver.rfind(".").unwrap();
                                Some(format!("{}{}", &receiver[..max_len], method_call))
                            }
                            else {
                                Some(format!("{}{}", receiver, method_call))
                            }
                        }
                    }).peekable();
                if suggestions.peek().is_some() {
                    err.span_suggestions_with_applicability(
                        expr.span,
                        "try using a conversion method",
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    /// A common error is to forget to add a semicolon at the end of a block:
    ///
    /// ```
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return expression in a block would make sense on its own as a
    /// statement and the return type has been left as default or has been specified as `()`. If so,
    /// it suggests adding a semicolon.
    fn suggest_missing_semicolon(&self,
                                 err: &mut DiagnosticBuilder<'tcx>,
                                 expression: &'gcx hir::Expr,
                                 expected: Ty<'tcx>,
                                 cause_span: Span) {
        if expected.is_unit() {
            // `BlockTailExpression` only relevant if the tail expr would be
            // useful on its own.
            match expression.node {
                hir::ExprKind::Call(..) |
                hir::ExprKind::MethodCall(..) |
                hir::ExprKind::If(..) |
                hir::ExprKind::While(..) |
                hir::ExprKind::Loop(..) |
                hir::ExprKind::Match(..) |
                hir::ExprKind::Block(..) => {
                    let sp = self.tcx.sess.source_map().next_point(cause_span);
                    err.span_suggestion_with_applicability(
                        sp,
                        "try adding a semicolon",
                        ";".to_string(),
                        Applicability::MachineApplicable);
                }
                _ => (),
            }
        }
    }

    /// A possible error is to forget to add a return type that is needed:
    ///
    /// ```
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return type is left as default, the method is not part of an
    /// `impl` block and that it isn't the `main` method. If so, it suggests setting the return
    /// type.
    fn suggest_missing_return_type(&self,
                                   err: &mut DiagnosticBuilder<'tcx>,
                                   fn_decl: &hir::FnDecl,
                                   expected: Ty<'tcx>,
                                   found: Ty<'tcx>,
                                   can_suggest: bool) {
        // Only suggest changing the return type for methods that
        // haven't set a return type at all (and aren't `fn main()` or an impl).
        match (&fn_decl.output, found.is_suggestable(), can_suggest, expected.is_unit()) {
            (&hir::FunctionRetTy::DefaultReturn(span), true, true, true) => {
                err.span_suggestion_with_applicability(
                    span,
                    "try adding a return type",
                    format!("-> {} ", self.resolve_type_vars_with_obligations(found)),
                    Applicability::MachineApplicable);
            }
            (&hir::FunctionRetTy::DefaultReturn(span), false, true, true) => {
                err.span_label(span, "possibly return type missing here?");
            }
            (&hir::FunctionRetTy::DefaultReturn(span), _, false, true) => {
                // `fn main()` must return `()`, do not suggest changing return type
                err.span_label(span, "expected `()` because of default return type");
            }
            // expectation was caused by something else, not the default return
            (&hir::FunctionRetTy::DefaultReturn(_), _, _, false) => {}
            (&hir::FunctionRetTy::Return(ref ty), _, _, _) => {
                // Only point to return type if the expected type is the return type, as if they
                // are not, the expectation must have been caused by something else.
                debug!("suggest_missing_return_type: return type {:?} node {:?}", ty, ty.node);
                let sp = ty.span;
                let ty = AstConv::ast_ty_to_ty(self, ty);
                debug!("suggest_missing_return_type: return type sty {:?}", ty.sty);
                debug!("suggest_missing_return_type: expected type sty {:?}", ty.sty);
                if ty.sty == expected.sty {
                    err.span_label(sp, format!("expected `{}` because of return type",
                                               expected));
                }
            }
        }
    }


    /// A common error is to add an extra semicolon:
    ///
    /// ```
    /// fn foo() -> usize {
    ///     22;
    /// }
    /// ```
    ///
    /// This routine checks if the final statement in a block is an
    /// expression with an explicit semicolon whose type is compatible
    /// with `expected_ty`. If so, it suggests removing the semicolon.
    fn consider_hint_about_removing_semicolon(&self,
                                              blk: &'gcx hir::Block,
                                              expected_ty: Ty<'tcx>,
                                              err: &mut DiagnosticBuilder) {
        // Be helpful when the user wrote `{... expr;}` and
        // taking the `;` off is enough to fix the error.
        let last_stmt = match blk.stmts.last() {
            Some(s) => s,
            None => return,
        };
        let last_expr = match last_stmt.node {
            hir::StmtKind::Semi(ref e, _) => e,
            _ => return,
        };
        let last_expr_ty = self.node_ty(last_expr.hir_id);
        if self.can_sub(self.param_env, last_expr_ty, expected_ty).is_err() {
            return;
        }
        let original_span = original_sp(last_stmt.span, blk.span);
        let span_semi = original_span.with_lo(original_span.hi() - BytePos(1));
        err.span_suggestion_with_applicability(
            span_semi,
            "consider removing this semicolon",
            String::new(),
            Applicability::MachineApplicable);
    }

    fn def_ids_for_path_segments(&self,
                                 segments: &[hir::PathSegment],
                                 def: Def)
                                 -> Vec<PathSeg> {
        // We need to extract the type parameters supplied by the user in
        // the path `path`. Due to the current setup, this is a bit of a
        // tricky-process; the problem is that resolve only tells us the
        // end-point of the path resolution, and not the intermediate steps.
        // Luckily, we can (at least for now) deduce the intermediate steps
        // just from the end-point.
        //
        // There are basically four cases to consider:
        //
        // 1. Reference to a constructor of enum variant or struct:
        //
        //        struct Foo<T>(...)
        //        enum E<T> { Foo(...) }
        //
        //    In these cases, the parameters are declared in the type
        //    space.
        //
        // 2. Reference to a fn item or a free constant:
        //
        //        fn foo<T>() { }
        //
        //    In this case, the path will again always have the form
        //    `a::b::foo::<T>` where only the final segment should have
        //    type parameters. However, in this case, those parameters are
        //    declared on a value, and hence are in the `FnSpace`.
        //
        // 3. Reference to a method or an associated constant:
        //
        //        impl<A> SomeStruct<A> {
        //            fn foo<B>(...)
        //        }
        //
        //    Here we can have a path like
        //    `a::b::SomeStruct::<A>::foo::<B>`, in which case parameters
        //    may appear in two places. The penultimate segment,
        //    `SomeStruct::<A>`, contains parameters in TypeSpace, and the
        //    final segment, `foo::<B>` contains parameters in fn space.
        //
        // 4. Reference to a local variable
        //
        //    Local variables can't have any type parameters.
        //
        // The first step then is to categorize the segments appropriately.

        assert!(!segments.is_empty());
        let last = segments.len() - 1;

        let mut path_segs = vec![];

        match def {
            // Case 1. Reference to a struct/variant constructor.
            Def::StructCtor(def_id, ..) |
            Def::VariantCtor(def_id, ..) |
            Def::SelfCtor(.., def_id) => {
                // Everything but the final segment should have no
                // parameters at all.
                let generics = self.tcx.generics_of(def_id);
                // Variant and struct constructors use the
                // generics of their parent type definition.
                let generics_def_id = generics.parent.unwrap_or(def_id);
                path_segs.push(PathSeg(generics_def_id, last));
            }

            // Case 2. Reference to a top-level value.
            Def::Fn(def_id) |
            Def::Const(def_id) |
            Def::Static(def_id, _) => {
                path_segs.push(PathSeg(def_id, last));
            }

            // Case 3. Reference to a method or associated const.
            Def::Method(def_id) |
            Def::AssociatedConst(def_id) => {
                if segments.len() >= 2 {
                    let generics = self.tcx.generics_of(def_id);
                    path_segs.push(PathSeg(generics.parent.unwrap(), last - 1));
                }
                path_segs.push(PathSeg(def_id, last));
            }

            // Case 4. Local variable, no generics.
            Def::Local(..) | Def::Upvar(..) => {}

            _ => bug!("unexpected definition: {:?}", def),
        }

        debug!("path_segs = {:?}", path_segs);

        path_segs
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    pub fn instantiate_value_path(&self,
                                  segments: &[hir::PathSegment],
                                  self_ty: Option<Ty<'tcx>>,
                                  def: Def,
                                  span: Span,
                                  node_id: ast::NodeId)
                                  -> (Ty<'tcx>, Def) {
        debug!(
            "instantiate_value_path(segments={:?}, self_ty={:?}, def={:?}, node_id={})",
            segments,
            self_ty,
            def,
            node_id,
        );

        let path_segs = self.def_ids_for_path_segments(segments, def);

        let mut user_self_ty = None;
        match def {
            Def::Method(def_id) |
            Def::AssociatedConst(def_id) => {
                let container = self.tcx.associated_item(def_id).container;
                match container {
                    ty::TraitContainer(trait_did) => {
                        callee::check_legal_trait_for_method_call(self.tcx, span, trait_did)
                    }
                    ty::ImplContainer(impl_def_id) => {
                        if segments.len() == 1 {
                            // `<T>::assoc` will end up here, and so
                            // can `T::assoc`. It this came from an
                            // inherent impl, we need to record the
                            // `T` for posterity (see `UserSelfTy` for
                            // details).
                            let self_ty = self_ty.expect("UFCS sugared assoc missing Self");
                            user_self_ty = Some(UserSelfTy {
                                impl_def_id,
                                self_ty,
                            });
                        }
                    }
                }
            }
            _ => {}
        }

        // Now that we have categorized what space the parameters for each
        // segment belong to, let's sort out the parameters that the user
        // provided (if any) into their appropriate spaces. We'll also report
        // errors if type parameters are provided in an inappropriate place.

        let generic_segs = path_segs.iter().map(|PathSeg(_, index)| index)
            .collect::<FxHashSet<_>>();
        AstConv::prohibit_generics(self, segments.iter().enumerate().filter_map(|(index, seg)| {
            if !generic_segs.contains(&index) {
                Some(seg)
            } else {
                None
            }
        }));

        match def {
            Def::Local(nid) | Def::Upvar(nid, ..) => {
                let ty = self.local_ty(span, nid).decl_ty;
                let ty = self.normalize_associated_types_in(span, &ty);
                self.write_ty(self.tcx.hir().node_to_hir_id(node_id), ty);
                return (ty, def);
            }
            _ => {}
        }

        // Now we have to compare the types that the user *actually*
        // provided against the types that were *expected*. If the user
        // did not provide any types, then we want to substitute inference
        // variables. If the user provided some types, we may still need
        // to add defaults. If the user provided *too many* types, that's
        // a problem.

        let mut infer_args_for_err = FxHashSet::default();
        for &PathSeg(def_id, index) in &path_segs {
            let seg = &segments[index];
            let generics = self.tcx.generics_of(def_id);
            // Argument-position `impl Trait` is treated as a normal generic
            // parameter internally, but we don't allow users to specify the
            // parameter's value explicitly, so we have to do some error-
            // checking here.
            let suppress_errors = AstConv::check_generic_arg_count_for_call(
                self.tcx,
                span,
                &generics,
                &seg,
                false, // `is_method_call`
            );
            if suppress_errors {
                infer_args_for_err.insert(index);
                self.set_tainted_by_errors(); // See issue #53251.
            }
        }

        let has_self = path_segs.last().map(|PathSeg(def_id, _)| {
            self.tcx.generics_of(*def_id).has_self
        }).unwrap_or(false);

        let mut new_def = def;
        let (def_id, ty) = match def {
            Def::SelfCtor(impl_def_id) => {
                let ty = self.impl_self_ty(span, impl_def_id).ty;
                let adt_def = ty.ty_adt_def();

                match adt_def {
                    Some(adt_def) if adt_def.has_ctor() => {
                        let variant = adt_def.non_enum_variant();
                        new_def = Def::StructCtor(variant.did, variant.ctor_kind);
                        (variant.did, self.tcx.type_of(variant.did))
                    }
                    _ => {
                        let mut err = self.tcx.sess.struct_span_err(span,
                            "the `Self` constructor can only be used with tuple or unit structs");
                        if let Some(adt_def) = adt_def {
                            match adt_def.adt_kind() {
                                AdtKind::Enum => {
                                    err.note("did you mean to use one of the enum's variants?");
                                },
                                AdtKind::Struct |
                                AdtKind::Union => {
                                    err.span_label(
                                        span,
                                        format!("did you mean `Self {{ /* fields */ }}`?"),
                                    );
                                }
                            }
                        }
                        err.emit();

                        (impl_def_id, self.tcx.types.err)
                    }
                }
            }
            _ => {
                let def_id = def.def_id();

                // The things we are substituting into the type should not contain
                // escaping late-bound regions, and nor should the base type scheme.
                let ty = self.tcx.type_of(def_id);
                (def_id, ty)
            }
        };

        let substs = AstConv::create_substs_for_generic_args(
            self.tcx,
            def_id,
            &[][..],
            has_self,
            self_ty,
            // Provide the generic args, and whether types should be inferred.
            |def_id| {
                if let Some(&PathSeg(_, index)) = path_segs.iter().find(|&PathSeg(did, _)| {
                    *did == def_id
                }) {
                    // If we've encountered an `impl Trait`-related error, we're just
                    // going to infer the arguments for better error messages.
                    if !infer_args_for_err.contains(&index) {
                        // Check whether the user has provided generic arguments.
                        if let Some(ref data) = segments[index].args {
                            return (Some(data), segments[index].infer_types);
                        }
                    }
                    return (None, segments[index].infer_types);
                }

                (None, true)
            },
            // Provide substitutions for parameters for which (valid) arguments have been provided.
            |param, arg| {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        AstConv::ast_region_to_region(self, lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.to_ty(ty).into()
                    }
                    _ => unreachable!(),
                }
            },
            // Provide substitutions for parameters for which arguments are inferred.
            |substs, param, infer_types| {
                match param.kind {
                    GenericParamDefKind::Lifetime => {
                        self.re_infer(span, Some(param)).unwrap().into()
                    }
                    GenericParamDefKind::Type { has_default, .. } => {
                        if !infer_types && has_default {
                            // If we have a default, then we it doesn't matter that we're not
                            // inferring the type arguments: we provide the default where any
                            // is missing.
                            let default = self.tcx.type_of(param.def_id);
                            self.normalize_ty(
                                span,
                                default.subst_spanned(self.tcx, substs.unwrap(), Some(span))
                            ).into()
                        } else {
                            // If no type arguments were provided, we have to infer them.
                            // This case also occurs as a result of some malformed input, e.g.
                            // a lifetime argument being given instead of a type parameter.
                            // Using inference instead of `Error` gives better error messages.
                            self.var_for_def(span, param)
                        }
                    }
                }
            },
        );
        assert!(!substs.has_escaping_bound_vars());
        assert!(!ty.has_escaping_bound_vars());

        // First, store the "user substs" for later.
        let hir_id = self.tcx.hir().node_to_hir_id(node_id);
        self.write_user_substs_from_substs(hir_id, substs, user_self_ty);

        // Add all the obligations that are required, substituting and
        // normalized appropriately.
        let bounds = self.instantiate_bounds(span, def_id, &substs);
        self.add_obligations_for_parameters(
            traits::ObligationCause::new(span, self.body_id, traits::ItemObligation(def_id)),
            &bounds);

        // Substitute the values for the type parameters into the type of
        // the referenced item.
        let ty_substituted = self.instantiate_type_scheme(span, &substs, &ty);

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter, instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the substituted impl type.
            let ty = self.tcx.type_of(impl_def_id);

            let impl_ty = self.instantiate_type_scheme(span, &substs, &ty);
            match self.at(&self.misc(span), self.param_env).sup(impl_ty, self_ty) {
                Ok(ok) => self.register_infer_ok_obligations(ok),
                Err(_) => {
                    span_bug!(span,
                        "instantiate_value_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                        self_ty,
                        impl_ty);
                }
            }
        }

        self.check_rustc_args_require_const(def_id, node_id, span);

        debug!("instantiate_value_path: type of {:?} is {:?}",
               node_id,
               ty_substituted);
        self.write_substs(hir_id, substs);

        (ty_substituted, new_def)
    }

    fn check_rustc_args_require_const(&self,
                                      def_id: DefId,
                                      node_id: ast::NodeId,
                                      span: Span) {
        // We're only interested in functions tagged with
        // #[rustc_args_required_const], so ignore anything that's not.
        if !self.tcx.has_attr(def_id, "rustc_args_required_const") {
            return
        }

        // If our calling expression is indeed the function itself, we're good!
        // If not, generate an error that this can only be called directly.
        if let Node::Expr(expr) = self.tcx.hir().get(self.tcx.hir().get_parent_node(node_id)) {
            if let hir::ExprKind::Call(ref callee, ..) = expr.node {
                if callee.id == node_id {
                    return
                }
            }
        }

        self.tcx.sess.span_err(span, "this function can only be invoked \
                                      directly, not through a function pointer");
    }

    // Resolves `typ` by a single level if `typ` is a type variable.
    // If no resolution is possible, then an error is reported.
    // Numeric inference variables may be left unresolved.
    pub fn structurally_resolved_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.resolve_type_vars_with_obligations(ty);
        if !ty.is_ty_var() {
            ty
        } else {
            if !self.is_tainted_by_errors() {
                self.need_type_info_err((**self).body_id, sp, ty)
                    .note("type must be known at this point")
                    .emit();
            }
            self.demand_suptype(sp, self.tcx.types.err, ty);
            self.tcx.types.err
        }
    }

    fn with_breakable_ctxt<F: FnOnce() -> R, R>(&self, id: ast::NodeId,
                                                ctxt: BreakableCtxt<'gcx, 'tcx>, f: F)
                                                -> (BreakableCtxt<'gcx, 'tcx>, R) {
        let index;
        {
            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            index = enclosing_breakables.stack.len();
            enclosing_breakables.by_id.insert(id, index);
            enclosing_breakables.stack.push(ctxt);
        }
        let result = f();
        let ctxt = {
            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            debug_assert!(enclosing_breakables.stack.len() == index + 1);
            enclosing_breakables.by_id.remove(&id).expect("missing breakable context");
            enclosing_breakables.stack.pop().expect("missing breakable context")
        };
        (ctxt, result)
    }
}

pub fn check_bounds_are_used<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       generics: &ty::Generics,
                                       ty: Ty<'tcx>) {
    let own_counts = generics.own_counts();
    debug!("check_bounds_are_used(n_tps={}, ty={:?})", own_counts.types, ty);

    if own_counts.types == 0 {
        return;
    }
    // Make a vector of booleans initially false, set to true when used.
    let mut types_used = vec![false; own_counts.types];

    for leaf_ty in ty.walk() {
        if let ty::Param(ty::ParamTy { idx, .. }) = leaf_ty.sty {
            debug!("Found use of ty param num {}", idx);
            types_used[idx as usize - own_counts.lifetimes] = true;
        } else if let ty::Error = leaf_ty.sty {
            // If there is already another error, do not emit
            // an error for not using a type Parameter.
            assert!(tcx.sess.err_count() > 0);
            return;
        }
    }

    let types = generics.params.iter().filter(|param| match param.kind {
        ty::GenericParamDefKind::Type { .. } => true,
        _ => false,
    });
    for (&used, param) in types_used.iter().zip(types) {
        if !used {
            let id = tcx.hir().as_local_node_id(param.def_id).unwrap();
            let span = tcx.hir().span(id);
            struct_span_err!(tcx.sess, span, E0091, "type parameter `{}` is unused", param.name)
                .span_label(span, "unused type parameter")
                .emit();
        }
    }
}

fn fatally_break_rust(sess: &Session) {
    let handler = sess.diagnostic();
    handler.span_bug_no_panic(
        MultiSpan::new(),
        "It looks like you're trying to break rust; would you like some ICE?",
    );
    handler.note_without_error("the compiler expectedly panicked. this is a feature.");
    handler.note_without_error(
        "we would appreciate a joke overview: \
        https://github.com/rust-lang/rust/issues/43162#issuecomment-320764675"
    );
    handler.note_without_error(&format!("rustc {} running on {}",
        option_env!("CFG_VERSION").unwrap_or("unknown_version"),
        ::session::config::host_triple(),
    ));
}

fn potentially_plural_count(count: usize, word: &str) -> String {
    format!("{} {}{}", count, word, if count == 1 { "" } else { "s" })
}
