// ignore-tidy-filelength

/*!

# typeck: check phase

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

pub mod _match;
mod autoderef;
mod callee;
mod cast;
mod closure;
pub mod coercion;
mod compare_method;
pub mod demand;
pub mod dropck;
mod expr;
mod generator_interior;
pub mod intrinsic;
pub mod method;
mod op;
mod pat;
mod regionck;
mod upvar;
mod wfcheck;
pub mod writeback;

use crate::astconv::{AstConv, GenericArgCountMismatch, PathSeg};
use crate::middle::lang_items;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::hir::map::Map;
use rustc::middle::region;
use rustc::mir::interpret::ConstValue;
use rustc::session::parse::feature_err;
use rustc::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability, PointerCast,
};
use rustc::ty::fold::{TypeFoldable, TypeFolder};
use rustc::ty::layout::VariantIdx;
use rustc::ty::query::Providers;
use rustc::ty::subst::{GenericArgKind, InternalSubsts, Subst, SubstsRef, UserSelfTy, UserSubsts};
use rustc::ty::util::{Discr, IntTypeExt, Representability};
use rustc::ty::{
    self, AdtKind, CanonicalUserType, Const, GenericParamDefKind, RegionKind, ToPolyTraitRef,
    ToPredicate, Ty, TyCtxt, UserType, WithConstness,
};
use rustc_ast::ast;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_attr as attr;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefIdSet, LOCAL_CRATE};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::{ExprKind, GenericArg, HirIdMap, Item, ItemKind, Node, PatKind, QPath};
use rustc_index::vec::Idx;
use rustc_infer::infer::canonical::{Canonical, OriginalQueryValues, QueryResponse};
use rustc_infer::infer::error_reporting::TypeAnnotationNeeded::E0282;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_infer::infer::{self, InferCtxt, InferOk, InferResult, TyCtxtInferExt};
use rustc_span::hygiene::DesugaringKind;
use rustc_span::source_map::{original_sp, DUMMY_SP};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{self, BytePos, MultiSpan, Span};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::opaque_types::{InferCtxtExt as _, OpaqueTypeDecl};
use rustc_trait_selection::traits::error_reporting::recursive_type_with_infinite_size_error;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCauseCode, TraitEngine, TraitEngineExt,
};

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::cmp;
use std::collections::hash_map::Entry;
use std::iter;
use std::mem::replace;
use std::ops::{self, Deref};
use std::slice;

use crate::lint;
use crate::require_c_abi_if_c_variadic;
use crate::session::config::EntryFnType;
use crate::session::Session;
use crate::util::common::{indenter, ErrorReported};
use crate::TypeAndSubsts;

use self::autoderef::Autoderef;
use self::callee::DeferredCallResolution;
use self::coercion::{CoerceMany, DynamicCoerceMany};
use self::compare_method::{compare_const_impl, compare_impl_method, compare_ty_impl};
use self::method::{MethodCallee, SelfSource};
pub use self::Expectation::*;
use self::TupleArgumentsFlag::*;

#[macro_export]
macro_rules! type_error_struct {
    ($session:expr, $span:expr, $typ:expr, $code:ident, $($message:tt)*) => ({
        if $typ.references_error() {
            $session.diagnostic().struct_dummy()
        } else {
            rustc_errors::struct_span_err!($session, $span, $code, $($message)*)
        }
    })
}

/// The type of a local binding, including the revealed type for anon types.
#[derive(Copy, Clone, Debug)]
pub struct LocalTy<'tcx> {
    decl_ty: Ty<'tcx>,
    revealed_ty: Ty<'tcx>,
}

/// A wrapper for `InferCtxt`'s `in_progress_tables` field.
#[derive(Copy, Clone)]
struct MaybeInProgressTables<'a, 'tcx> {
    maybe_tables: Option<&'a RefCell<ty::TypeckTables<'tcx>>>,
}

impl<'a, 'tcx> MaybeInProgressTables<'a, 'tcx> {
    fn borrow(self) -> Ref<'a, ty::TypeckTables<'tcx>> {
        match self.maybe_tables {
            Some(tables) => tables.borrow(),
            None => bug!("MaybeInProgressTables: inh/fcx.tables.borrow() with no tables"),
        }
    }

    fn borrow_mut(self) -> RefMut<'a, ty::TypeckTables<'tcx>> {
        match self.maybe_tables {
            Some(tables) => tables.borrow_mut(),
            None => bug!("MaybeInProgressTables: inh/fcx.tables.borrow_mut() with no tables"),
        }
    }
}

/// Closures defined within the function. For example:
///
///     fn foo() {
///         bar(move|| { ... })
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited<'a, 'tcx> {
    infcx: InferCtxt<'a, 'tcx>,

    tables: MaybeInProgressTables<'a, 'tcx>,

    locals: RefCell<HirIdMap<LocalTy<'tcx>>>,

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
    deferred_call_resolutions: RefCell<DefIdMap<Vec<DeferredCallResolution<'tcx>>>>,

    deferred_cast_checks: RefCell<Vec<cast::CastCheck<'tcx>>>,

    deferred_generator_interiors: RefCell<Vec<(hir::BodyId, Ty<'tcx>, hir::GeneratorKind)>>,

    // Opaque types found in explicit return types and their
    // associated fresh inference variable. Writeback resolves these
    // variables to get the concrete type, which can be used to
    // 'de-opaque' OpaqueTypeDecl, after typeck is done with all functions.
    opaque_types: RefCell<DefIdMap<OpaqueTypeDecl<'tcx>>>,

    /// A map from inference variables created from opaque
    /// type instantiations (`ty::Infer`) to the actual opaque
    /// type (`ty::Opaque`). Used during fallback to map unconstrained
    /// opaque type inference variables to their corresponding
    /// opaque type.
    opaque_types_vars: RefCell<FxHashMap<Ty<'tcx>, Ty<'tcx>>>,

    /// Each type parameter has an implicit region bound that
    /// indicates it must outlive at least the function body (the user
    /// may specify stronger requirements). This field indicates the
    /// region of the callee. If it is `None`, then the parameter
    /// environment is for an item or something where the "callee" is
    /// not clear.
    implicit_region_bound: Option<ty::Region<'tcx>>,

    body_id: Option<hir::BodyId>,
}

impl<'a, 'tcx> Deref for Inherited<'a, 'tcx> {
    type Target = InferCtxt<'a, 'tcx>;
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

    /// This expression should have the type given (or some subtype).
    ExpectHasType(Ty<'tcx>),

    /// This expression will be cast to the `Ty`.
    ExpectCastableToType(Ty<'tcx>),

    /// This rvalue expression will be wrapped in `&` or `Box` and coerced
    /// to `&Ty` or `Box<Ty>`, respectively. `Ty` is `[A]` or `Trait`.
    ExpectRvalueLikeUnsized(Ty<'tcx>),
}

impl<'a, 'tcx> Expectation<'tcx> {
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
    fn adjust_for_branches(&self, fcx: &FnCtxt<'a, 'tcx>) -> Expectation<'tcx> {
        match *self {
            ExpectHasType(ety) => {
                let ety = fcx.shallow_resolve(ety);
                if !ety.is_ty_var() { ExpectHasType(ety) } else { NoExpectation }
            }
            ExpectRvalueLikeUnsized(ety) => ExpectRvalueLikeUnsized(ety),
            _ => NoExpectation,
        }
    }

    /// Provides an expectation for an rvalue expression given an *optional*
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
    /// See the test case `test/ui/coerce-expect-unsized.rs` and #20169
    /// for examples of where this comes up,.
    fn rvalue_hint(fcx: &FnCtxt<'a, 'tcx>, ty: Ty<'tcx>) -> Expectation<'tcx> {
        match fcx.tcx.struct_tail_without_normalization(ty).kind {
            ty::Slice(_) | ty::Str | ty::Dynamic(..) => ExpectRvalueLikeUnsized(ty),
            _ => ExpectHasType(ty),
        }
    }

    // Resolves `expected` by a single level if it is a variable. If
    // there is no expected type or resolution is not possible (e.g.,
    // no constraints yet present), just returns `None`.
    fn resolve(self, fcx: &FnCtxt<'a, 'tcx>) -> Expectation<'tcx> {
        match self {
            NoExpectation => NoExpectation,
            ExpectCastableToType(t) => ExpectCastableToType(fcx.resolve_vars_if_possible(&t)),
            ExpectHasType(t) => ExpectHasType(fcx.resolve_vars_if_possible(&t)),
            ExpectRvalueLikeUnsized(t) => ExpectRvalueLikeUnsized(fcx.resolve_vars_if_possible(&t)),
        }
    }

    fn to_option(self, fcx: &FnCtxt<'a, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            NoExpectation => None,
            ExpectCastableToType(ty) | ExpectHasType(ty) | ExpectRvalueLikeUnsized(ty) => Some(ty),
        }
    }

    /// It sometimes happens that we want to turn an expectation into
    /// a **hard constraint** (i.e., something that must be satisfied
    /// for the program to type-check). `only_has_type` will return
    /// such a constraint, if it exists.
    fn only_has_type(self, fcx: &FnCtxt<'a, 'tcx>) -> Option<Ty<'tcx>> {
        match self.resolve(fcx) {
            ExpectHasType(ty) => Some(ty),
            NoExpectation | ExpectCastableToType(_) | ExpectRvalueLikeUnsized(_) => None,
        }
    }

    /// Like `only_has_type`, but instead of returning `None` if no
    /// hard constraint exists, creates a fresh type variable.
    fn coercion_target_type(self, fcx: &FnCtxt<'a, 'tcx>, span: Span) -> Ty<'tcx> {
        self.only_has_type(fcx).unwrap_or_else(|| {
            fcx.next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::MiscVariable, span })
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Needs {
    MutPlace,
    None,
}

impl Needs {
    fn maybe_mut_place(m: hir::Mutability) -> Self {
        match m {
            hir::Mutability::Mut => Needs::MutPlace,
            hir::Mutability::Not => Needs::None,
        }
    }
}

#[derive(Copy, Clone)]
pub struct UnsafetyState {
    pub def: hir::HirId,
    pub unsafety: hir::Unsafety,
    pub unsafe_push_count: u32,
    from_fn: bool,
}

impl UnsafetyState {
    pub fn function(unsafety: hir::Unsafety, def: hir::HirId) -> UnsafetyState {
        UnsafetyState { def, unsafety, unsafe_push_count: 0, from_fn: true }
    }

    pub fn recurse(&mut self, blk: &hir::Block<'_>) -> UnsafetyState {
        use hir::BlockCheckMode;
        match self.unsafety {
            // If this unsafe, then if the outer function was already marked as
            // unsafe we shouldn't attribute the unsafe'ness to the block. This
            // way the block can be warned about instead of ignoring this
            // extraneous block (functions are never warned about).
            hir::Unsafety::Unsafe if self.from_fn => *self,

            unsafety => {
                let (unsafety, def, count) = match blk.rules {
                    BlockCheckMode::PushUnsafeBlock(..) => {
                        (unsafety, blk.hir_id, self.unsafe_push_count.checked_add(1).unwrap())
                    }
                    BlockCheckMode::PopUnsafeBlock(..) => {
                        (unsafety, blk.hir_id, self.unsafe_push_count.checked_sub(1).unwrap())
                    }
                    BlockCheckMode::UnsafeBlock(..) => {
                        (hir::Unsafety::Unsafe, blk.hir_id, self.unsafe_push_count)
                    }
                    BlockCheckMode::DefaultBlock => (unsafety, self.def, self.unsafe_push_count),
                };
                UnsafetyState { def, unsafety, unsafe_push_count: count, from_fn: false }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum PlaceOp {
    Deref,
    Index,
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
    Always {
        /// The `Span` points to the expression
        /// that caused us to diverge
        /// (e.g. `return`, `break`, etc).
        span: Span,
        /// In some cases (e.g. a `match` expression
        /// where all arms diverge), we may be
        /// able to provide a more informative
        /// message to the user.
        /// If this is `None`, a default message
        /// will be generated, which is suitable
        /// for most cases.
        custom_note: Option<&'static str>,
    },

    /// Same as `Always` but with a reachability
    /// warning already emitted.
    WarnedAlways,
}

// Convenience impls for combining `Diverges`.

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
    /// Creates a `Diverges::Always` with the provided `span` and the default note message.
    fn always(span: Span) -> Diverges {
        Diverges::Always { span, custom_note: None }
    }

    fn is_always(self) -> bool {
        // Enum comparison ignores the
        // contents of fields, so we just
        // fill them in with garbage here.
        self >= Diverges::Always { span: DUMMY_SP, custom_note: None }
    }
}

pub struct BreakableCtxt<'tcx> {
    may_break: bool,

    // this is `null` for loops where break with a value is illegal,
    // such as `while`, `for`, and `while let`
    coerce: Option<DynamicCoerceMany<'tcx>>,
}

pub struct EnclosingBreakables<'tcx> {
    stack: Vec<BreakableCtxt<'tcx>>,
    by_id: HirIdMap<usize>,
}

impl<'tcx> EnclosingBreakables<'tcx> {
    fn find_breakable(&mut self, target_id: hir::HirId) -> &mut BreakableCtxt<'tcx> {
        self.opt_find_breakable(target_id).unwrap_or_else(|| {
            bug!("could not find enclosing breakable with id {}", target_id);
        })
    }

    fn opt_find_breakable(&mut self, target_id: hir::HirId) -> Option<&mut BreakableCtxt<'tcx>> {
        match self.by_id.get(&target_id) {
            Some(ix) => Some(&mut self.stack[*ix]),
            None => None,
        }
    }
}

pub struct FnCtxt<'a, 'tcx> {
    body_id: hir::HirId,

    /// The parameter environment used for proving trait obligations
    /// in this function. This can change when we descend into
    /// closures (as they bring new things into scope), hence it is
    /// not part of `Inherited` (as of the time of this writing,
    /// closures do not yet change the environment, but they will
    /// eventually).
    param_env: ty::ParamEnv<'tcx>,

    /// Number of errors that had been reported when we started
    /// checking this function. On exit, if we find that *more* errors
    /// have been reported, we will skip regionck and other work that
    /// expects the types within the function to be consistent.
    // FIXME(matthewjasper) This should not exist, and it's not correct
    // if type checking is run in parallel.
    err_count_on_creation: usize,

    /// If `Some`, this stores coercion information for returned
    /// expressions. If `None`, this is in a context where return is
    /// inappropriate, such as a const expression.
    ///
    /// This is a `RefCell<DynamicCoerceMany>`, which means that we
    /// can track all the return expressions and then use them to
    /// compute a useful coercion from the set, similar to a match
    /// expression or other branching context. You can use methods
    /// like `expected_ty` to access the declared return type (if
    /// any).
    ret_coercion: Option<RefCell<DynamicCoerceMany<'tcx>>>,

    /// First span of a return site that we find. Used in error messages.
    ret_coercion_span: RefCell<Option<Span>>,

    resume_yield_tys: Option<(Ty<'tcx>, Ty<'tcx>)>,

    ps: RefCell<UnsafetyState>,

    /// Whether the last checked node generates a divergence (e.g.,
    /// `return` will set this to `Always`). In general, when entering
    /// an expression or other node in the tree, the initial value
    /// indicates whether prior parts of the containing expression may
    /// have diverged. It is then typically set to `Maybe` (and the
    /// old value remembered) for processing the subparts of the
    /// current expression. As each subpart is processed, they may set
    /// the flag to `Always`, etc. Finally, at the end, we take the
    /// result and "union" it with the original value, so that when we
    /// return the flag indicates if any subpart of the parent
    /// expression (up to and including this part) has diverged. So,
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
    /// An expression represents dead code if, after checking it,
    /// the diverges flag is set to something other than `Maybe`.
    diverges: Cell<Diverges>,

    /// Whether any child nodes have any type errors.
    has_errors: Cell<bool>,

    enclosing_breakables: RefCell<EnclosingBreakables<'tcx>>,

    inh: &'a Inherited<'a, 'tcx>,
}

impl<'a, 'tcx> Deref for FnCtxt<'a, 'tcx> {
    type Target = Inherited<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.inh
    }
}

/// Helper type of a temporary returned by `Inherited::build(...)`.
/// Necessary because we can't write the following bound:
/// `F: for<'b, 'tcx> where 'tcx FnOnce(Inherited<'b, 'tcx>)`.
pub struct InheritedBuilder<'tcx> {
    infcx: infer::InferCtxtBuilder<'tcx>,
    def_id: DefId,
}

impl Inherited<'_, 'tcx> {
    pub fn build(tcx: TyCtxt<'tcx>, def_id: DefId) -> InheritedBuilder<'tcx> {
        let hir_id_root = if def_id.is_local() {
            let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
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

impl<'tcx> InheritedBuilder<'tcx> {
    fn enter<F, R>(&mut self, f: F) -> R
    where
        F: for<'a> FnOnce(Inherited<'a, 'tcx>) -> R,
    {
        let def_id = self.def_id;
        self.infcx.enter(|infcx| f(Inherited::new(infcx, def_id)))
    }
}

impl Inherited<'a, 'tcx> {
    fn new(infcx: InferCtxt<'a, 'tcx>, def_id: DefId) -> Self {
        let tcx = infcx.tcx;
        let item_id = tcx.hir().as_local_hir_id(def_id);
        let body_id = item_id.and_then(|id| tcx.hir().maybe_body_owned_by(id));
        let implicit_region_bound = body_id.map(|body_id| {
            let body = tcx.hir().body(body_id);
            tcx.mk_region(ty::ReScope(region::Scope {
                id: body.value.hir_id.local_id,
                data: region::ScopeData::CallSite,
            }))
        });

        Inherited {
            tables: MaybeInProgressTables { maybe_tables: infcx.in_progress_tables },
            infcx,
            fulfillment_cx: RefCell::new(TraitEngine::new(tcx)),
            locals: RefCell::new(Default::default()),
            deferred_sized_obligations: RefCell::new(Vec::new()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_generator_interiors: RefCell::new(Vec::new()),
            opaque_types: RefCell::new(Default::default()),
            opaque_types_vars: RefCell::new(Default::default()),
            implicit_region_bound,
            body_id,
        }
    }

    fn register_predicate(&self, obligation: traits::PredicateObligation<'tcx>) {
        debug!("register_predicate({:?})", obligation);
        if obligation.has_escaping_bound_vars() {
            span_bug!(obligation.cause.span, "escaping bound vars in predicate {:?}", obligation);
        }
        self.fulfillment_cx.borrow_mut().register_predicate_obligation(self, obligation);
    }

    fn register_predicates<I>(&self, obligations: I)
    where
        I: IntoIterator<Item = traits::PredicateObligation<'tcx>>,
    {
        for obligation in obligations {
            self.register_predicate(obligation);
        }
    }

    fn register_infer_ok_obligations<T>(&self, infer_ok: InferOk<'tcx, T>) -> T {
        self.register_predicates(infer_ok.obligations);
        infer_ok.value
    }

    fn normalize_associated_types_in<T>(
        &self,
        span: Span,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let ok = self.partially_normalize_associated_types_in(span, body_id, param_env, value);
        self.register_infer_ok_obligations(ok)
    }
}

struct CheckItemTypesVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'tcx> for CheckItemTypesVisitor<'tcx> {
    fn visit_item(&mut self, i: &'tcx hir::Item<'tcx>) {
        check_item_type(self.tcx, i);
    }
    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem<'tcx>) {}
}

pub fn check_wf_new(tcx: TyCtxt<'_>) {
    let visit = wfcheck::CheckTypeWellFormedVisitor::new(tcx);
    tcx.hir().krate().par_visit_all_item_likes(&visit);
}

fn check_mod_item_types(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CheckItemTypesVisitor { tcx });
}

fn typeck_item_bodies(tcx: TyCtxt<'_>, crate_num: CrateNum) {
    debug_assert!(crate_num == LOCAL_CRATE);
    tcx.par_body_owners(|body_owner_def_id| {
        tcx.ensure().typeck_tables_of(body_owner_def_id);
    });
}

fn check_item_well_formed(tcx: TyCtxt<'_>, def_id: DefId) {
    wfcheck::check_item_well_formed(tcx, def_id);
}

fn check_trait_item_well_formed(tcx: TyCtxt<'_>, def_id: DefId) {
    wfcheck::check_trait_item(tcx, def_id);
}

fn check_impl_item_well_formed(tcx: TyCtxt<'_>, def_id: DefId) {
    wfcheck::check_impl_item(tcx, def_id);
}

pub fn provide(providers: &mut Providers<'_>) {
    method::provide(providers);
    *providers = Providers {
        typeck_item_bodies,
        typeck_tables_of,
        diagnostic_only_typeck_tables_of,
        has_typeck_tables,
        adt_destructor,
        used_trait_imports,
        check_item_well_formed,
        check_trait_item_well_formed,
        check_impl_item_well_formed,
        check_mod_item_types,
        ..*providers
    };
}

fn adt_destructor(tcx: TyCtxt<'_>, def_id: DefId) -> Option<ty::Destructor> {
    tcx.calculate_dtor(def_id, &mut dropck::check_drop_impl)
}

/// If this `DefId` is a "primary tables entry", returns
/// `Some((body_id, header, decl))` with information about
/// it's body-id, fn-header and fn-decl (if any). Otherwise,
/// returns `None`.
///
/// If this function returns `Some`, then `typeck_tables(def_id)` will
/// succeed; if it returns `None`, then `typeck_tables(def_id)` may or
/// may not succeed. In some cases where this function returns `None`
/// (notably closures), `typeck_tables(def_id)` would wind up
/// redirecting to the owning function.
fn primary_body_of(
    tcx: TyCtxt<'_>,
    id: hir::HirId,
) -> Option<(hir::BodyId, Option<&hir::Ty<'_>>, Option<&hir::FnHeader>, Option<&hir::FnDecl<'_>>)> {
    match tcx.hir().get(id) {
        Node::Item(item) => match item.kind {
            hir::ItemKind::Const(ref ty, body) | hir::ItemKind::Static(ref ty, _, body) => {
                Some((body, Some(ty), None, None))
            }
            hir::ItemKind::Fn(ref sig, .., body) => {
                Some((body, None, Some(&sig.header), Some(&sig.decl)))
            }
            _ => None,
        },
        Node::TraitItem(item) => match item.kind {
            hir::TraitItemKind::Const(ref ty, Some(body)) => Some((body, Some(ty), None, None)),
            hir::TraitItemKind::Fn(ref sig, hir::TraitMethod::Provided(body)) => {
                Some((body, None, Some(&sig.header), Some(&sig.decl)))
            }
            _ => None,
        },
        Node::ImplItem(item) => match item.kind {
            hir::ImplItemKind::Const(ref ty, body) => Some((body, Some(ty), None, None)),
            hir::ImplItemKind::Method(ref sig, body) => {
                Some((body, None, Some(&sig.header), Some(&sig.decl)))
            }
            _ => None,
        },
        Node::AnonConst(constant) => Some((constant.body, None, None, None)),
        _ => None,
    }
}

fn has_typeck_tables(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // Closures' tables come from their outermost function,
    // as they are part of the same "inference environment".
    let outer_def_id = tcx.closure_base_def_id(def_id);
    if outer_def_id != def_id {
        return tcx.has_typeck_tables(outer_def_id);
    }

    if let Some(id) = tcx.hir().as_local_hir_id(def_id) {
        primary_body_of(tcx, id).is_some()
    } else {
        false
    }
}

fn used_trait_imports(tcx: TyCtxt<'_>, def_id: DefId) -> &DefIdSet {
    &*tcx.typeck_tables_of(def_id).used_trait_imports
}

/// Inspects the substs of opaque types, replacing any inference variables
/// with proper generic parameter from the identity substs.
///
/// This is run after we normalize the function signature, to fix any inference
/// variables introduced by the projection of associated types. This ensures that
/// any opaque types used in the signature continue to refer to generic parameters,
/// allowing them to be considered for defining uses in the function body
///
/// For example, consider this code.
///
/// ```rust
/// trait MyTrait {
///     type MyItem;
///     fn use_it(self) -> Self::MyItem
/// }
/// impl<T, I> MyTrait for T where T: Iterator<Item = I> {
///     type MyItem = impl Iterator<Item = I>;
///     fn use_it(self) -> Self::MyItem {
///         self
///     }
/// }
/// ```
///
/// When we normalize the signature of `use_it` from the impl block,
/// we will normalize `Self::MyItem` to the opaque type `impl Iterator<Item = I>`
/// However, this projection result may contain inference variables, due
/// to the way that projection works. We didn't have any inference variables
/// in the signature to begin with - leaving them in will cause us to incorrectly
/// conclude that we don't have a defining use of `MyItem`. By mapping inference
/// variables back to the actual generic parameters, we will correctly see that
/// we have a defining use of `MyItem`
fn fixup_opaque_types<'tcx, T>(tcx: TyCtxt<'tcx>, val: &T) -> T
where
    T: TypeFoldable<'tcx>,
{
    struct FixupFolder<'tcx> {
        tcx: TyCtxt<'tcx>,
    }

    impl<'tcx> TypeFolder<'tcx> for FixupFolder<'tcx> {
        fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
            match ty.kind {
                ty::Opaque(def_id, substs) => {
                    debug!("fixup_opaque_types: found type {:?}", ty);
                    // Here, we replace any inference variables that occur within
                    // the substs of an opaque type. By definition, any type occurring
                    // in the substs has a corresponding generic parameter, which is what
                    // we replace it with.
                    // This replacement is only run on the function signature, so any
                    // inference variables that we come across must be the rust of projection
                    // (there's no other way for a user to get inference variables into
                    // a function signature).
                    if ty.needs_infer() {
                        let new_substs = InternalSubsts::for_item(self.tcx, def_id, |param, _| {
                            let old_param = substs[param.index as usize];
                            match old_param.unpack() {
                                GenericArgKind::Type(old_ty) => {
                                    if let ty::Infer(_) = old_ty.kind {
                                        // Replace inference type with a generic parameter
                                        self.tcx.mk_param_from_def(param)
                                    } else {
                                        old_param.fold_with(self)
                                    }
                                }
                                GenericArgKind::Const(old_const) => {
                                    if let ty::ConstKind::Infer(_) = old_const.val {
                                        // This should never happen - we currently do not support
                                        // 'const projections', e.g.:
                                        // `impl<T: SomeTrait> MyTrait for T where <T as SomeTrait>::MyConst == 25`
                                        // which should be the only way for us to end up with a const inference
                                        // variable after projection. If Rust ever gains support for this kind
                                        // of projection, this should *probably* be changed to
                                        // `self.tcx.mk_param_from_def(param)`
                                        bug!(
                                            "Found infer const: `{:?}` in opaque type: {:?}",
                                            old_const,
                                            ty
                                        );
                                    } else {
                                        old_param.fold_with(self)
                                    }
                                }
                                GenericArgKind::Lifetime(old_region) => {
                                    if let RegionKind::ReVar(_) = old_region {
                                        self.tcx.mk_param_from_def(param)
                                    } else {
                                        old_param.fold_with(self)
                                    }
                                }
                            }
                        });
                        let new_ty = self.tcx.mk_opaque(def_id, new_substs);
                        debug!("fixup_opaque_types: new type: {:?}", new_ty);
                        new_ty
                    } else {
                        ty
                    }
                }
                _ => ty.super_fold_with(self),
            }
        }
    }

    debug!("fixup_opaque_types({:?})", val);
    val.fold_with(&mut FixupFolder { tcx })
}

fn typeck_tables_of<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &ty::TypeckTables<'tcx> {
    let fallback = move || tcx.type_of(def_id);
    typeck_tables_of_with_fallback(tcx, def_id, fallback)
}

/// Used only to get `TypeckTables` for type inference during error recovery.
/// Currently only used for type inference of `static`s and `const`s to avoid type cycle errors.
fn diagnostic_only_typeck_tables_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> &ty::TypeckTables<'tcx> {
    assert!(def_id.is_local());
    let fallback = move || {
        let span = tcx.hir().span(tcx.hir().as_local_hir_id(def_id).unwrap());
        tcx.sess.delay_span_bug(span, "diagnostic only typeck table used");
        tcx.types.err
    };
    typeck_tables_of_with_fallback(tcx, def_id, fallback)
}

fn typeck_tables_of_with_fallback<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    fallback: impl Fn() -> Ty<'tcx> + 'tcx,
) -> &'tcx ty::TypeckTables<'tcx> {
    // Closures' tables come from their outermost function,
    // as they are part of the same "inference environment".
    let outer_def_id = tcx.closure_base_def_id(def_id);
    if outer_def_id != def_id {
        return tcx.typeck_tables_of(outer_def_id);
    }

    let id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let span = tcx.hir().span(id);

    // Figure out what primary body this item has.
    let (body_id, body_ty, fn_header, fn_decl) = primary_body_of(tcx, id).unwrap_or_else(|| {
        span_bug!(span, "can't type-check body of {:?}", def_id);
    });
    let body = tcx.hir().body(body_id);

    let tables = Inherited::build(tcx, def_id).enter(|inh| {
        let param_env = tcx.param_env(def_id);
        let fcx = if let (Some(header), Some(decl)) = (fn_header, fn_decl) {
            let fn_sig = if crate::collect::get_infer_ret_ty(&decl.output).is_some() {
                let fcx = FnCtxt::new(&inh, param_env, body.value.hir_id);
                AstConv::ty_of_fn(&fcx, header.unsafety, header.abi, decl, &[], None)
            } else {
                tcx.fn_sig(def_id)
            };

            check_abi(tcx, span, fn_sig.abi());

            // Compute the fty from point of view of inside the fn.
            let fn_sig = tcx.liberate_late_bound_regions(def_id, &fn_sig);
            let fn_sig = inh.normalize_associated_types_in(
                body.value.span,
                body_id.hir_id,
                param_env,
                &fn_sig,
            );

            let fn_sig = fixup_opaque_types(tcx, &fn_sig);

            let fcx = check_fn(&inh, param_env, fn_sig, decl, id, body, None).0;
            fcx
        } else {
            let fcx = FnCtxt::new(&inh, param_env, body.value.hir_id);
            let expected_type = body_ty
                .and_then(|ty| match ty.kind {
                    hir::TyKind::Infer => Some(AstConv::ast_ty_to_ty(&fcx, ty)),
                    _ => None,
                })
                .unwrap_or_else(fallback);
            let expected_type = fcx.normalize_associated_types_in(body.value.span, &expected_type);
            fcx.require_type_is_sized(expected_type, body.value.span, traits::ConstSized);

            let revealed_ty = if tcx.features().impl_trait_in_bindings {
                fcx.instantiate_opaque_types_from_value(id, &expected_type, body.value.span)
            } else {
                expected_type
            };

            // Gather locals in statics (because of block expressions).
            GatherLocalsVisitor { fcx: &fcx, parent_id: id }.visit_body(body);

            fcx.check_expr_coercable_to_type(&body.value, revealed_ty);

            fcx.write_ty(id, revealed_ty);

            fcx
        };

        // All type checking constraints were added, try to fallback unsolved variables.
        fcx.select_obligations_where_possible(false, |_| {});
        let mut fallback_has_occurred = false;

        // We do fallback in two passes, to try to generate
        // better error messages.
        // The first time, we do *not* replace opaque types.
        for ty in &fcx.unsolved_variables() {
            fallback_has_occurred |= fcx.fallback_if_possible(ty, FallbackMode::NoOpaque);
        }
        // We now see if we can make progress. This might
        // cause us to unify inference variables for opaque types,
        // since we may have unified some other type variables
        // during the first phase of fallback.
        // This means that we only replace inference variables with their underlying
        // opaque types as a last resort.
        //
        // In code like this:
        //
        // ```rust
        // type MyType = impl Copy;
        // fn produce() -> MyType { true }
        // fn bad_produce() -> MyType { panic!() }
        // ```
        //
        // we want to unify the opaque inference variable in `bad_produce`
        // with the diverging fallback for `panic!` (e.g. `()` or `!`).
        // This will produce a nice error message about conflicting concrete
        // types for `MyType`.
        //
        // If we had tried to fallback the opaque inference variable to `MyType`,
        // we will generate a confusing type-check error that does not explicitly
        // refer to opaque types.
        fcx.select_obligations_where_possible(fallback_has_occurred, |_| {});

        // We now run fallback again, but this time we allow it to replace
        // unconstrained opaque type variables, in addition to performing
        // other kinds of fallback.
        for ty in &fcx.unsolved_variables() {
            fallback_has_occurred |= fcx.fallback_if_possible(ty, FallbackMode::All);
        }

        // See if we can make any more progress.
        fcx.select_obligations_where_possible(fallback_has_occurred, |_| {});

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
    assert_eq!(tables.local_id_root, Some(DefId::local(id.owner)));

    tables
}

fn check_abi(tcx: TyCtxt<'_>, span: Span, abi: Abi) {
    if !tcx.sess.target.target.is_abi_supported(abi) {
        struct_span_err!(
            tcx.sess,
            span,
            E0570,
            "The ABI `{}` is not supported for the current target",
            abi
        )
        .emit()
    }
}

struct GatherLocalsVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    parent_id: hir::HirId,
}

impl<'a, 'tcx> GatherLocalsVisitor<'a, 'tcx> {
    fn assign(&mut self, span: Span, nid: hir::HirId, ty_opt: Option<LocalTy<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // infer the variable's type
                let var_ty = self.fcx.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeInference,
                    span,
                });
                self.fcx
                    .locals
                    .borrow_mut()
                    .insert(nid, LocalTy { decl_ty: var_ty, revealed_ty: var_ty });
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

impl<'a, 'tcx> Visitor<'tcx> for GatherLocalsVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::None
    }

    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'tcx hir::Local<'tcx>) {
        let local_ty = match local.ty {
            Some(ref ty) => {
                let o_ty = self.fcx.to_ty(&ty);

                let revealed_ty = if self.fcx.tcx.features().impl_trait_in_bindings {
                    self.fcx.instantiate_opaque_types_from_value(self.parent_id, &o_ty, ty.span)
                } else {
                    o_ty
                };

                let c_ty = self
                    .fcx
                    .inh
                    .infcx
                    .canonicalize_user_type_annotation(&UserType::Ty(revealed_ty));
                debug!(
                    "visit_local: ty.hir_id={:?} o_ty={:?} revealed_ty={:?} c_ty={:?}",
                    ty.hir_id, o_ty, revealed_ty, c_ty
                );
                self.fcx.tables.borrow_mut().user_provided_types_mut().insert(ty.hir_id, c_ty);

                Some(LocalTy { decl_ty: o_ty, revealed_ty })
            }
            None => None,
        };
        self.assign(local.span, local.hir_id, local_ty);

        debug!(
            "local variable {:?} is assigned type {}",
            local.pat,
            self.fcx.ty_to_string(&*self.fcx.locals.borrow().get(&local.hir_id).unwrap().decl_ty)
        );
        intravisit::walk_local(self, local);
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        if let PatKind::Binding(_, _, ident, _) = p.kind {
            let var_ty = self.assign(p.span, p.hir_id, None);

            if !self.fcx.tcx.features().unsized_locals {
                self.fcx.require_type_is_sized(var_ty, p.span, traits::VariableType(p.hir_id));
            }

            debug!(
                "pattern binding {} is assigned to {} with type {:?}",
                ident,
                self.fcx.ty_to_string(&*self.fcx.locals.borrow().get(&p.hir_id).unwrap().decl_ty),
                var_ty
            );
        }
        intravisit::walk_pat(self, p);
    }

    // Don't descend into the bodies of nested closures
    fn visit_fn(
        &mut self,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        _: hir::BodyId,
        _: Span,
        _: hir::HirId,
    ) {
    }
}

/// When `check_fn` is invoked on a generator (i.e., a body that
/// includes yield), it returns back some information about the yield
/// points.
struct GeneratorTypes<'tcx> {
    /// Type of generator argument / values returned by `yield`.
    resume_ty: Ty<'tcx>,

    /// Type of value that is yielded.
    yield_ty: Ty<'tcx>,

    /// Types that are captured (see `GeneratorInterior` for more).
    interior: Ty<'tcx>,

    /// Indicates if the generator is movable or static (immovable).
    movability: hir::Movability,
}

/// Helper used for fns and closures. Does the grungy work of checking a function
/// body and returns the function context used for that purpose, since in the case of a fn item
/// there is still a bit more to do.
///
/// * ...
/// * inherited: other fields inherited from the enclosing fn (if any)
fn check_fn<'a, 'tcx>(
    inherited: &'a Inherited<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    fn_sig: ty::FnSig<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    fn_id: hir::HirId,
    body: &'tcx hir::Body<'tcx>,
    can_be_generator: Option<hir::Movability>,
) -> (FnCtxt<'a, 'tcx>, Option<GeneratorTypes<'tcx>>) {
    let mut fn_sig = fn_sig;

    debug!("check_fn(sig={:?}, fn_id={}, param_env={:?})", fn_sig, fn_id, param_env);

    // Create the function context.  This is either derived from scratch or,
    // in the case of closures, based on the outer context.
    let mut fcx = FnCtxt::new(inherited, param_env, body.value.hir_id);
    *fcx.ps.borrow_mut() = UnsafetyState::function(fn_sig.unsafety, fn_id);

    let tcx = fcx.tcx;
    let sess = tcx.sess;
    let hir = tcx.hir();

    let declared_ret_ty = fn_sig.output();
    fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
    let revealed_ret_ty =
        fcx.instantiate_opaque_types_from_value(fn_id, &declared_ret_ty, decl.output.span());
    debug!("check_fn: declared_ret_ty: {}, revealed_ret_ty: {}", declared_ret_ty, revealed_ret_ty);
    fcx.ret_coercion = Some(RefCell::new(CoerceMany::new(revealed_ret_ty)));
    fn_sig = tcx.mk_fn_sig(
        fn_sig.inputs().iter().cloned(),
        revealed_ret_ty,
        fn_sig.c_variadic,
        fn_sig.unsafety,
        fn_sig.abi,
    );

    let span = body.value.span;

    fn_maybe_err(tcx, span, fn_sig.abi);

    if body.generator_kind.is_some() && can_be_generator.is_some() {
        let yield_ty = fcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span });
        fcx.require_type_is_sized(yield_ty, span, traits::SizedYieldType);

        // Resume type defaults to `()` if the generator has no argument.
        let resume_ty = fn_sig.inputs().get(0).copied().unwrap_or_else(|| tcx.mk_unit());

        fcx.resume_yield_tys = Some((resume_ty, yield_ty));
    }

    let outer_def_id = tcx.closure_base_def_id(hir.local_def_id(fn_id));
    let outer_hir_id = hir.as_local_hir_id(outer_def_id).unwrap();
    GatherLocalsVisitor { fcx: &fcx, parent_id: outer_hir_id }.visit_body(body);

    // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
    // (as it's created inside the body itself, not passed in from outside).
    let maybe_va_list = if fn_sig.c_variadic {
        let va_list_did = tcx.require_lang_item(
            lang_items::VaListTypeLangItem,
            Some(body.params.last().unwrap().span),
        );
        let region = tcx.mk_region(ty::ReScope(region::Scope {
            id: body.value.hir_id.local_id,
            data: region::ScopeData::CallSite,
        }));

        Some(tcx.type_of(va_list_did).subst(tcx, &[region.into()]))
    } else {
        None
    };

    // Add formal parameters.
    let inputs_hir = hir.fn_decl_by_hir_id(fn_id).map(|decl| &decl.inputs);
    let inputs_fn = fn_sig.inputs().iter().copied();
    for (idx, (param_ty, param)) in inputs_fn.chain(maybe_va_list).zip(body.params).enumerate() {
        // Check the pattern.
        fcx.check_pat_top(&param.pat, param_ty, try { inputs_hir?.get(idx)?.span }, false);

        // Check that argument is Sized.
        // The check for a non-trivial pattern is a hack to avoid duplicate warnings
        // for simple cases like `fn foo(x: Trait)`,
        // where we would error once on the parameter as a whole, and once on the binding `x`.
        if param.pat.simple_ident().is_none() && !tcx.features().unsized_locals {
            fcx.require_type_is_sized(param_ty, param.pat.span, traits::SizedArgumentType);
        }

        fcx.write_ty(param.hir_id, param_ty);
    }

    inherited.tables.borrow_mut().liberated_fn_sigs_mut().insert(fn_id, fn_sig);

    fcx.check_return_expr(&body.value);

    // We insert the deferred_generator_interiors entry after visiting the body.
    // This ensures that all nested generators appear before the entry of this generator.
    // resolve_generator_interiors relies on this property.
    let gen_ty = if let (Some(_), Some(gen_kind)) = (can_be_generator, body.generator_kind) {
        let interior = fcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::MiscVariable, span });
        fcx.deferred_generator_interiors.borrow_mut().push((body.id(), interior, gen_kind));

        let (resume_ty, yield_ty) = fcx.resume_yield_tys.unwrap();
        Some(GeneratorTypes {
            resume_ty,
            yield_ty,
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
        actual_return_ty = fcx.next_diverging_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::DivergingFn,
            span,
        });
    }
    fcx.demand_suptype(span, revealed_ret_ty, actual_return_ty);

    // Check that the main return type implements the termination trait.
    if let Some(term_id) = tcx.lang_items().termination() {
        if let Some((def_id, EntryFnType::Main)) = tcx.entry_fn(LOCAL_CRATE) {
            let main_id = hir.as_local_hir_id(def_id).unwrap();
            if main_id == fn_id {
                let substs = tcx.mk_substs_trait(declared_ret_ty, &[]);
                let trait_ref = ty::TraitRef::new(term_id, substs);
                let return_ty_span = decl.output.span();
                let cause = traits::ObligationCause::new(
                    return_ty_span,
                    fn_id,
                    ObligationCauseCode::MainFunctionType,
                );

                inherited.register_predicate(traits::Obligation::new(
                    cause,
                    param_env,
                    trait_ref.without_const().to_predicate(),
                ));
            }
        }
    }

    // Check that a function marked as `#[panic_handler]` has signature `fn(&PanicInfo) -> !`
    if let Some(panic_impl_did) = tcx.lang_items().panic_impl() {
        if panic_impl_did == hir.local_def_id(fn_id) {
            if let Some(panic_info_did) = tcx.lang_items().panic_info() {
                if declared_ret_ty.kind != ty::Never {
                    sess.span_err(decl.output.span(), "return type should be `!`");
                }

                let inputs = fn_sig.inputs();
                let span = hir.span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_panic_info = match inputs[0].kind {
                        ty::Ref(region, ty, mutbl) => match ty.kind {
                            ty::Adt(ref adt, _) => {
                                adt.did == panic_info_did
                                    && mutbl == hir::Mutability::Not
                                    && *region != RegionKind::ReStatic
                            }
                            _ => false,
                        },
                        _ => false,
                    };

                    if !arg_is_panic_info {
                        sess.span_err(decl.inputs[0].span, "argument should be `&PanicInfo`");
                    }

                    if let Node::Item(item) = hir.get(fn_id) {
                        if let ItemKind::Fn(_, ref generics, _) = item.kind {
                            if !generics.params.is_empty() {
                                sess.span_err(span, "should have no type parameters");
                            }
                        }
                    }
                } else {
                    let span = sess.source_map().def_span(span);
                    sess.span_err(span, "function should have one argument");
                }
            } else {
                sess.err("language item required, but not found: `panic_info`");
            }
        }
    }

    // Check that a function marked as `#[alloc_error_handler]` has signature `fn(Layout) -> !`
    if let Some(alloc_error_handler_did) = tcx.lang_items().oom() {
        if alloc_error_handler_did == hir.local_def_id(fn_id) {
            if let Some(alloc_layout_did) = tcx.lang_items().alloc_layout() {
                if declared_ret_ty.kind != ty::Never {
                    sess.span_err(decl.output.span(), "return type should be `!`");
                }

                let inputs = fn_sig.inputs();
                let span = hir.span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_alloc_layout = match inputs[0].kind {
                        ty::Adt(ref adt, _) => adt.did == alloc_layout_did,
                        _ => false,
                    };

                    if !arg_is_alloc_layout {
                        sess.span_err(decl.inputs[0].span, "argument should be `Layout`");
                    }

                    if let Node::Item(item) = hir.get(fn_id) {
                        if let ItemKind::Fn(_, ref generics, _) = item.kind {
                            if !generics.params.is_empty() {
                                sess.span_err(
                                    span,
                                    "`#[alloc_error_handler]` function should have no type \
                                     parameters",
                                );
                            }
                        }
                    }
                } else {
                    let span = sess.source_map().def_span(span);
                    sess.span_err(span, "function should have one argument");
                }
            } else {
                sess.err("language item required, but not found: `alloc_layout`");
            }
        }
    }

    (fcx, gen_ty)
}

fn check_struct(tcx: TyCtxt<'_>, id: hir::HirId, span: Span) {
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

fn check_union(tcx: TyCtxt<'_>, id: hir::HirId, span: Span) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);
    check_transparent(tcx, span, def_id);
    check_union_fields(tcx, span, def_id);
    check_packed(tcx, span, def_id);
}

/// When the `#![feature(untagged_unions)]` gate is active,
/// check that the fields of the `union` does not contain fields that need dropping.
fn check_union_fields(tcx: TyCtxt<'_>, span: Span, item_def_id: DefId) -> bool {
    let item_type = tcx.type_of(item_def_id);
    if let ty::Adt(def, substs) = item_type.kind {
        assert!(def.is_union());
        let fields = &def.non_enum_variant().fields;
        let param_env = tcx.param_env(item_def_id);
        for field in fields {
            let field_ty = field.ty(tcx, substs);
            // We are currently checking the type this field came from, so it must be local.
            let field_span = tcx.hir().span_if_local(field.did).unwrap();
            if field_ty.needs_drop(tcx, param_env) {
                struct_span_err!(
                    tcx.sess,
                    field_span,
                    E0740,
                    "unions may not contain fields that need dropping"
                )
                .span_note(field_span, "`std::mem::ManuallyDrop` can be used to wrap the type")
                .emit();
                return false;
            }
        }
    } else {
        span_bug!(span, "unions must be ty::Adt, but got {:?}", item_type.kind);
    }
    return true;
}

/// Checks that an opaque type does not contain cycles and does not use `Self` or `T::Foo`
/// projections that would result in "inheriting lifetimes".
fn check_opaque<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
    check_opaque_for_inheriting_lifetimes(tcx, def_id, span);
    check_opaque_for_cycles(tcx, def_id, substs, span, origin);
}

/// Checks that an opaque type does not use `Self` or `T::Foo` projections that would result
/// in "inheriting lifetimes".
fn check_opaque_for_inheriting_lifetimes(tcx: TyCtxt<'tcx>, def_id: DefId, span: Span) {
    let item =
        tcx.hir().expect_item(tcx.hir().as_local_hir_id(def_id).expect("opaque type is not local"));
    debug!(
        "check_opaque_for_inheriting_lifetimes: def_id={:?} span={:?} item={:?}",
        def_id, span, item
    );

    #[derive(Debug)]
    struct ProhibitOpaqueVisitor<'tcx> {
        opaque_identity_ty: Ty<'tcx>,
        generics: &'tcx ty::Generics,
    };

    impl<'tcx> ty::fold::TypeVisitor<'tcx> for ProhibitOpaqueVisitor<'tcx> {
        fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
            debug!("check_opaque_for_inheriting_lifetimes: (visit_ty) t={:?}", t);
            if t == self.opaque_identity_ty { false } else { t.super_visit_with(self) }
        }

        fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
            debug!("check_opaque_for_inheriting_lifetimes: (visit_region) r={:?}", r);
            if let RegionKind::ReEarlyBound(ty::EarlyBoundRegion { index, .. }) = r {
                return *index < self.generics.parent_count as u32;
            }

            r.super_visit_with(self)
        }
    }

    let prohibit_opaque = match item.kind {
        ItemKind::OpaqueTy(hir::OpaqueTy { origin: hir::OpaqueTyOrigin::AsyncFn, .. })
        | ItemKind::OpaqueTy(hir::OpaqueTy { origin: hir::OpaqueTyOrigin::FnReturn, .. }) => {
            let mut visitor = ProhibitOpaqueVisitor {
                opaque_identity_ty: tcx
                    .mk_opaque(def_id, InternalSubsts::identity_for_item(tcx, def_id)),
                generics: tcx.generics_of(def_id),
            };
            debug!("check_opaque_for_inheriting_lifetimes: visitor={:?}", visitor);

            tcx.predicates_of(def_id)
                .predicates
                .iter()
                .any(|(predicate, _)| predicate.visit_with(&mut visitor))
        }
        _ => false,
    };

    debug!("check_opaque_for_inheriting_lifetimes: prohibit_opaque={:?}", prohibit_opaque);
    if prohibit_opaque {
        let is_async = match item.kind {
            ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => match origin {
                hir::OpaqueTyOrigin::AsyncFn => true,
                _ => false,
            },
            _ => unreachable!(),
        };

        tcx.sess.span_err(span, &format!(
            "`{}` return type cannot contain a projection or `Self` that references lifetimes from \
             a parent scope",
            if is_async { "async fn" } else { "impl Trait" },
        ));
    }
}

/// Checks that an opaque type does not contain cycles.
fn check_opaque_for_cycles<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
    if let Err(partially_expanded_type) = tcx.try_expand_impl_trait_type(def_id, substs) {
        if let hir::OpaqueTyOrigin::AsyncFn = origin {
            struct_span_err!(tcx.sess, span, E0733, "recursion in an `async fn` requires boxing",)
                .span_label(span, "recursive `async fn`")
                .note("a recursive `async fn` must be rewritten to return a boxed `dyn Future`")
                .emit();
        } else {
            let mut err =
                struct_span_err!(tcx.sess, span, E0720, "opaque type expands to a recursive type",);
            err.span_label(span, "expands to a recursive type");
            if let ty::Opaque(..) = partially_expanded_type.kind {
                err.note("type resolves to itself");
            } else {
                err.note(&format!("expanded type is `{}`", partially_expanded_type));
            }
            err.emit();
        }
    }
}

// Forbid defining intrinsics in Rust code,
// as they must always be defined by the compiler.
fn fn_maybe_err(tcx: TyCtxt<'_>, sp: Span, abi: Abi) {
    if let Abi::RustIntrinsic | Abi::PlatformIntrinsic = abi {
        tcx.sess.span_err(sp, "intrinsic must be in `extern \"rust-intrinsic\" { ... }` block");
    }
}

pub fn check_item_type<'tcx>(tcx: TyCtxt<'tcx>, it: &'tcx hir::Item<'tcx>) {
    debug!(
        "check_item_type(it.hir_id={}, it.name={})",
        it.hir_id,
        tcx.def_path_str(tcx.hir().local_def_id(it.hir_id))
    );
    let _indenter = indenter();
    match it.kind {
        // Consts can play a role in type-checking, so they are included here.
        hir::ItemKind::Static(..) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            tcx.typeck_tables_of(def_id);
            maybe_check_static_with_link_section(tcx, def_id, it.span);
        }
        hir::ItemKind::Const(..) => {
            tcx.typeck_tables_of(tcx.hir().local_def_id(it.hir_id));
        }
        hir::ItemKind::Enum(ref enum_definition, _) => {
            check_enum(tcx, it.span, &enum_definition.variants, it.hir_id);
        }
        hir::ItemKind::Fn(..) => {} // entirely within check_item_body
        hir::ItemKind::Impl { ref items, .. } => {
            debug!("ItemKind::Impl {} with id {}", it.ident, it.hir_id);
            let impl_def_id = tcx.hir().local_def_id(it.hir_id);
            if let Some(impl_trait_ref) = tcx.impl_trait_ref(impl_def_id) {
                check_impl_items_against_trait(tcx, it.span, impl_def_id, impl_trait_ref, items);
                let trait_def_id = impl_trait_ref.def_id;
                check_on_unimplemented(tcx, trait_def_id, it);
            }
        }
        hir::ItemKind::Trait(_, _, _, _, ref items) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            check_on_unimplemented(tcx, def_id, it);

            for item in items.iter() {
                let item = tcx.hir().trait_item(item.id);
                if let hir::TraitItemKind::Fn(sig, _) = &item.kind {
                    let abi = sig.header.abi;
                    fn_maybe_err(tcx, item.ident.span, abi);
                }
            }
        }
        hir::ItemKind::Struct(..) => {
            check_struct(tcx, it.hir_id, it.span);
        }
        hir::ItemKind::Union(..) => {
            check_union(tcx, it.hir_id, it.span);
        }
        hir::ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);

            let substs = InternalSubsts::identity_for_item(tcx, def_id);
            check_opaque(tcx, def_id, substs, it.span, &origin);
        }
        hir::ItemKind::TyAlias(..) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            let pty_ty = tcx.type_of(def_id);
            let generics = tcx.generics_of(def_id);
            check_bounds_are_used(tcx, &generics, pty_ty);
        }
        hir::ItemKind::ForeignMod(ref m) => {
            check_abi(tcx, it.span, m.abi);

            if m.abi == Abi::RustIntrinsic {
                for item in m.items {
                    intrinsic::check_intrinsic_type(tcx, item);
                }
            } else if m.abi == Abi::PlatformIntrinsic {
                for item in m.items {
                    intrinsic::check_platform_intrinsic_type(tcx, item);
                }
            } else {
                for item in m.items {
                    let generics = tcx.generics_of(tcx.hir().local_def_id(item.hir_id));
                    let own_counts = generics.own_counts();
                    if generics.params.len() - own_counts.lifetimes != 0 {
                        let (kinds, kinds_pl, egs) = match (own_counts.types, own_counts.consts) {
                            (_, 0) => ("type", "types", Some("u32")),
                            // We don't specify an example value, because we can't generate
                            // a valid value for any type.
                            (0, _) => ("const", "consts", None),
                            _ => ("type or const", "types or consts", None),
                        };
                        struct_span_err!(
                            tcx.sess,
                            item.span,
                            E0044,
                            "foreign items may not have {} parameters",
                            kinds,
                        )
                        .span_label(item.span, &format!("can't have {} parameters", kinds))
                        .help(
                            // FIXME: once we start storing spans for type arguments, turn this
                            // into a suggestion.
                            &format!(
                                "replace the {} parameters with concrete {}{}",
                                kinds,
                                kinds_pl,
                                egs.map(|egs| format!(" like `{}`", egs)).unwrap_or_default(),
                            ),
                        )
                        .emit();
                    }

                    if let hir::ForeignItemKind::Fn(ref fn_decl, _, _) = item.kind {
                        require_c_abi_if_c_variadic(tcx, fn_decl, m.abi, item.span);
                    }
                }
            }
        }
        _ => { /* nothing to do */ }
    }
}

fn maybe_check_static_with_link_section(tcx: TyCtxt<'_>, id: DefId, span: Span) {
    // Only restricted on wasm32 target for now
    if !tcx.sess.opts.target_triple.triple().starts_with("wasm32") {
        return;
    }

    // If `#[link_section]` is missing, then nothing to verify
    let attrs = tcx.codegen_fn_attrs(id);
    if attrs.link_section.is_none() {
        return;
    }

    // For the wasm32 target statics with `#[link_section]` are placed into custom
    // sections of the final output file, but this isn't link custom sections of
    // other executable formats. Namely we can only embed a list of bytes,
    // nothing with pointers to anything else or relocations. If any relocation
    // show up, reject them here.
    // `#[link_section]` may contain arbitrary, or even undefined bytes, but it is
    // the consumer's responsibility to ensure all bytes that have been read
    // have defined values.
    match tcx.const_eval_poly(id) {
        Ok(ConstValue::ByRef { alloc, .. }) => {
            if alloc.relocations().len() != 0 {
                let msg = "statics with a custom `#[link_section]` must be a \
                       simple list of bytes on the wasm target with no \
                       extra levels of indirection such as references";
                tcx.sess.span_err(span, msg);
            }
        }
        Ok(_) => bug!("Matching on non-ByRef static"),
        Err(_) => {}
    }
}

fn check_on_unimplemented(tcx: TyCtxt<'_>, trait_def_id: DefId, item: &hir::Item<'_>) {
    let item_def_id = tcx.hir().local_def_id(item.hir_id);
    // an error would be reported if this fails.
    let _ = traits::OnUnimplementedDirective::of_item(tcx, trait_def_id, item_def_id);
}

fn report_forbidden_specialization(
    tcx: TyCtxt<'_>,
    impl_item: &hir::ImplItem<'_>,
    parent_impl: DefId,
) {
    let mut err = struct_span_err!(
        tcx.sess,
        impl_item.span,
        E0520,
        "`{}` specializes an item from a parent `impl`, but \
         that item is not marked `default`",
        impl_item.ident
    );
    err.span_label(impl_item.span, format!("cannot specialize default item `{}`", impl_item.ident));

    match tcx.span_of_impl(parent_impl) {
        Ok(span) => {
            err.span_label(span, "parent `impl` is here");
            err.note(&format!(
                "to specialize, `{}` in the parent `impl` must be marked `default`",
                impl_item.ident
            ));
        }
        Err(cname) => {
            err.note(&format!("parent implementation is in crate `{}`", cname));
        }
    }

    err.emit();
}

fn check_specialization_validity<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def: &ty::TraitDef,
    trait_item: &ty::AssocItem,
    impl_id: DefId,
    impl_item: &hir::ImplItem<'_>,
) {
    let kind = match impl_item.kind {
        hir::ImplItemKind::Const(..) => ty::AssocKind::Const,
        hir::ImplItemKind::Method(..) => ty::AssocKind::Method,
        hir::ImplItemKind::OpaqueTy(..) => ty::AssocKind::OpaqueTy,
        hir::ImplItemKind::TyAlias(_) => ty::AssocKind::Type,
    };

    let mut ancestor_impls = trait_def
        .ancestors(tcx, impl_id)
        .skip(1)
        .filter_map(|parent| {
            if parent.is_from_trait() {
                None
            } else {
                Some((parent, parent.item(tcx, trait_item.ident, kind, trait_def.def_id)))
            }
        })
        .peekable();

    if ancestor_impls.peek().is_none() {
        // No parent, nothing to specialize.
        return;
    }

    let opt_result = ancestor_impls.find_map(|(parent_impl, parent_item)| {
        match parent_item {
            // Parent impl exists, and contains the parent item we're trying to specialize, but
            // doesn't mark it `default`.
            Some(parent_item) if traits::impl_item_is_final(tcx, &parent_item) => {
                Some(Err(parent_impl.def_id()))
            }

            // Parent impl contains item and makes it specializable.
            Some(_) => Some(Ok(())),

            // Parent impl doesn't mention the item. This means it's inherited from the
            // grandparent. In that case, if parent is a `default impl`, inherited items use the
            // "defaultness" from the grandparent, else they are final.
            None => {
                if traits::impl_is_default(tcx, parent_impl.def_id()) {
                    None
                } else {
                    Some(Err(parent_impl.def_id()))
                }
            }
        }
    });

    // If `opt_result` is `None`, we have only encountered `default impl`s that don't contain the
    // item. This is allowed, the item isn't actually getting specialized here.
    let result = opt_result.unwrap_or(Ok(()));

    if let Err(parent_impl) = result {
        report_forbidden_specialization(tcx, impl_item, parent_impl);
    }
}

fn check_impl_items_against_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    full_impl_span: Span,
    impl_id: DefId,
    impl_trait_ref: ty::TraitRef<'tcx>,
    impl_item_refs: &[hir::ImplItemRef<'_>],
) {
    let impl_span = tcx.sess.source_map().def_span(full_impl_span);

    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() {
        return;
    }

    // Locate trait definition and items
    let trait_def = tcx.trait_def(impl_trait_ref.def_id);

    let impl_items = || impl_item_refs.iter().map(|iiref| tcx.hir().impl_item(iiref.id));

    // Check existing impl methods to see if they are both present in trait
    // and compatible with trait signature
    for impl_item in impl_items() {
        let namespace = impl_item.kind.namespace();
        let ty_impl_item = tcx.associated_item(tcx.hir().local_def_id(impl_item.hir_id));
        let ty_trait_item = tcx
            .associated_items(impl_trait_ref.def_id)
            .find_by_name_and_namespace(tcx, ty_impl_item.ident, namespace, impl_trait_ref.def_id)
            .or_else(|| {
                // Not compatible, but needed for the error message
                tcx.associated_items(impl_trait_ref.def_id)
                    .filter_by_name(tcx, ty_impl_item.ident, impl_trait_ref.def_id)
                    .next()
            });

        // Check that impl definition matches trait definition
        if let Some(ty_trait_item) = ty_trait_item {
            match impl_item.kind {
                hir::ImplItemKind::Const(..) => {
                    // Find associated const definition.
                    if ty_trait_item.kind == ty::AssocKind::Const {
                        compare_const_impl(
                            tcx,
                            &ty_impl_item,
                            impl_item.span,
                            &ty_trait_item,
                            impl_trait_ref,
                        );
                    } else {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            impl_item.span,
                            E0323,
                            "item `{}` is an associated const, \
                              which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref.print_only_trait_path()
                        );
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
                    let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssocKind::Method {
                        compare_impl_method(
                            tcx,
                            &ty_impl_item,
                            impl_item.span,
                            &ty_trait_item,
                            impl_trait_ref,
                            opt_trait_span,
                        );
                    } else {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            impl_item.span,
                            E0324,
                            "item `{}` is an associated method, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref.print_only_trait_path()
                        );
                        err.span_label(impl_item.span, "does not match trait");
                        if let Some(trait_span) = opt_trait_span {
                            err.span_label(trait_span, "item in trait");
                        }
                        err.emit()
                    }
                }
                hir::ImplItemKind::OpaqueTy(..) | hir::ImplItemKind::TyAlias(_) => {
                    let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssocKind::Type {
                        compare_ty_impl(
                            tcx,
                            &ty_impl_item,
                            impl_item.span,
                            &ty_trait_item,
                            impl_trait_ref,
                            opt_trait_span,
                        )
                    } else {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            impl_item.span,
                            E0325,
                            "item `{}` is an associated type, \
                             which doesn't match its trait `{}`",
                            ty_impl_item.ident,
                            impl_trait_ref.print_only_trait_path()
                        );
                        err.span_label(impl_item.span, "does not match trait");
                        if let Some(trait_span) = opt_trait_span {
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
    for trait_item in tcx.associated_items(impl_trait_ref.def_id).in_definition_order() {
        let is_implemented = trait_def
            .ancestors(tcx, impl_id)
            .leaf_def(tcx, trait_item.ident, trait_item.kind)
            .map(|node_item| !node_item.node.is_from_trait())
            .unwrap_or(false);

        if !is_implemented && !traits::impl_is_default(tcx, impl_id) {
            if !trait_item.defaultness.has_value() {
                missing_items.push(*trait_item);
            }
        }
    }

    if !missing_items.is_empty() {
        missing_items_err(tcx, impl_span, &missing_items, full_impl_span);
    }
}

fn missing_items_err(
    tcx: TyCtxt<'_>,
    impl_span: Span,
    missing_items: &[ty::AssocItem],
    full_impl_span: Span,
) {
    let missing_items_msg = missing_items
        .iter()
        .map(|trait_item| trait_item.ident.to_string())
        .collect::<Vec<_>>()
        .join("`, `");

    let mut err = struct_span_err!(
        tcx.sess,
        impl_span,
        E0046,
        "not all trait items implemented, missing: `{}`",
        missing_items_msg
    );
    err.span_label(impl_span, format!("missing `{}` in implementation", missing_items_msg));

    // `Span` before impl block closing brace.
    let hi = full_impl_span.hi() - BytePos(1);
    // Point at the place right before the closing brace of the relevant `impl` to suggest
    // adding the associated item at the end of its body.
    let sugg_sp = full_impl_span.with_lo(hi).with_hi(hi);
    // Obtain the level of indentation ending in `sugg_sp`.
    let indentation = tcx.sess.source_map().span_to_margin(sugg_sp).unwrap_or(0);
    // Make the whitespace that will make the suggestion have the right indentation.
    let padding: String = (0..indentation).map(|_| " ").collect();

    for trait_item in missing_items {
        let snippet = suggestion_signature(&trait_item, tcx);
        let code = format!("{}{}\n{}", padding, snippet, padding);
        let msg = format!("implement the missing item: `{}`", snippet);
        let appl = Applicability::HasPlaceholders;
        if let Some(span) = tcx.hir().span_if_local(trait_item.def_id) {
            err.span_label(span, format!("`{}` from trait", trait_item.ident));
            err.tool_only_span_suggestion(sugg_sp, &msg, code, appl);
        } else {
            err.span_suggestion_hidden(sugg_sp, &msg, code, appl);
        }
    }
    err.emit();
}

/// Resugar `ty::GenericPredicates` in a way suitable to be used in structured suggestions.
fn bounds_from_generic_predicates(
    tcx: TyCtxt<'_>,
    predicates: ty::GenericPredicates<'_>,
) -> (String, String) {
    let mut types: FxHashMap<Ty<'_>, Vec<DefId>> = FxHashMap::default();
    let mut projections = vec![];
    for (predicate, _) in predicates.predicates {
        debug!("predicate {:?}", predicate);
        match predicate {
            ty::Predicate::Trait(trait_predicate, _) => {
                let entry = types.entry(trait_predicate.skip_binder().self_ty()).or_default();
                let def_id = trait_predicate.skip_binder().def_id();
                if Some(def_id) != tcx.lang_items().sized_trait() {
                    // Type params are `Sized` by default, do not add that restriction to the list
                    // if it is a positive requirement.
                    entry.push(trait_predicate.skip_binder().def_id());
                }
            }
            ty::Predicate::Projection(projection_pred) => {
                projections.push(projection_pred);
            }
            _ => {}
        }
    }
    let generics = if types.is_empty() {
        "".to_string()
    } else {
        format!(
            "<{}>",
            types
                .keys()
                .filter_map(|t| match t.kind {
                    ty::Param(_) => Some(t.to_string()),
                    // Avoid suggesting the following:
                    // fn foo<T, <T as Trait>::Bar>(_: T) where T: Trait, <T as Trait>::Bar: Other {}
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut where_clauses = vec![];
    for (ty, bounds) in types {
        for bound in &bounds {
            where_clauses.push(format!("{}: {}", ty, tcx.def_path_str(*bound)));
        }
    }
    for projection in &projections {
        let p = projection.skip_binder();
        // FIXME: this is not currently supported syntax, we should be looking at the `types` and
        // insert the associated types where they correspond, but for now let's be "lazy" and
        // propose this instead of the following valid resugaring:
        // `T: Trait, Trait::Assoc = K` → `T: Trait<Assoc = K>`
        where_clauses.push(format!("{} = {}", tcx.def_path_str(p.projection_ty.item_def_id), p.ty));
    }
    let where_clauses = if where_clauses.is_empty() {
        String::new()
    } else {
        format!(" where {}", where_clauses.join(", "))
    };
    (generics, where_clauses)
}

/// Return placeholder code for the given function.
fn fn_sig_suggestion(
    tcx: TyCtxt<'_>,
    sig: &ty::FnSig<'_>,
    ident: Ident,
    predicates: ty::GenericPredicates<'_>,
) -> String {
    let args = sig
        .inputs()
        .iter()
        .map(|ty| {
            Some(match ty.kind {
                ty::Param(param) if param.name == kw::SelfUpper => "self".to_string(),
                ty::Ref(reg, ref_ty, mutability) => {
                    let reg = match &format!("{}", reg)[..] {
                        "'_" | "" => String::new(),
                        reg => format!("{} ", reg),
                    };
                    match ref_ty.kind {
                        ty::Param(param) if param.name == kw::SelfUpper => {
                            format!("&{}{}self", reg, mutability.prefix_str())
                        }
                        _ => format!("_: {:?}", ty),
                    }
                }
                _ => format!("_: {:?}", ty),
            })
        })
        .chain(std::iter::once(if sig.c_variadic { Some("...".to_string()) } else { None }))
        .filter_map(|arg| arg)
        .collect::<Vec<String>>()
        .join(", ");
    let output = sig.output();
    let output = if !output.is_unit() { format!(" -> {:?}", output) } else { String::new() };

    let unsafety = sig.unsafety.prefix_str();
    let (generics, where_clauses) = bounds_from_generic_predicates(tcx, predicates);

    // FIXME: this is not entirely correct, as the lifetimes from borrowed params will
    // not be present in the `fn` definition, not will we account for renamed
    // lifetimes between the `impl` and the `trait`, but this should be good enough to
    // fill in a significant portion of the missing code, and other subsequent
    // suggestions can help the user fix the code.
    format!(
        "{}fn {}{}({}){}{} {{ todo!() }}",
        unsafety, ident, generics, args, output, where_clauses
    )
}

/// Return placeholder code for the given associated item.
/// Similar to `ty::AssocItem::suggestion`, but appropriate for use as the code snippet of a
/// structured suggestion.
fn suggestion_signature(assoc: &ty::AssocItem, tcx: TyCtxt<'_>) -> String {
    match assoc.kind {
        ty::AssocKind::Method => {
            // We skip the binder here because the binder would deanonymize all
            // late-bound regions, and we don't want method signatures to show up
            // `as for<'r> fn(&'r MyType)`.  Pretty-printing handles late-bound
            // regions just fine, showing `fn(&MyType)`.
            fn_sig_suggestion(
                tcx,
                tcx.fn_sig(assoc.def_id).skip_binder(),
                assoc.ident,
                tcx.predicates_of(assoc.def_id),
            )
        }
        ty::AssocKind::Type => format!("type {} = Type;", assoc.ident),
        // FIXME(type_alias_impl_trait): we should print bounds here too.
        ty::AssocKind::OpaqueTy => format!("type {} = Type;", assoc.ident),
        ty::AssocKind::Const => {
            let ty = tcx.type_of(assoc.def_id);
            let val = expr::ty_kind_suggestion(ty).unwrap_or("value");
            format!("const {}: {:?} = {};", assoc.ident, ty, val)
        }
    }
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
fn check_representable(tcx: TyCtxt<'_>, sp: Span, item_def_id: DefId) -> bool {
    let rty = tcx.type_of(item_def_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match rty.is_representable(tcx, sp) {
        Representability::SelfRecursive(spans) => {
            let mut err = recursive_type_with_infinite_size_error(tcx, item_def_id);
            for span in spans {
                err.span_label(span, "recursive without indirection");
            }
            err.emit();
            return false;
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    return true;
}

pub fn check_simd(tcx: TyCtxt<'_>, sp: Span, def_id: DefId) {
    let t = tcx.type_of(def_id);
    if let ty::Adt(def, substs) = t.kind {
        if def.is_struct() {
            let fields = &def.non_enum_variant().fields;
            if fields.is_empty() {
                struct_span_err!(tcx.sess, sp, E0075, "SIMD vector cannot be empty").emit();
                return;
            }
            let e = fields[0].ty(tcx, substs);
            if !fields.iter().all(|f| f.ty(tcx, substs) == e) {
                struct_span_err!(tcx.sess, sp, E0076, "SIMD vector should be homogeneous")
                    .span_label(sp, "SIMD elements must have the same type")
                    .emit();
                return;
            }
            match e.kind {
                ty::Param(_) => { /* struct<T>(T, T, T, T) is ok */ }
                _ if e.is_machine() => { /* struct(u8, u8, u8, u8) is ok */ }
                _ => {
                    struct_span_err!(
                        tcx.sess,
                        sp,
                        E0077,
                        "SIMD vector element type should be machine type"
                    )
                    .emit();
                    return;
                }
            }
        }
    }
}

fn check_packed(tcx: TyCtxt<'_>, sp: Span, def_id: DefId) {
    let repr = tcx.adt_def(def_id).repr;
    if repr.packed() {
        for attr in tcx.get_attrs(def_id).iter() {
            for r in attr::find_repr_attrs(&tcx.sess.parse_sess, attr) {
                if let attr::ReprPacked(pack) = r {
                    if let Some(repr_pack) = repr.pack {
                        if pack as u64 != repr_pack.bytes() {
                            struct_span_err!(
                                tcx.sess,
                                sp,
                                E0634,
                                "type has conflicting packed representation hints"
                            )
                            .emit();
                        }
                    }
                }
            }
        }
        if repr.align.is_some() {
            struct_span_err!(
                tcx.sess,
                sp,
                E0587,
                "type has conflicting packed and align representation hints"
            )
            .emit();
        } else {
            if let Some(def_spans) = check_packed_inner(tcx, def_id, &mut vec![]) {
                let mut err = struct_span_err!(
                    tcx.sess,
                    sp,
                    E0588,
                    "packed type cannot transitively contain a `#[repr(align)]` type"
                );

                let hir = tcx.hir();
                if let Some(hir_id) = hir.as_local_hir_id(def_spans[0].0) {
                    if let Node::Item(Item { ident, .. }) = hir.get(hir_id) {
                        err.span_note(
                            tcx.def_span(def_spans[0].0),
                            &format!("`{}` has a `#[repr(align)]` attribute", ident),
                        );
                    }
                }

                if def_spans.len() > 2 {
                    let mut first = true;
                    for (adt_def, span) in def_spans.iter().skip(1).rev() {
                        if let Some(hir_id) = hir.as_local_hir_id(*adt_def) {
                            if let Node::Item(Item { ident, .. }) = hir.get(hir_id) {
                                err.span_note(
                                    *span,
                                    &if first {
                                        format!(
                                            "`{}` contains a field of type `{}`",
                                            tcx.type_of(def_id),
                                            ident
                                        )
                                    } else {
                                        format!("...which contains a field of type `{}`", ident)
                                    },
                                );
                                first = false;
                            }
                        }
                    }
                }

                err.emit();
            }
        }
    }
}

fn check_packed_inner(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    stack: &mut Vec<DefId>,
) -> Option<Vec<(DefId, Span)>> {
    if let ty::Adt(def, substs) = tcx.type_of(def_id).kind {
        if def.is_struct() || def.is_union() {
            if def.repr.align.is_some() {
                return Some(vec![(def.did, DUMMY_SP)]);
            }

            stack.push(def_id);
            for field in &def.non_enum_variant().fields {
                if let ty::Adt(def, _) = field.ty(tcx, substs).kind {
                    if !stack.contains(&def.did) {
                        if let Some(mut defs) = check_packed_inner(tcx, def.did, stack) {
                            defs.push((def.did, field.ident.span));
                            return Some(defs);
                        }
                    }
                }
            }
            stack.pop();
        }
    }

    None
}

/// Emit an error when encountering more or less than one variant in a transparent enum.
fn bad_variant_count<'tcx>(tcx: TyCtxt<'tcx>, adt: &'tcx ty::AdtDef, sp: Span, did: DefId) {
    let variant_spans: Vec<_> = adt
        .variants
        .iter()
        .map(|variant| tcx.hir().span_if_local(variant.def_id).unwrap())
        .collect();
    let msg = format!("needs exactly one variant, but has {}", adt.variants.len(),);
    let mut err = struct_span_err!(tcx.sess, sp, E0731, "transparent enum {}", msg);
    err.span_label(sp, &msg);
    if let [start @ .., end] = &*variant_spans {
        for variant_span in start {
            err.span_label(*variant_span, "");
        }
        err.span_label(*end, &format!("too many variants in `{}`", tcx.def_path_str(did)));
    }
    err.emit();
}

/// Emit an error when encountering more or less than one non-zero-sized field in a transparent
/// enum.
fn bad_non_zero_sized_fields<'tcx>(
    tcx: TyCtxt<'tcx>,
    adt: &'tcx ty::AdtDef,
    field_count: usize,
    field_spans: impl Iterator<Item = Span>,
    sp: Span,
) {
    let msg = format!("needs exactly one non-zero-sized field, but has {}", field_count);
    let mut err = struct_span_err!(
        tcx.sess,
        sp,
        E0690,
        "{}transparent {} {}",
        if adt.is_enum() { "the variant of a " } else { "" },
        adt.descr(),
        msg,
    );
    err.span_label(sp, &msg);
    for sp in field_spans {
        err.span_label(sp, "this field is non-zero-sized");
    }
    err.emit();
}

fn check_transparent(tcx: TyCtxt<'_>, sp: Span, def_id: DefId) {
    let adt = tcx.adt_def(def_id);
    if !adt.repr.transparent() {
        return;
    }
    let sp = tcx.sess.source_map().def_span(sp);

    if adt.is_union() && !tcx.features().transparent_unions {
        feature_err(
            &tcx.sess.parse_sess,
            sym::transparent_unions,
            sp,
            "transparent unions are unstable",
        )
        .emit();
    }

    if adt.variants.len() != 1 {
        bad_variant_count(tcx, adt, sp, def_id);
        if adt.variants.is_empty() {
            // Don't bother checking the fields. No variants (and thus no fields) exist.
            return;
        }
    }

    // For each field, figure out if it's known to be a ZST and align(1)
    let field_infos = adt.all_fields().map(|field| {
        let ty = field.ty(tcx, InternalSubsts::identity_for_item(tcx, field.did));
        let param_env = tcx.param_env(field.did);
        let layout = tcx.layout_of(param_env.and(ty));
        // We are currently checking the type this field came from, so it must be local
        let span = tcx.hir().span_if_local(field.did).unwrap();
        let zst = layout.map(|layout| layout.is_zst()).unwrap_or(false);
        let align1 = layout.map(|layout| layout.align.abi.bytes() == 1).unwrap_or(false);
        (span, zst, align1)
    });

    let non_zst_fields =
        field_infos.clone().filter_map(|(span, zst, _align1)| if !zst { Some(span) } else { None });
    let non_zst_count = non_zst_fields.clone().count();
    if non_zst_count != 1 {
        bad_non_zero_sized_fields(tcx, adt, non_zst_count, non_zst_fields, sp);
    }
    for (span, zst, align1) in field_infos {
        if zst && !align1 {
            struct_span_err!(
                tcx.sess,
                span,
                E0691,
                "zero-sized field in transparent {} has alignment larger than 1",
                adt.descr(),
            )
            .span_label(span, "has alignment larger than 1")
            .emit();
        }
    }
}

#[allow(trivial_numeric_casts)]
pub fn check_enum<'tcx>(
    tcx: TyCtxt<'tcx>,
    sp: Span,
    vs: &'tcx [hir::Variant<'tcx>],
    id: hir::HirId,
) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated

    if vs.is_empty() {
        let attributes = tcx.get_attrs(def_id);
        if let Some(attr) = attr::find_by_name(&attributes, sym::repr) {
            struct_span_err!(
                tcx.sess,
                attr.span,
                E0084,
                "unsupported representation for zero-variant enum"
            )
            .span_label(sp, "zero-variant enum")
            .emit();
        }
    }

    let repr_type_ty = def.repr.discr_type().to_ty(tcx);
    if repr_type_ty == tcx.types.i128 || repr_type_ty == tcx.types.u128 {
        if !tcx.features().repr128 {
            feature_err(
                &tcx.sess.parse_sess,
                sym::repr128,
                sp,
                "repr with 128-bit type is unstable",
            )
            .emit();
        }
    }

    for v in vs {
        if let Some(ref e) = v.disr_expr {
            tcx.typeck_tables_of(tcx.hir().local_def_id(e.hir_id));
        }
    }

    if tcx.adt_def(def_id).repr.int.is_none() && tcx.features().arbitrary_enum_discriminant {
        let is_unit = |var: &hir::Variant<'_>| match var.data {
            hir::VariantData::Unit(..) => true,
            _ => false,
        };

        let has_disr = |var: &hir::Variant<'_>| var.disr_expr.is_some();
        let has_non_units = vs.iter().any(|var| !is_unit(var));
        let disr_units = vs.iter().any(|var| is_unit(&var) && has_disr(&var));
        let disr_non_unit = vs.iter().any(|var| !is_unit(&var) && has_disr(&var));

        if disr_non_unit || (disr_units && has_non_units) {
            let mut err =
                struct_span_err!(tcx.sess, sp, E0732, "`#[repr(inttype)]` must be specified");
            err.emit();
        }
    }

    let mut disr_vals: Vec<Discr<'tcx>> = Vec::with_capacity(vs.len());
    for ((_, discr), v) in def.discriminants(tcx).zip(vs) {
        // Check for duplicate discriminant values
        if let Some(i) = disr_vals.iter().position(|&x| x.val == discr.val) {
            let variant_did = def.variants[VariantIdx::new(i)].def_id;
            let variant_i_hir_id = tcx.hir().as_local_hir_id(variant_did).unwrap();
            let variant_i = tcx.hir().expect_variant(variant_i_hir_id);
            let i_span = match variant_i.disr_expr {
                Some(ref expr) => tcx.hir().span(expr.hir_id),
                None => tcx.hir().span(variant_i_hir_id),
            };
            let span = match v.disr_expr {
                Some(ref expr) => tcx.hir().span(expr.hir_id),
                None => v.span,
            };
            struct_span_err!(
                tcx.sess,
                span,
                E0081,
                "discriminant value `{}` already exists",
                disr_vals[i]
            )
            .span_label(i_span, format!("first use of `{}`", disr_vals[i]))
            .span_label(span, format!("enum already has `{}`", disr_vals[i]))
            .emit();
        }
        disr_vals.push(discr);
    }

    check_representable(tcx, sp, def_id);
    check_transparent(tcx, sp, def_id);
}

fn report_unexpected_variant_res(tcx: TyCtxt<'_>, res: Res, span: Span, qpath: &QPath<'_>) {
    struct_span_err!(
        tcx.sess,
        span,
        E0533,
        "expected unit struct, unit variant or constant, found {} `{}`",
        res.descr(),
        hir::print::to_string(&tcx.hir(), |s| s.print_qpath(qpath, false))
    )
    .emit();
}

impl<'a, 'tcx> AstConv<'tcx> for FnCtxt<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn item_def_id(&self) -> Option<DefId> {
        None
    }

    fn default_constness_for_trait_bounds(&self) -> hir::Constness {
        // FIXME: refactor this into a method
        let node = self.tcx.hir().get(self.body_id);
        if let Some(fn_like) = FnLikeNode::from_node(node) {
            fn_like.constness()
        } else {
            hir::Constness::NotConst
        }
    }

    fn get_type_parameter_bounds(&self, _: Span, def_id: DefId) -> ty::GenericPredicates<'tcx> {
        let tcx = self.tcx;
        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let item_id = tcx.hir().ty_param_owner(hir_id);
        let item_def_id = tcx.hir().local_def_id(item_id);
        let generics = tcx.generics_of(item_def_id);
        let index = generics.param_def_id_to_index[&def_id];
        ty::GenericPredicates {
            parent: None,
            predicates: tcx.arena.alloc_from_iter(self.param_env.caller_bounds.iter().filter_map(
                |&predicate| match predicate {
                    ty::Predicate::Trait(ref data, _)
                        if data.skip_binder().self_ty().is_param(index) =>
                    {
                        // HACK(eddyb) should get the original `Span`.
                        let span = tcx.def_span(def_id);
                        Some((predicate, span))
                    }
                    _ => None,
                },
            )),
        }
    }

    fn re_infer(&self, def: Option<&ty::GenericParamDef>, span: Span) -> Option<ty::Region<'tcx>> {
        let v = match def {
            Some(def) => infer::EarlyBoundRegion(span, def.name),
            None => infer::MiscVariable(span),
        };
        Some(self.next_region_var(v))
    }

    fn allow_ty_infer(&self) -> bool {
        true
    }

    fn ty_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx> {
        if let Some(param) = param {
            if let GenericArgKind::Type(ty) = self.var_for_def(span, param).unpack() {
                return ty;
            }
            unreachable!()
        } else {
            self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span,
            })
        }
    }

    fn ct_infer(
        &self,
        ty: Ty<'tcx>,
        param: Option<&ty::GenericParamDef>,
        span: Span,
    ) -> &'tcx Const<'tcx> {
        if let Some(param) = param {
            if let GenericArgKind::Const(ct) = self.var_for_def(span, param).unpack() {
                return ct;
            }
            unreachable!()
        } else {
            self.next_const_var(
                ty,
                ConstVariableOrigin { kind: ConstVariableOriginKind::ConstInference, span },
            )
        }
    }

    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        let (trait_ref, _) = self.replace_bound_vars_with_fresh_vars(
            span,
            infer::LateBoundRegionConversionTime::AssocTypeProjection(item_def_id),
            &poly_trait_ref,
        );

        let item_substs = <dyn AstConv<'tcx>>::create_substs_for_associated_item(
            self,
            self.tcx,
            span,
            item_def_id,
            item_segment,
            trait_ref.substs,
        );

        self.tcx().mk_projection(item_def_id, item_substs)
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

/// Controls how we perform fallback for unconstrained
/// type variables.
enum FallbackMode {
    /// Do not fallback type variables to opaque types.
    NoOpaque,
    /// Perform all possible kinds of fallback, including
    /// turning type variables to opaque types.
    All,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn new(
        inh: &'a Inherited<'a, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
    ) -> FnCtxt<'a, 'tcx> {
        FnCtxt {
            body_id,
            param_env,
            err_count_on_creation: inh.tcx.sess.err_count(),
            ret_coercion: None,
            ret_coercion_span: RefCell::new(None),
            resume_yield_tys: None,
            ps: RefCell::new(UnsafetyState::function(hir::Unsafety::Normal, hir::CRATE_HIR_ID)),
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

    pub fn errors_reported_since_creation(&self) -> bool {
        self.tcx.sess.err_count() > self.err_count_on_creation
    }

    /// Produces warning on the given node, if the current point in the
    /// function is unreachable, and there hasn't been another warning.
    fn warn_if_unreachable(&self, id: hir::HirId, span: Span, kind: &str) {
        // FIXME: Combine these two 'if' expressions into one once
        // let chains are implemented
        if let Diverges::Always { span: orig_span, custom_note } = self.diverges.get() {
            // If span arose from a desugaring of `if` or `while`, then it is the condition itself,
            // which diverges, that we are about to lint on. This gives suboptimal diagnostics.
            // Instead, stop here so that the `if`- or `while`-expression's block is linted instead.
            if !span.is_desugaring(DesugaringKind::CondTemporary)
                && !span.is_desugaring(DesugaringKind::Async)
                && !orig_span.is_desugaring(DesugaringKind::Await)
            {
                self.diverges.set(Diverges::WarnedAlways);

                debug!("warn_if_unreachable: id={:?} span={:?} kind={}", id, span, kind);

                self.tcx().struct_span_lint_hir(lint::builtin::UNREACHABLE_CODE, id, span, |lint| {
                    let msg = format!("unreachable {}", kind);
                    lint.build(&msg)
                        .span_label(span, &msg)
                        .span_label(
                            orig_span,
                            custom_note
                                .unwrap_or("any code following this expression is unreachable"),
                        )
                        .emit();
                })
            }
        }
    }

    pub fn cause(&self, span: Span, code: ObligationCauseCode<'tcx>) -> ObligationCause<'tcx> {
        ObligationCause::new(span, self.body_id, code)
    }

    pub fn misc(&self, span: Span) -> ObligationCause<'tcx> {
        self.cause(span, ObligationCauseCode::MiscObligation)
    }

    /// Resolves type and const variables in `ty` if possible. Unlike the infcx
    /// version (resolve_vars_if_possible), this version will
    /// also select obligations if it seems useful, in an effort
    /// to get more type information.
    fn resolve_vars_with_obligations(&self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        debug!("resolve_vars_with_obligations(ty={:?})", ty);

        // No Infer()? Nothing needs doing.
        if !ty.has_infer_types_or_consts() {
            debug!("resolve_vars_with_obligations: ty={:?}", ty);
            return ty;
        }

        // If `ty` is a type variable, see whether we already know what it is.
        ty = self.resolve_vars_if_possible(&ty);
        if !ty.has_infer_types_or_consts() {
            debug!("resolve_vars_with_obligations: ty={:?}", ty);
            return ty;
        }

        // If not, try resolving pending obligations as much as
        // possible. This can help substantially when there are
        // indirect dependencies that don't seem worth tracking
        // precisely.
        self.select_obligations_where_possible(false, |_| {});
        ty = self.resolve_vars_if_possible(&ty);

        debug!("resolve_vars_with_obligations: ty={:?}", ty);
        ty
    }

    fn record_deferred_call_resolution(
        &self,
        closure_def_id: DefId,
        r: DeferredCallResolution<'tcx>,
    ) {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.entry(closure_def_id).or_default().push(r);
    }

    fn remove_deferred_call_resolutions(
        &self,
        closure_def_id: DefId,
    ) -> Vec<DeferredCallResolution<'tcx>> {
        let mut deferred_call_resolutions = self.deferred_call_resolutions.borrow_mut();
        deferred_call_resolutions.remove(&closure_def_id).unwrap_or(vec![])
    }

    pub fn tag(&self) -> String {
        format!("{:p}", self)
    }

    pub fn local_ty(&self, span: Span, nid: hir::HirId) -> LocalTy<'tcx> {
        self.locals.borrow().get(&nid).cloned().unwrap_or_else(|| {
            span_bug!(span, "no type for local variable {}", self.tcx.hir().node_to_string(nid))
        })
    }

    #[inline]
    pub fn write_ty(&self, id: hir::HirId, ty: Ty<'tcx>) {
        debug!(
            "write_ty({:?}, {:?}) in fcx {}",
            id,
            self.resolve_vars_if_possible(&ty),
            self.tag()
        );
        self.tables.borrow_mut().node_types_mut().insert(id, ty);

        if ty.references_error() {
            self.has_errors.set(true);
            self.set_tainted_by_errors();
        }
    }

    pub fn write_field_index(&self, hir_id: hir::HirId, index: usize) {
        self.tables.borrow_mut().field_indices_mut().insert(hir_id, index);
    }

    fn write_resolution(&self, hir_id: hir::HirId, r: Result<(DefKind, DefId), ErrorReported>) {
        self.tables.borrow_mut().type_dependent_defs_mut().insert(hir_id, r);
    }

    pub fn write_method_call(&self, hir_id: hir::HirId, method: MethodCallee<'tcx>) {
        debug!("write_method_call(hir_id={:?}, method={:?})", hir_id, method);
        self.write_resolution(hir_id, Ok((DefKind::AssocFn, method.def_id)));
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
                let user_type_annotation = self.infcx.probe(|_| {
                    let user_substs = UserSubsts {
                        substs: InternalSubsts::for_item(self.tcx, method.def_id, |param, _| {
                            let i = param.index as usize;
                            if i < method_generics.parent_count {
                                self.infcx.var_for_def(DUMMY_SP, param)
                            } else {
                                method.substs[i]
                            }
                        }),
                        user_self_ty: None, // not relevant here
                    };

                    self.infcx.canonicalize_user_type_annotation(&UserType::TypeOf(
                        method.def_id,
                        user_substs,
                    ))
                });

                debug!("write_method_call: user_type_annotation={:?}", user_type_annotation);
                self.write_user_type_annotation(hir_id, user_type_annotation);
            }
        }
    }

    pub fn write_substs(&self, node_id: hir::HirId, substs: SubstsRef<'tcx>) {
        if !substs.is_noop() {
            debug!("write_substs({:?}, {:?}) in fcx {}", node_id, substs, self.tag());

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
    pub fn write_user_type_annotation_from_substs(
        &self,
        hir_id: hir::HirId,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        user_self_ty: Option<UserSelfTy<'tcx>>,
    ) {
        debug!(
            "write_user_type_annotation_from_substs: hir_id={:?} def_id={:?} substs={:?} \
             user_self_ty={:?} in fcx {}",
            hir_id,
            def_id,
            substs,
            user_self_ty,
            self.tag(),
        );

        if Self::can_contain_user_lifetime_bounds((substs, user_self_ty)) {
            let canonicalized = self.infcx.canonicalize_user_type_annotation(&UserType::TypeOf(
                def_id,
                UserSubsts { substs, user_self_ty },
            ));
            debug!("write_user_type_annotation_from_substs: canonicalized={:?}", canonicalized);
            self.write_user_type_annotation(hir_id, canonicalized);
        }
    }

    pub fn write_user_type_annotation(
        &self,
        hir_id: hir::HirId,
        canonical_user_type_annotation: CanonicalUserType<'tcx>,
    ) {
        debug!(
            "write_user_type_annotation: hir_id={:?} canonical_user_type_annotation={:?} tag={}",
            hir_id,
            canonical_user_type_annotation,
            self.tag(),
        );

        if !canonical_user_type_annotation.is_identity() {
            self.tables
                .borrow_mut()
                .user_provided_types_mut()
                .insert(hir_id, canonical_user_type_annotation);
        } else {
            debug!("write_user_type_annotation: skipping identity substs");
        }
    }

    pub fn apply_adjustments(&self, expr: &hir::Expr<'_>, adj: Vec<Adjustment<'tcx>>) {
        debug!("apply_adjustments(expr={:?}, adj={:?})", expr, adj);

        if adj.is_empty() {
            return;
        }

        match self.tables.borrow_mut().adjustments_mut().entry(expr.hir_id) {
            Entry::Vacant(entry) => {
                entry.insert(adj);
            }
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
    fn instantiate_type_scheme<T>(&self, span: Span, substs: SubstsRef<'tcx>, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let value = value.subst(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, &value);
        debug!("instantiate_type_scheme(value={:?}, substs={:?}) = {:?}", value, substs, result);
        result
    }

    /// As `instantiate_type_scheme`, but for the bounds found in a
    /// generic type scheme.
    fn instantiate_bounds(
        &self,
        span: Span,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> (ty::InstantiatedPredicates<'tcx>, Vec<Span>) {
        let bounds = self.tcx.predicates_of(def_id);
        let spans: Vec<Span> = bounds.predicates.iter().map(|(_, span)| *span).collect();
        let result = bounds.instantiate(self.tcx, substs);
        let result = self.normalize_associated_types_in(span, &result);
        debug!(
            "instantiate_bounds(bounds={:?}, substs={:?}) = {:?}, {:?}",
            bounds, substs, result, spans,
        );
        (result, spans)
    }

    /// Replaces the opaque types from the given value with type variables,
    /// and records the `OpaqueTypeMap` for later use during writeback. See
    /// `InferCtxt::instantiate_opaque_types` for more details.
    fn instantiate_opaque_types_from_value<T: TypeFoldable<'tcx>>(
        &self,
        parent_id: hir::HirId,
        value: &T,
        value_span: Span,
    ) -> T {
        let parent_def_id = self.tcx.hir().local_def_id(parent_id);
        debug!(
            "instantiate_opaque_types_from_value(parent_def_id={:?}, value={:?})",
            parent_def_id, value
        );

        let (value, opaque_type_map) =
            self.register_infer_ok_obligations(self.instantiate_opaque_types(
                parent_def_id,
                self.body_id,
                self.param_env,
                value,
                value_span,
            ));

        let mut opaque_types = self.opaque_types.borrow_mut();
        let mut opaque_types_vars = self.opaque_types_vars.borrow_mut();
        for (ty, decl) in opaque_type_map {
            let _ = opaque_types.insert(ty, decl);
            let _ = opaque_types_vars.insert(decl.concrete_ty, decl.opaque_type);
        }

        value
    }

    fn normalize_associated_types_in<T>(&self, span: Span, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.inh.normalize_associated_types_in(span, self.body_id, self.param_env, value)
    }

    fn normalize_associated_types_in_as_infer_ok<T>(
        &self,
        span: Span,
        value: &T,
    ) -> InferOk<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        self.inh.partially_normalize_associated_types_in(span, self.body_id, self.param_env, value)
    }

    pub fn require_type_meets(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
        def_id: DefId,
    ) {
        self.register_bound(ty, def_id, traits::ObligationCause::new(span, self.body_id, code));
    }

    pub fn require_type_is_sized(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        if !ty.references_error() {
            let lang_item = self.tcx.require_lang_item(lang_items::SizedTraitLangItem, None);
            self.require_type_meets(ty, span, code, lang_item);
        }
    }

    pub fn require_type_is_sized_deferred(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        if !ty.references_error() {
            self.deferred_sized_obligations.borrow_mut().push((ty, span, code));
        }
    }

    pub fn register_bound(
        &self,
        ty: Ty<'tcx>,
        def_id: DefId,
        cause: traits::ObligationCause<'tcx>,
    ) {
        if !ty.references_error() {
            self.fulfillment_cx.borrow_mut().register_bound(
                self,
                self.param_env,
                ty,
                def_id,
                cause,
            );
        }
    }

    pub fn to_ty(&self, ast_t: &hir::Ty<'_>) -> Ty<'tcx> {
        let t = AstConv::ast_ty_to_ty(self, ast_t);
        self.register_wf_obligation(t, ast_t.span, traits::MiscObligation);
        t
    }

    pub fn to_ty_saving_user_provided_ty(&self, ast_ty: &hir::Ty<'_>) -> Ty<'tcx> {
        let ty = self.to_ty(ast_ty);
        debug!("to_ty_saving_user_provided_ty: ty={:?}", ty);

        if Self::can_contain_user_lifetime_bounds(ty) {
            let c_ty = self.infcx.canonicalize_response(&UserType::Ty(ty));
            debug!("to_ty_saving_user_provided_ty: c_ty={:?}", c_ty);
            self.tables.borrow_mut().user_provided_types_mut().insert(ast_ty.hir_id, c_ty);
        }

        ty
    }

    /// Returns the `DefId` of the constant parameter that the provided expression is a path to.
    pub fn const_param_def_id(&self, hir_c: &hir::AnonConst) -> Option<DefId> {
        AstConv::const_param_def_id(self, &self.tcx.hir().body(hir_c.body).value)
    }

    pub fn to_const(&self, ast_c: &hir::AnonConst, ty: Ty<'tcx>) -> &'tcx ty::Const<'tcx> {
        AstConv::ast_const_to_const(self, ast_c, ty)
    }

    // If the type given by the user has free regions, save it for later, since
    // NLL would like to enforce those. Also pass in types that involve
    // projections, since those can resolve to `'static` bounds (modulo #54940,
    // which hopefully will be fixed by the time you see this comment, dear
    // reader, although I have my doubts). Also pass in types with inference
    // types, because they may be repeated. Other sorts of things are already
    // sufficiently enforced with erased regions. =)
    fn can_contain_user_lifetime_bounds<T>(t: T) -> bool
    where
        T: TypeFoldable<'tcx>,
    {
        t.has_free_regions() || t.has_projections() || t.has_infer_types()
    }

    pub fn node_ty(&self, id: hir::HirId) -> Ty<'tcx> {
        match self.tables.borrow().node_types().get(id) {
            Some(&t) => t,
            None if self.is_tainted_by_errors() => self.tcx.types.err,
            None => {
                bug!(
                    "no type for node {}: {} in fcx {}",
                    id,
                    self.tcx.hir().node_to_string(id),
                    self.tag()
                );
            }
        }
    }

    /// Registers an obligation for checking later, during regionck, that the type `ty` must
    /// outlive the region `r`.
    pub fn register_wf_obligation(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        code: traits::ObligationCauseCode<'tcx>,
    ) {
        // WF obligations never themselves fail, so no real need to give a detailed cause:
        let cause = traits::ObligationCause::new(span, self.body_id, code);
        self.register_predicate(traits::Obligation::new(
            cause,
            self.param_env,
            ty::Predicate::WellFormed(ty),
        ));
    }

    /// Registers obligations that all types appearing in `substs` are well-formed.
    pub fn add_wf_bounds(&self, substs: SubstsRef<'tcx>, expr: &hir::Expr<'_>) {
        for ty in substs.types() {
            if !ty.references_error() {
                self.register_wf_obligation(ty, expr.span, traits::MiscObligation);
            }
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
    pub fn add_obligations_for_parameters(
        &self,
        cause: traits::ObligationCause<'tcx>,
        predicates: &ty::InstantiatedPredicates<'tcx>,
    ) {
        assert!(!predicates.has_escaping_bound_vars());

        debug!("add_obligations_for_parameters(predicates={:?})", predicates);

        for obligation in traits::predicates_for_generics(cause, self.param_env, predicates) {
            self.register_predicate(obligation);
        }
    }

    // FIXME(arielb1): use this instead of field.ty everywhere
    // Only for fields! Returns <none> for methods>
    // Indifferent to privacy flags
    pub fn field_ty(
        &self,
        span: Span,
        field: &'tcx ty::FieldDef,
        substs: SubstsRef<'tcx>,
    ) -> Ty<'tcx> {
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
        for (body_id, interior, kind) in generators.drain(..) {
            self.select_obligations_where_possible(false, |_| {});
            generator_interior::resolve_interior(self, def_id, body_id, interior, kind);
        }
    }

    // Tries to apply a fallback to `ty` if it is an unsolved variable.
    //
    // - Unconstrained ints are replaced with `i32`.
    //
    // - Unconstrained floats are replaced with with `f64`.
    //
    // - Non-numerics get replaced with `!` when `#![feature(never_type_fallback)]`
    //   is enabled. Otherwise, they are replaced with `()`.
    //
    // Fallback becomes very dubious if we have encountered type-checking errors.
    // In that case, fallback to Error.
    // The return value indicates whether fallback has occurred.
    fn fallback_if_possible(&self, ty: Ty<'tcx>, mode: FallbackMode) -> bool {
        use rustc::ty::error::UnconstrainedNumeric::Neither;
        use rustc::ty::error::UnconstrainedNumeric::{UnconstrainedFloat, UnconstrainedInt};

        assert!(ty.is_ty_infer());
        let fallback = match self.type_is_unconstrained_numeric(ty) {
            _ if self.is_tainted_by_errors() => self.tcx().types.err,
            UnconstrainedInt => self.tcx.types.i32,
            UnconstrainedFloat => self.tcx.types.f64,
            Neither if self.type_var_diverges(ty) => self.tcx.mk_diverging_default(),
            Neither => {
                // This type variable was created from the instantiation of an opaque
                // type. The fact that we're attempting to perform fallback for it
                // means that the function neither constrained it to a concrete
                // type, nor to the opaque type itself.
                //
                // For example, in this code:
                //
                //```
                // type MyType = impl Copy;
                // fn defining_use() -> MyType { true }
                // fn other_use() -> MyType { defining_use() }
                // ```
                //
                // `defining_use` will constrain the instantiated inference
                // variable to `bool`, while `other_use` will constrain
                // the instantiated inference variable to `MyType`.
                //
                // When we process opaque types during writeback, we
                // will handle cases like `other_use`, and not count
                // them as defining usages
                //
                // However, we also need to handle cases like this:
                //
                // ```rust
                // pub type Foo = impl Copy;
                // fn produce() -> Option<Foo> {
                //     None
                //  }
                //  ```
                //
                // In the above snippet, the inference variable created by
                // instantiating `Option<Foo>` will be completely unconstrained.
                // We treat this as a non-defining use by making the inference
                // variable fall back to the opaque type itself.
                if let FallbackMode::All = mode {
                    if let Some(opaque_ty) = self.opaque_types_vars.borrow().get(ty) {
                        debug!(
                            "fallback_if_possible: falling back opaque type var {:?} to {:?}",
                            ty, opaque_ty
                        );
                        *opaque_ty
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        };
        debug!("fallback_if_possible: defaulting `{:?}` to `{:?}`", ty, fallback);
        self.demand_eqtype(rustc_span::DUMMY_SP, ty, fallback);
        true
    }

    fn select_all_obligations_or_error(&self) {
        debug!("select_all_obligations_or_error");
        if let Err(errors) = self.fulfillment_cx.borrow_mut().select_all_or_error(&self) {
            self.report_fulfillment_errors(&errors, self.inh.body_id, false);
        }
    }

    /// Select as many obligations as we can at present.
    fn select_obligations_where_possible(
        &self,
        fallback_has_occurred: bool,
        mutate_fullfillment_errors: impl Fn(&mut Vec<traits::FulfillmentError<'tcx>>),
    ) {
        let result = self.fulfillment_cx.borrow_mut().select_where_possible(self);
        if let Err(mut errors) = result {
            mutate_fullfillment_errors(&mut errors);
            self.report_fulfillment_errors(&errors, self.inh.body_id, fallback_has_occurred);
        }
    }

    /// For the overloaded place expressions (`*x`, `x[3]`), the trait
    /// returns a type of `&T`, but the actual type we assign to the
    /// *expression* is `T`. So this function just peels off the return
    /// type by one layer to yield `T`.
    fn make_overloaded_place_return_type(
        &self,
        method: MethodCallee<'tcx>,
    ) -> ty::TypeAndMut<'tcx> {
        // extract method return type, which will be &T;
        let ret_ty = method.sig.output();

        // method returns &T, but the type as visible to user is T, so deref
        ret_ty.builtin_deref(true).unwrap()
    }

    fn lookup_indexing(
        &self,
        expr: &hir::Expr<'_>,
        base_expr: &'tcx hir::Expr<'tcx>,
        base_ty: Ty<'tcx>,
        idx_ty: Ty<'tcx>,
        needs: Needs,
    ) -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)> {
        // FIXME(#18741) -- this is almost but not quite the same as the
        // autoderef that normal method probing does. They could likely be
        // consolidated.

        let mut autoderef = self.autoderef(base_expr.span, base_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_index_step(expr, base_expr, &autoderef, needs, idx_ty);
        }
        autoderef.finalize(self);
        result
    }

    /// To type-check `base_expr[index_expr]`, we progressively autoderef
    /// (and otherwise adjust) `base_expr`, looking for a type which either
    /// supports builtin indexing or overloaded indexing.
    /// This loop implements one step in that search; the autoderef loop
    /// is implemented by `lookup_indexing`.
    fn try_index_step(
        &self,
        expr: &hir::Expr<'_>,
        base_expr: &hir::Expr<'_>,
        autoderef: &Autoderef<'a, 'tcx>,
        needs: Needs,
        index_ty: Ty<'tcx>,
    ) -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)> {
        let adjusted_ty = autoderef.unambiguous_final_ty(self);
        debug!(
            "try_index_step(expr={:?}, base_expr={:?}, adjusted_ty={:?}, \
                               index_ty={:?})",
            expr, base_expr, adjusted_ty, index_ty
        );

        for &unsize in &[false, true] {
            let mut self_ty = adjusted_ty;
            if unsize {
                // We only unsize arrays here.
                if let ty::Array(element_ty, _) = adjusted_ty.kind {
                    self_ty = self.tcx.mk_slice(element_ty);
                } else {
                    continue;
                }
            }

            // If some lookup succeeds, write callee into table and extract index/element
            // type from the method signature.
            // If some lookup succeeded, install method in table
            let input_ty = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::AutoDeref,
                span: base_expr.span,
            });
            let method = self.try_overloaded_place_op(
                expr.span,
                self_ty,
                &[input_ty],
                needs,
                PlaceOp::Index,
            );

            let result = method.map(|ok| {
                debug!("try_index_step: success, using overloaded indexing");
                let method = self.register_infer_ok_obligations(ok);

                let mut adjustments = autoderef.adjust_steps(self, needs);
                if let ty::Ref(region, _, r_mutbl) = method.sig.inputs()[0].kind {
                    let mutbl = match r_mutbl {
                        hir::Mutability::Not => AutoBorrowMutability::Not,
                        hir::Mutability::Mut => AutoBorrowMutability::Mut {
                            // Indexing can be desugared to a method call,
                            // so maybe we could use two-phase here.
                            // See the documentation of AllowTwoPhase for why that's
                            // not the case today.
                            allow_two_phase_borrow: AllowTwoPhase::No,
                        },
                    };
                    adjustments.push(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                        target: self
                            .tcx
                            .mk_ref(region, ty::TypeAndMut { mutbl: r_mutbl, ty: adjusted_ty }),
                    });
                }
                if unsize {
                    adjustments.push(Adjustment {
                        kind: Adjust::Pointer(PointerCast::Unsize),
                        target: method.sig.inputs()[0],
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
            (PlaceOp::Deref, false) => (self.tcx.lang_items().deref_trait(), sym::deref),
            (PlaceOp::Deref, true) => (self.tcx.lang_items().deref_mut_trait(), sym::deref_mut),
            (PlaceOp::Index, false) => (self.tcx.lang_items().index_trait(), sym::index),
            (PlaceOp::Index, true) => (self.tcx.lang_items().index_mut_trait(), sym::index_mut),
        };
        (tr, ast::Ident::with_dummy_span(name))
    }

    fn try_overloaded_place_op(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
        arg_tys: &[Ty<'tcx>],
        needs: Needs,
        op: PlaceOp,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        debug!("try_overloaded_place_op({:?},{:?},{:?},{:?})", span, base_ty, needs, op);

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

    fn check_method_argument_types(
        &self,
        sp: Span,
        expr: &'tcx hir::Expr<'tcx>,
        method: Result<MethodCallee<'tcx>, ()>,
        args_no_rcvr: &'tcx [hir::Expr<'tcx>],
        tuple_arguments: TupleArgumentsFlag,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let has_error = match method {
            Ok(method) => method.substs.references_error() || method.sig.references_error(),
            Err(_) => true,
        };
        if has_error {
            let err_inputs = self.err_args(args_no_rcvr.len());

            let err_inputs = match tuple_arguments {
                DontTupleArguments => err_inputs,
                TupleArguments => vec![self.tcx.intern_tup(&err_inputs[..])],
            };

            self.check_argument_types(
                sp,
                expr,
                &err_inputs[..],
                &[],
                args_no_rcvr,
                false,
                tuple_arguments,
                None,
            );
            return self.tcx.types.err;
        }

        let method = method.unwrap();
        // HACK(eddyb) ignore self in the definition (see above).
        let expected_arg_tys = self.expected_inputs_for_expected_output(
            sp,
            expected,
            method.sig.output(),
            &method.sig.inputs()[1..],
        );
        self.check_argument_types(
            sp,
            expr,
            &method.sig.inputs()[1..],
            &expected_arg_tys[..],
            args_no_rcvr,
            method.sig.c_variadic,
            tuple_arguments,
            self.tcx.hir().span_if_local(method.def_id),
        );
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
        match self_ty.kind {
            ty::Infer(ty::TyVar(found_vid)) => {
                // FIXME: consider using `sub_root_var` here so we
                // can see through subtyping.
                let found_vid = self.root_var(found_vid);
                debug!("self_type_matches_expected_vid - found_vid={:?}", found_vid);
                expected_vid == found_vid
            }
            _ => false,
        }
    }

    fn obligations_for_self_ty<'b>(
        &'b self,
        self_ty: ty::TyVid,
    ) -> impl Iterator<Item = (ty::PolyTraitRef<'tcx>, traits::PredicateObligation<'tcx>)>
    + Captures<'tcx>
    + 'b {
        // FIXME: consider using `sub_root_var` here so we
        // can see through subtyping.
        let ty_var_root = self.root_var(self_ty);
        debug!(
            "obligations_for_self_ty: self_ty={:?} ty_var_root={:?} pending_obligations={:?}",
            self_ty,
            ty_var_root,
            self.fulfillment_cx.borrow().pending_obligations()
        );

        self.fulfillment_cx
            .borrow()
            .pending_obligations()
            .into_iter()
            .filter_map(move |obligation| match obligation.predicate {
                ty::Predicate::Projection(ref data) => {
                    Some((data.to_poly_trait_ref(self.tcx), obligation))
                }
                ty::Predicate::Trait(ref data, _) => Some((data.to_poly_trait_ref(), obligation)),
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
            })
            .filter(move |(tr, _)| self.self_type_matches_expected_vid(*tr, ty_var_root))
    }

    fn type_var_is_sized(&self, self_ty: ty::TyVid) -> bool {
        self.obligations_for_self_ty(self_ty)
            .any(|(tr, _)| Some(tr.def_id()) == self.tcx.lang_items().sized_trait())
    }

    /// Generic function that factors out common logic from function calls,
    /// method calls and overloaded operators.
    fn check_argument_types(
        &self,
        sp: Span,
        expr: &'tcx hir::Expr<'tcx>,
        fn_inputs: &[Ty<'tcx>],
        expected_arg_tys: &[Ty<'tcx>],
        args: &'tcx [hir::Expr<'tcx>],
        c_variadic: bool,
        tuple_arguments: TupleArgumentsFlag,
        def_span: Option<Span>,
    ) {
        let tcx = self.tcx;
        // Grab the argument types, supplying fresh type variables
        // if the wrong number of arguments were supplied
        let supplied_arg_count = if tuple_arguments == DontTupleArguments { args.len() } else { 1 };

        // All the input types from the fn signature must outlive the call
        // so as to validate implied bounds.
        for (fn_input_ty, arg_expr) in fn_inputs.iter().zip(args.iter()) {
            self.register_wf_obligation(fn_input_ty, arg_expr.span, traits::MiscObligation);
        }

        let expected_arg_count = fn_inputs.len();

        let param_count_error = |expected_count: usize,
                                 arg_count: usize,
                                 error_code: &str,
                                 c_variadic: bool,
                                 sugg_unit: bool| {
            let (span, start_span, args) = match &expr.kind {
                hir::ExprKind::Call(hir::Expr { span, .. }, args) => (*span, *span, &args[..]),
                hir::ExprKind::MethodCall(path_segment, span, args) => (
                    *span,
                    // `sp` doesn't point at the whole `foo.bar()`, only at `bar`.
                    path_segment
                        .args
                        .and_then(|args| args.args.iter().last())
                        // Account for `foo.bar::<T>()`.
                        .map(|arg| {
                            // Skip the closing `>`.
                            tcx.sess
                                .source_map()
                                .next_point(tcx.sess.source_map().next_point(arg.span()))
                        })
                        .unwrap_or(*span),
                    &args[1..], // Skip the receiver.
                ),
                k => span_bug!(sp, "checking argument types on a non-call: `{:?}`", k),
            };
            let arg_spans = if args.is_empty() {
                // foo()
                // ^^^-- supplied 0 arguments
                // |
                // expected 2 arguments
                vec![tcx.sess.source_map().next_point(start_span).with_hi(sp.hi())]
            } else {
                // foo(1, 2, 3)
                // ^^^ -  -  - supplied 3 arguments
                // |
                // expected 2 arguments
                args.iter().map(|arg| arg.span).collect::<Vec<Span>>()
            };

            let mut err = tcx.sess.struct_span_err_with_code(
                span,
                &format!(
                    "this function takes {}{} but {} {} supplied",
                    if c_variadic { "at least " } else { "" },
                    potentially_plural_count(expected_count, "argument"),
                    potentially_plural_count(arg_count, "argument"),
                    if arg_count == 1 { "was" } else { "were" }
                ),
                DiagnosticId::Error(error_code.to_owned()),
            );
            let label = format!("supplied {}", potentially_plural_count(arg_count, "argument"));
            for (i, span) in arg_spans.into_iter().enumerate() {
                err.span_label(
                    span,
                    if arg_count == 0 || i + 1 == arg_count { &label } else { "" },
                );
            }

            if let Some(def_s) = def_span.map(|sp| tcx.sess.source_map().def_span(sp)) {
                err.span_label(def_s, "defined here");
            }
            if sugg_unit {
                let sugg_span = tcx.sess.source_map().end_point(expr.span);
                // remove closing `)` from the span
                let sugg_span = sugg_span.shrink_to_lo();
                err.span_suggestion(
                    sugg_span,
                    "expected the unit value `()`; create it with empty parentheses",
                    String::from("()"),
                    Applicability::MachineApplicable,
                );
            } else {
                err.span_label(
                    span,
                    format!(
                        "expected {}{}",
                        if c_variadic { "at least " } else { "" },
                        potentially_plural_count(expected_count, "argument")
                    ),
                );
            }
            err.emit();
        };

        let mut expected_arg_tys = expected_arg_tys.to_vec();

        let formal_tys = if tuple_arguments == TupleArguments {
            let tuple_type = self.structurally_resolved_type(sp, fn_inputs[0]);
            match tuple_type.kind {
                ty::Tuple(arg_types) if arg_types.len() != args.len() => {
                    param_count_error(arg_types.len(), args.len(), "E0057", false, false);
                    expected_arg_tys = vec![];
                    self.err_args(args.len())
                }
                ty::Tuple(arg_types) => {
                    expected_arg_tys = match expected_arg_tys.get(0) {
                        Some(&ty) => match ty.kind {
                            ty::Tuple(ref tys) => tys.iter().map(|k| k.expect_ty()).collect(),
                            _ => vec![],
                        },
                        None => vec![],
                    };
                    arg_types.iter().map(|k| k.expect_ty()).collect()
                }
                _ => {
                    struct_span_err!(
                        tcx.sess,
                        sp,
                        E0059,
                        "cannot use call notation; the first type parameter \
                         for the function trait is neither a tuple nor unit"
                    )
                    .emit();
                    expected_arg_tys = vec![];
                    self.err_args(args.len())
                }
            }
        } else if expected_arg_count == supplied_arg_count {
            fn_inputs.to_vec()
        } else if c_variadic {
            if supplied_arg_count >= expected_arg_count {
                fn_inputs.to_vec()
            } else {
                param_count_error(expected_arg_count, supplied_arg_count, "E0060", true, false);
                expected_arg_tys = vec![];
                self.err_args(supplied_arg_count)
            }
        } else {
            // is the missing argument of type `()`?
            let sugg_unit = if expected_arg_tys.len() == 1 && supplied_arg_count == 0 {
                self.resolve_vars_if_possible(&expected_arg_tys[0]).is_unit()
            } else if fn_inputs.len() == 1 && supplied_arg_count == 0 {
                self.resolve_vars_if_possible(&fn_inputs[0]).is_unit()
            } else {
                false
            };
            param_count_error(expected_arg_count, supplied_arg_count, "E0061", false, sugg_unit);

            expected_arg_tys = vec![];
            self.err_args(supplied_arg_count)
        };

        debug!(
            "check_argument_types: formal_tys={:?}",
            formal_tys.iter().map(|t| self.ty_to_string(*t)).collect::<Vec<String>>()
        );

        // If there is no expectation, expect formal_tys.
        let expected_arg_tys =
            if !expected_arg_tys.is_empty() { expected_arg_tys } else { formal_tys.clone() };

        let mut final_arg_types: Vec<(usize, Ty<'_>, Ty<'_>)> = vec![];

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
                self.select_obligations_where_possible(false, |errors| {
                    self.point_at_type_arg_instead_of_call_if_possible(errors, expr);
                    self.point_at_arg_instead_of_call_if_possible(
                        errors,
                        &final_arg_types[..],
                        sp,
                        &args,
                    );
                })
            }

            // For C-variadic functions, we don't have a declared type for all of
            // the arguments hence we only do our usual type checking with
            // the arguments who's types we do know.
            let t = if c_variadic {
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
                    self.warn_if_unreachable(arg.hir_id, arg.span, "expression");
                }

                let is_closure = match arg.kind {
                    ExprKind::Closure(..) => true,
                    _ => false,
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
                final_arg_types.push((i, checked_ty, coerce_ty));

                // 3. Relate the expected type and the formal one,
                //    if the expected type was used for the coercion.
                self.demand_suptype(arg.span, formal_ty, coerce_ty);
            }
        }

        // We also need to make sure we at least write the ty of the other
        // arguments which we skipped above.
        if c_variadic {
            fn variadic_error<'tcx>(s: &Session, span: Span, t: Ty<'tcx>, cast_ty: &str) {
                use crate::structured_errors::{StructuredDiagnostic, VariadicError};
                VariadicError::new(s, span, t, cast_ty).diagnostic().emit();
            }

            for arg in args.iter().skip(expected_arg_count) {
                let arg_ty = self.check_expr(&arg);

                // There are a few types which get autopromoted when passed via varargs
                // in C but we just error out instead and require explicit casts.
                let arg_ty = self.structurally_resolved_type(arg.span, arg_ty);
                match arg_ty.kind {
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
                        let ptr_ty = self.resolve_vars_if_possible(&ptr_ty);
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

    /// Given a vec of evaluated `FulfillmentError`s and an `fn` call argument expressions, we walk
    /// the checked and coerced types for each argument to see if any of the `FulfillmentError`s
    /// reference a type argument. The reason to walk also the checked type is that the coerced type
    /// can be not easily comparable with predicate type (because of coercion). If the types match
    /// for either checked or coerced type, and there's only *one* argument that does, we point at
    /// the corresponding argument's expression span instead of the `fn` call path span.
    fn point_at_arg_instead_of_call_if_possible(
        &self,
        errors: &mut Vec<traits::FulfillmentError<'_>>,
        final_arg_types: &[(usize, Ty<'tcx>, Ty<'tcx>)],
        call_sp: Span,
        args: &'tcx [hir::Expr<'tcx>],
    ) {
        // We *do not* do this for desugared call spans to keep good diagnostics when involving
        // the `?` operator.
        if call_sp.desugaring_kind().is_some() {
            return;
        }

        for error in errors {
            // Only if the cause is somewhere inside the expression we want try to point at arg.
            // Otherwise, it means that the cause is somewhere else and we should not change
            // anything because we can break the correct span.
            if !call_sp.contains(error.obligation.cause.span) {
                continue;
            }

            if let ty::Predicate::Trait(predicate, _) = error.obligation.predicate {
                // Collect the argument position for all arguments that could have caused this
                // `FulfillmentError`.
                let mut referenced_in = final_arg_types
                    .iter()
                    .map(|(i, checked_ty, _)| (i, checked_ty))
                    .chain(final_arg_types.iter().map(|(i, _, coerced_ty)| (i, coerced_ty)))
                    .flat_map(|(i, ty)| {
                        let ty = self.resolve_vars_if_possible(ty);
                        // We walk the argument type because the argument's type could have
                        // been `Option<T>`, but the `FulfillmentError` references `T`.
                        ty.walk()
                            .filter(|&ty| ty == predicate.skip_binder().self_ty())
                            .map(move |_| *i)
                    })
                    .collect::<Vec<_>>();

                // Both checked and coerced types could have matched, thus we need to remove
                // duplicates.
                referenced_in.dedup();

                if let (Some(ref_in), None) = (referenced_in.pop(), referenced_in.pop()) {
                    // We make sure that only *one* argument matches the obligation failure
                    // and we assign the obligation's span to its expression's.
                    error.obligation.cause.span = args[ref_in].span;
                    error.points_at_arg_span = true;
                }
            }
        }
    }

    /// Given a vec of evaluated `FulfillmentError`s and an `fn` call expression, we walk the
    /// `PathSegment`s and resolve their type parameters to see if any of the `FulfillmentError`s
    /// were caused by them. If they were, we point at the corresponding type argument's span
    /// instead of the `fn` call path span.
    fn point_at_type_arg_instead_of_call_if_possible(
        &self,
        errors: &mut Vec<traits::FulfillmentError<'_>>,
        call_expr: &'tcx hir::Expr<'tcx>,
    ) {
        if let hir::ExprKind::Call(path, _) = &call_expr.kind {
            if let hir::ExprKind::Path(qpath) = &path.kind {
                if let hir::QPath::Resolved(_, path) = &qpath {
                    for error in errors {
                        if let ty::Predicate::Trait(predicate, _) = error.obligation.predicate {
                            // If any of the type arguments in this path segment caused the
                            // `FullfillmentError`, point at its span (#61860).
                            for arg in path
                                .segments
                                .iter()
                                .filter_map(|seg| seg.args.as_ref())
                                .flat_map(|a| a.args.iter())
                            {
                                if let hir::GenericArg::Type(hir_ty) = &arg {
                                    if let hir::TyKind::Path(hir::QPath::TypeRelative(..)) =
                                        &hir_ty.kind
                                    {
                                        // Avoid ICE with associated types. As this is best
                                        // effort only, it's ok to ignore the case. It
                                        // would trigger in `is_send::<T::AssocType>();`
                                        // from `typeck-default-trait-impl-assoc-type.rs`.
                                    } else {
                                        let ty = AstConv::ast_ty_to_ty(self, hir_ty);
                                        let ty = self.resolve_vars_if_possible(&ty);
                                        if ty == predicate.skip_binder().self_ty() {
                                            error.obligation.cause.span = hir_ty.span;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // AST fragment checking
    fn check_lit(&self, lit: &hir::Lit, expected: Expectation<'tcx>) -> Ty<'tcx> {
        let tcx = self.tcx;

        match lit.node {
            ast::LitKind::Str(..) => tcx.mk_static_str(),
            ast::LitKind::ByteStr(ref v) => {
                tcx.mk_imm_ref(tcx.lifetimes.re_static, tcx.mk_array(tcx.types.u8, v.len() as u64))
            }
            ast::LitKind::Byte(_) => tcx.types.u8,
            ast::LitKind::Char(_) => tcx.types.char,
            ast::LitKind::Int(_, ast::LitIntType::Signed(t)) => tcx.mk_mach_int(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsigned(t)) => tcx.mk_mach_uint(t),
            ast::LitKind::Int(_, ast::LitIntType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| match ty.kind {
                    ty::Int(_) | ty::Uint(_) => Some(ty),
                    ty::Char => Some(tcx.types.u8),
                    ty::RawPtr(..) => Some(tcx.types.usize),
                    ty::FnDef(..) | ty::FnPtr(_) => Some(tcx.types.usize),
                    _ => None,
                });
                opt_ty.unwrap_or_else(|| self.next_int_var())
            }
            ast::LitKind::Float(_, ast::LitFloatType::Suffixed(t)) => tcx.mk_mach_float(t),
            ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) => {
                let opt_ty = expected.to_option(self).and_then(|ty| match ty.kind {
                    ty::Float(_) => Some(ty),
                    _ => None,
                });
                opt_ty.unwrap_or_else(|| self.next_float_var())
            }
            ast::LitKind::Bool(_) => tcx.types.bool,
            ast::LitKind::Err(_) => tcx.types.err,
        }
    }

    // Determine the `Self` type, using fresh variables for all variables
    // declared on the impl declaration e.g., `impl<A,B> for Vec<(A,B)>`
    // would return `($0, $1)` where `$0` and `$1` are freshly instantiated type
    // variables.
    pub fn impl_self_ty(
        &self,
        span: Span, // (potential) receiver for this impl
        did: DefId,
    ) -> TypeAndSubsts<'tcx> {
        let ity = self.tcx.type_of(did);
        debug!("impl_self_ty: ity={:?}", ity);

        let substs = self.fresh_substs_for_item(span, did);
        let substd_ty = self.instantiate_type_scheme(span, &substs, &ity);

        TypeAndSubsts { substs, ty: substd_ty }
    }

    /// Unifies the output type with the expected type early, for more coercions
    /// and forward type information on the input expressions.
    fn expected_inputs_for_expected_output(
        &self,
        call_span: Span,
        expected_ret: Expectation<'tcx>,
        formal_ret: Ty<'tcx>,
        formal_args: &[Ty<'tcx>],
    ) -> Vec<Ty<'tcx>> {
        let formal_ret = self.resolve_vars_with_obligations(formal_ret);
        let ret_ty = match expected_ret.only_has_type(self) {
            Some(ret) => ret,
            None => return Vec::new(),
        };
        let expect_args = self
            .fudge_inference_if_ok(|| {
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
                        })
                        .map_err(|_| ())?;
                    }
                    Err(_) => return Err(()),
                }

                // Record all the argument types, with the substitutions
                // produced from the above subtyping unification.
                Ok(formal_args.iter().map(|ty| self.resolve_vars_if_possible(ty)).collect())
            })
            .unwrap_or_default();
        debug!(
            "expected_inputs_for_expected_output(formal={:?} -> {:?}, expected={:?} -> {:?})",
            formal_args, formal_ret, expect_args, expected_ret
        );
        expect_args
    }

    pub fn check_struct_path(
        &self,
        qpath: &QPath<'_>,
        hir_id: hir::HirId,
    ) -> Option<(&'tcx ty::VariantDef, Ty<'tcx>)> {
        let path_span = match *qpath {
            QPath::Resolved(_, ref path) => path.span,
            QPath::TypeRelative(ref qself, _) => qself.span,
        };
        let (def, ty) = self.finish_resolving_struct_path(qpath, path_span, hir_id);
        let variant = match def {
            Res::Err => {
                self.set_tainted_by_errors();
                return None;
            }
            Res::Def(DefKind::Variant, _) => match ty.kind {
                ty::Adt(adt, substs) => Some((adt.variant_of_res(def), adt.did, substs)),
                _ => bug!("unexpected type: {:?}", ty),
            },
            Res::Def(DefKind::Struct, _)
            | Res::Def(DefKind::Union, _)
            | Res::Def(DefKind::TyAlias, _)
            | Res::Def(DefKind::AssocTy, _)
            | Res::SelfTy(..) => match ty.kind {
                ty::Adt(adt, substs) if !adt.is_enum() => {
                    Some((adt.non_enum_variant(), adt.did, substs))
                }
                _ => None,
            },
            _ => bug!("unexpected definition: {:?}", def),
        };

        if let Some((variant, did, substs)) = variant {
            debug!("check_struct_path: did={:?} substs={:?}", did, substs);
            self.write_user_type_annotation_from_substs(hir_id, did, substs, None);

            // Check bounds on type arguments used in the path.
            let (bounds, _) = self.instantiate_bounds(path_span, did, substs);
            let cause =
                traits::ObligationCause::new(path_span, self.body_id, traits::ItemObligation(did));
            self.add_obligations_for_parameters(cause, &bounds);

            Some((variant, ty))
        } else {
            struct_span_err!(
                self.tcx.sess,
                path_span,
                E0071,
                "expected struct, variant or union type, found {}",
                ty.sort_string(self.tcx)
            )
            .span_label(path_span, "not a struct")
            .emit();
            None
        }
    }

    // Finish resolving a path in a struct expression or pattern `S::A { .. }` if necessary.
    // The newly resolved definition is written into `type_dependent_defs`.
    fn finish_resolving_struct_path(
        &self,
        qpath: &QPath<'_>,
        path_span: Span,
        hir_id: hir::HirId,
    ) -> (Res, Ty<'tcx>) {
        match *qpath {
            QPath::Resolved(ref maybe_qself, ref path) => {
                let self_ty = maybe_qself.as_ref().map(|qself| self.to_ty(qself));
                let ty = AstConv::res_to_ty(self, self_ty, path, true);
                (path.res, ty)
            }
            QPath::TypeRelative(ref qself, ref segment) => {
                let ty = self.to_ty(qself);

                let res = if let hir::TyKind::Path(QPath::Resolved(_, ref path)) = qself.kind {
                    path.res
                } else {
                    Res::Err
                };
                let result =
                    AstConv::associated_path_to_ty(self, hir_id, path_span, ty, res, segment, true);
                let ty = result.map(|(ty, _, _)| ty).unwrap_or(self.tcx().types.err);
                let result = result.map(|(_, kind, def_id)| (kind, def_id));

                // Write back the new resolution.
                self.write_resolution(hir_id, result);

                (result.map(|(kind, def_id)| Res::Def(kind, def_id)).unwrap_or(Res::Err), ty)
            }
        }
    }

    /// Resolves an associated value path into a base type and associated constant, or method
    /// resolution. The newly resolved definition is written into `type_dependent_defs`.
    pub fn resolve_ty_and_res_ufcs<'b>(
        &self,
        qpath: &'b QPath<'b>,
        hir_id: hir::HirId,
        span: Span,
    ) -> (Res, Option<Ty<'tcx>>, &'b [hir::PathSegment<'b>]) {
        debug!("resolve_ty_and_res_ufcs: qpath={:?} hir_id={:?} span={:?}", qpath, hir_id, span);
        let (ty, qself, item_segment) = match *qpath {
            QPath::Resolved(ref opt_qself, ref path) => {
                return (
                    path.res,
                    opt_qself.as_ref().map(|qself| self.to_ty(qself)),
                    &path.segments[..],
                );
            }
            QPath::TypeRelative(ref qself, ref segment) => (self.to_ty(qself), qself, segment),
        };
        if let Some(&cached_result) = self.tables.borrow().type_dependent_defs().get(hir_id) {
            // Return directly on cache hit. This is useful to avoid doubly reporting
            // errors with default match binding modes. See #44614.
            let def =
                cached_result.map(|(kind, def_id)| Res::Def(kind, def_id)).unwrap_or(Res::Err);
            return (def, Some(ty), slice::from_ref(&**item_segment));
        }
        let item_name = item_segment.ident;
        let result = self.resolve_ufcs(span, item_name, ty, hir_id).or_else(|error| {
            let result = match error {
                method::MethodError::PrivateMatch(kind, def_id, _) => Ok((kind, def_id)),
                _ => Err(ErrorReported),
            };
            if item_name.name != kw::Invalid {
                self.report_method_error(
                    span,
                    ty,
                    item_name,
                    SelfSource::QPath(qself),
                    error,
                    None,
                )
                .map(|mut e| e.emit());
            }
            result
        });

        // Write back the new resolution.
        self.write_resolution(hir_id, result);
        (
            result.map(|(kind, def_id)| Res::Def(kind, def_id)).unwrap_or(Res::Err),
            Some(ty),
            slice::from_ref(&**item_segment),
        )
    }

    pub fn check_decl_initializer(
        &self,
        local: &'tcx hir::Local<'tcx>,
        init: &'tcx hir::Expr<'tcx>,
    ) -> Ty<'tcx> {
        // FIXME(tschottdorf): `contains_explicit_ref_binding()` must be removed
        // for #42640 (default match binding modes).
        //
        // See #44848.
        let ref_bindings = local.pat.contains_explicit_ref_binding();

        let local_ty = self.local_ty(init.span, local.hir_id).revealed_ty;
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

    /// Type check a `let` statement.
    pub fn check_decl_local(&self, local: &'tcx hir::Local<'tcx>) {
        // Determine and write the type which we'll check the pattern against.
        let ty = self.local_ty(local.span, local.hir_id).decl_ty;
        self.write_ty(local.hir_id, ty);

        // Type check the initializer.
        if let Some(ref init) = local.init {
            let init_ty = self.check_decl_initializer(local, &init);
            self.overwrite_local_ty_if_err(local, ty, init_ty);
        }

        // Does the expected pattern type originate from an expression and what is the span?
        let (origin_expr, ty_span) = match (local.ty, local.init) {
            (Some(ty), _) => (false, Some(ty.span)), // Bias towards the explicit user type.
            (_, Some(init)) => (true, Some(init.span)), // No explicit type; so use the scrutinee.
            _ => (false, None), // We have `let $pat;`, so the expected type is unconstrained.
        };

        // Type check the pattern. Override if necessary to avoid knock-on errors.
        self.check_pat_top(&local.pat, ty, ty_span, origin_expr);
        let pat_ty = self.node_ty(local.pat.hir_id);
        self.overwrite_local_ty_if_err(local, ty, pat_ty);
    }

    fn overwrite_local_ty_if_err(
        &self,
        local: &'tcx hir::Local<'tcx>,
        decl_ty: Ty<'tcx>,
        ty: Ty<'tcx>,
    ) {
        if ty.references_error() {
            // Override the types everywhere with `types.err` to avoid knock on errors.
            self.write_ty(local.hir_id, ty);
            self.write_ty(local.pat.hir_id, ty);
            let local_ty = LocalTy { decl_ty, revealed_ty: ty };
            self.locals.borrow_mut().insert(local.hir_id, local_ty);
            self.locals.borrow_mut().insert(local.pat.hir_id, local_ty);
        }
    }

    fn suggest_semicolon_at_end(&self, span: Span, err: &mut DiagnosticBuilder<'_>) {
        err.span_suggestion_short(
            span.shrink_to_hi(),
            "consider using a semicolon here",
            ";".to_string(),
            Applicability::MachineApplicable,
        );
    }

    pub fn check_stmt(&self, stmt: &'tcx hir::Stmt<'tcx>) {
        // Don't do all the complex logic below for `DeclItem`.
        match stmt.kind {
            hir::StmtKind::Item(..) => return,
            hir::StmtKind::Local(..) | hir::StmtKind::Expr(..) | hir::StmtKind::Semi(..) => {}
        }

        self.warn_if_unreachable(stmt.hir_id, stmt.span, "statement");

        // Hide the outer diverging and `has_errors` flags.
        let old_diverges = self.diverges.replace(Diverges::Maybe);
        let old_has_errors = self.has_errors.replace(false);

        match stmt.kind {
            hir::StmtKind::Local(ref l) => {
                self.check_decl_local(&l);
            }
            // Ignore for now.
            hir::StmtKind::Item(_) => {}
            hir::StmtKind::Expr(ref expr) => {
                // Check with expected type of `()`.
                self.check_expr_has_type_or_error(&expr, self.tcx.mk_unit(), |err| {
                    self.suggest_semicolon_at_end(expr.span, err);
                });
            }
            hir::StmtKind::Semi(ref expr) => {
                self.check_expr(&expr);
            }
        }

        // Combine the diverging and `has_error` flags.
        self.diverges.set(self.diverges.get() | old_diverges);
        self.has_errors.set(self.has_errors.get() | old_has_errors);
    }

    pub fn check_block_no_value(&self, blk: &'tcx hir::Block<'tcx>) {
        let unit = self.tcx.mk_unit();
        let ty = self.check_block_with_expected(blk, ExpectHasType(unit));

        // if the block produces a `!` value, that can always be
        // (effectively) coerced to unit.
        if !ty.is_never() {
            self.demand_suptype(blk.span, unit, ty);
        }
    }

    /// If `expr` is a `match` expression that has only one non-`!` arm, use that arm's tail
    /// expression's `Span`, otherwise return `expr.span`. This is done to give better errors
    /// when given code like the following:
    /// ```text
    /// if false { return 0i32; } else { 1u32 }
    /// //                               ^^^^ point at this instead of the whole `if` expression
    /// ```
    fn get_expr_coercion_span(&self, expr: &hir::Expr<'_>) -> rustc_span::Span {
        if let hir::ExprKind::Match(_, arms, _) = &expr.kind {
            let arm_spans: Vec<Span> = arms
                .iter()
                .filter_map(|arm| {
                    self.in_progress_tables
                        .and_then(|tables| tables.borrow().node_type_opt(arm.body.hir_id))
                        .and_then(|arm_ty| {
                            if arm_ty.is_never() {
                                None
                            } else {
                                Some(match &arm.body.kind {
                                    // Point at the tail expression when possible.
                                    hir::ExprKind::Block(block, _) => {
                                        block.expr.as_ref().map(|e| e.span).unwrap_or(block.span)
                                    }
                                    _ => arm.body.span,
                                })
                            }
                        })
                })
                .collect();
            if arm_spans.len() == 1 {
                return arm_spans[0];
            }
        }
        expr.span
    }

    fn check_block_with_expected(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
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
            let tail_expr: &[&hir::Expr<'_>] = match tail_expr {
                Some(e) => slice::from_ref(e),
                None => &[],
            };
            CoerceMany::with_coercion_sites(coerce_to_ty, tail_expr)
        };

        let prev_diverges = self.diverges.get();
        let ctxt = BreakableCtxt { coerce: Some(coerce), may_break: false };

        let (ctxt, ()) = self.with_breakable_ctxt(blk.hir_id, ctxt, || {
            for s in blk.stmts {
                self.check_stmt(s);
            }

            // check the tail expression **without** holding the
            // `enclosing_breakables` lock below.
            let tail_expr_ty = tail_expr.map(|t| self.check_expr_with_expectation(t, expected));

            let mut enclosing_breakables = self.enclosing_breakables.borrow_mut();
            let ctxt = enclosing_breakables.find_breakable(blk.hir_id);
            let coerce = ctxt.coerce.as_mut().unwrap();
            if let Some(tail_expr_ty) = tail_expr_ty {
                let tail_expr = tail_expr.unwrap();
                let span = self.get_expr_coercion_span(tail_expr);
                let cause = self.cause(span, ObligationCauseCode::BlockTailExpression(blk.hir_id));
                coerce.coerce(self, &cause, tail_expr, tail_expr_ty);
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
                if !self.diverges.get().is_always() {
                    // #50009 -- Do not point at the entire fn block span, point at the return type
                    // span, as it is the cause of the requirement, and
                    // `consider_hint_about_removing_semicolon` will point at the last expression
                    // if it were a relevant part of the error. This improves usability in editors
                    // that highlight errors inline.
                    let mut sp = blk.span;
                    let mut fn_span = None;
                    if let Some((decl, ident)) = self.get_parent_fn_decl(blk.hir_id) {
                        let ret_sp = decl.output.span();
                        if let Some(block_sp) = self.parent_item_span(blk.hir_id) {
                            // HACK: on some cases (`ui/liveness/liveness-issue-2163.rs`) the
                            // output would otherwise be incorrect and even misleading. Make sure
                            // the span we're aiming at correspond to a `fn` body.
                            if block_sp == blk.span {
                                sp = ret_sp;
                                fn_span = Some(ident.span);
                            }
                        }
                    }
                    coerce.coerce_forced_unit(
                        self,
                        &self.misc(sp),
                        &mut |err| {
                            if let Some(expected_ty) = expected.only_has_type(self) {
                                self.consider_hint_about_removing_semicolon(blk, expected_ty, err);
                            }
                            if let Some(fn_span) = fn_span {
                                err.span_label(
                                    fn_span,
                                    "implicitly returns `()` as its body has no tail or `return` \
                                 expression",
                                );
                            }
                        },
                        false,
                    );
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

    fn parent_item_span(&self, id: hir::HirId) -> Option<Span> {
        let node = self.tcx.hir().get(self.tcx.hir().get_parent_item(id));
        match node {
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn(_, _, body_id), .. })
            | Node::ImplItem(&hir::ImplItem {
                kind: hir::ImplItemKind::Method(_, body_id), ..
            }) => {
                let body = self.tcx.hir().body(body_id);
                if let ExprKind::Block(block, _) = &body.value.kind {
                    return Some(block.span);
                }
            }
            _ => {}
        }
        None
    }

    /// Given a function block's `HirId`, returns its `FnDecl` if it exists, or `None` otherwise.
    fn get_parent_fn_decl(
        &self,
        blk_id: hir::HirId,
    ) -> Option<(&'tcx hir::FnDecl<'tcx>, ast::Ident)> {
        let parent = self.tcx.hir().get(self.tcx.hir().get_parent_item(blk_id));
        self.get_node_fn_decl(parent).map(|(fn_decl, ident, _)| (fn_decl, ident))
    }

    /// Given a function `Node`, return its `FnDecl` if it exists, or `None` otherwise.
    fn get_node_fn_decl(
        &self,
        node: Node<'tcx>,
    ) -> Option<(&'tcx hir::FnDecl<'tcx>, ast::Ident, bool)> {
        match node {
            Node::Item(&hir::Item { ident, kind: hir::ItemKind::Fn(ref sig, ..), .. }) => {
                // This is less than ideal, it will not suggest a return type span on any
                // method called `main`, regardless of whether it is actually the entry point,
                // but it will still present it as the reason for the expected type.
                Some((&sig.decl, ident, ident.name != sym::main))
            }
            Node::TraitItem(&hir::TraitItem {
                ident,
                kind: hir::TraitItemKind::Fn(ref sig, ..),
                ..
            }) => Some((&sig.decl, ident, true)),
            Node::ImplItem(&hir::ImplItem {
                ident,
                kind: hir::ImplItemKind::Method(ref sig, ..),
                ..
            }) => Some((&sig.decl, ident, false)),
            _ => None,
        }
    }

    /// Given a `HirId`, return the `FnDecl` of the method it is enclosed by and whether a
    /// suggestion can be made, `None` otherwise.
    pub fn get_fn_decl(&self, blk_id: hir::HirId) -> Option<(&'tcx hir::FnDecl<'tcx>, bool)> {
        // Get enclosing Fn, if it is a function or a trait method, unless there's a `loop` or
        // `while` before reaching it, as block tail returns are not available in them.
        self.tcx.hir().get_return_block(blk_id).and_then(|blk_id| {
            let parent = self.tcx.hir().get(blk_id);
            self.get_node_fn_decl(parent).map(|(fn_decl, _, is_main)| (fn_decl, is_main))
        })
    }

    /// On implicit return expressions with mismatched types, provides the following suggestions:
    ///
    /// - Points out the method's return type as the reason for the expected type.
    /// - Possible missing semicolon.
    /// - Possible missing return type if the return type is the default, and not `fn main()`.
    pub fn suggest_mismatched_types_on_tail(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        cause_span: Span,
        blk_id: hir::HirId,
    ) -> bool {
        let expr = expr.peel_drop_temps();
        self.suggest_missing_semicolon(err, expr, expected, cause_span);
        let mut pointing_at_return_type = false;
        if let Some((fn_decl, can_suggest)) = self.get_fn_decl(blk_id) {
            pointing_at_return_type =
                self.suggest_missing_return_type(err, &fn_decl, expected, found, can_suggest);
        }
        pointing_at_return_type
    }

    /// When encountering an fn-like ctor that needs to unify with a value, check whether calling
    /// the ctor would successfully solve the type mismatch and if so, suggest it:
    /// ```
    /// fn foo(x: usize) -> usize { x }
    /// let x: usize = foo;  // suggest calling the `foo` function: `foo(42)`
    /// ```
    fn suggest_fn_call(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        let hir = self.tcx.hir();
        let (def_id, sig) = match found.kind {
            ty::FnDef(def_id, _) => (def_id, found.fn_sig(self.tcx)),
            ty::Closure(def_id, substs) => {
                // We don't use `closure_sig` to account for malformed closures like
                // `|_: [_; continue]| {}` and instead we don't suggest anything.
                let closure_sig_ty = substs.as_closure().sig_ty(def_id, self.tcx);
                (
                    def_id,
                    match closure_sig_ty.kind {
                        ty::FnPtr(sig) => sig,
                        _ => return false,
                    },
                )
            }
            _ => return false,
        };

        let sig = self.replace_bound_vars_with_fresh_vars(expr.span, infer::FnCall, &sig).0;
        let sig = self.normalize_associated_types_in(expr.span, &sig);
        if self.can_coerce(sig.output(), expected) {
            let (mut sugg_call, applicability) = if sig.inputs().is_empty() {
                (String::new(), Applicability::MachineApplicable)
            } else {
                ("...".to_string(), Applicability::HasPlaceholders)
            };
            let mut msg = "call this function";
            match hir.get_if_local(def_id) {
                Some(Node::Item(hir::Item { kind: ItemKind::Fn(.., body_id), .. }))
                | Some(Node::ImplItem(hir::ImplItem {
                    kind: hir::ImplItemKind::Method(_, body_id),
                    ..
                }))
                | Some(Node::TraitItem(hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(.., hir::TraitMethod::Provided(body_id)),
                    ..
                })) => {
                    let body = hir.body(*body_id);
                    sugg_call = body
                        .params
                        .iter()
                        .map(|param| match &param.pat.kind {
                            hir::PatKind::Binding(_, _, ident, None)
                                if ident.name != kw::SelfLower =>
                            {
                                ident.to_string()
                            }
                            _ => "_".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                Some(Node::Expr(hir::Expr {
                    kind: ExprKind::Closure(_, _, body_id, _, _),
                    span: full_closure_span,
                    ..
                })) => {
                    if *full_closure_span == expr.span {
                        return false;
                    }
                    msg = "call this closure";
                    let body = hir.body(*body_id);
                    sugg_call = body
                        .params
                        .iter()
                        .map(|param| match &param.pat.kind {
                            hir::PatKind::Binding(_, _, ident, None)
                                if ident.name != kw::SelfLower =>
                            {
                                ident.to_string()
                            }
                            _ => "_".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                Some(Node::Ctor(hir::VariantData::Tuple(fields, _))) => {
                    sugg_call = fields.iter().map(|_| "_").collect::<Vec<_>>().join(", ");
                    match hir.as_local_hir_id(def_id).and_then(|hir_id| hir.def_kind(hir_id)) {
                        Some(hir::def::DefKind::Ctor(hir::def::CtorOf::Variant, _)) => {
                            msg = "instantiate this tuple variant";
                        }
                        Some(hir::def::DefKind::Ctor(hir::def::CtorOf::Struct, _)) => {
                            msg = "instantiate this tuple struct";
                        }
                        _ => {}
                    }
                }
                Some(Node::ForeignItem(hir::ForeignItem {
                    kind: hir::ForeignItemKind::Fn(_, idents, _),
                    ..
                })) => {
                    sugg_call = idents
                        .iter()
                        .map(|ident| {
                            if ident.name != kw::SelfLower {
                                ident.to_string()
                            } else {
                                "_".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                Some(Node::TraitItem(hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(.., hir::TraitMethod::Required(idents)),
                    ..
                })) => {
                    sugg_call = idents
                        .iter()
                        .map(|ident| {
                            if ident.name != kw::SelfLower {
                                ident.to_string()
                            } else {
                                "_".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                _ => {}
            }
            if let Ok(code) = self.sess().source_map().span_to_snippet(expr.span) {
                err.span_suggestion(
                    expr.span,
                    &format!("use parentheses to {}", msg),
                    format!("{}({})", code, sugg_call),
                    applicability,
                );
                return true;
            }
        }
        false
    }

    pub fn suggest_ref_or_into(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if let Some((sp, msg, suggestion)) = self.check_ref(expr, found, expected) {
            err.span_suggestion(sp, msg, suggestion, Applicability::MachineApplicable);
        } else if let (ty::FnDef(def_id, ..), true) =
            (&found.kind, self.suggest_fn_call(err, expr, expected, found))
        {
            if let Some(sp) = self.tcx.hir().span_if_local(*def_id) {
                let sp = self.sess().source_map().def_span(sp);
                err.span_label(sp, &format!("{} defined here", found));
            }
        } else if !self.check_for_cast(err, expr, found, expected) {
            let is_struct_pat_shorthand_field =
                self.is_hir_id_from_struct_pattern_shorthand_field(expr.hir_id, expr.span);
            let methods = self.get_conversion_methods(expr.span, expected, found);
            if let Ok(expr_text) = self.sess().source_map().span_to_snippet(expr.span) {
                let mut suggestions = iter::repeat(&expr_text)
                    .zip(methods.iter())
                    .filter_map(|(receiver, method)| {
                        let method_call = format!(".{}()", method.ident);
                        if receiver.ends_with(&method_call) {
                            None // do not suggest code that is already there (#53348)
                        } else {
                            let method_call_list = [".to_vec()", ".to_string()"];
                            let sugg = if receiver.ends_with(".clone()")
                                && method_call_list.contains(&method_call.as_str())
                            {
                                let max_len = receiver.rfind('.').unwrap();
                                format!("{}{}", &receiver[..max_len], method_call)
                            } else {
                                if expr.precedence().order() < ExprPrecedence::MethodCall.order() {
                                    format!("({}){}", receiver, method_call)
                                } else {
                                    format!("{}{}", receiver, method_call)
                                }
                            };
                            Some(if is_struct_pat_shorthand_field {
                                format!("{}: {}", receiver, sugg)
                            } else {
                                sugg
                            })
                        }
                    })
                    .peekable();
                if suggestions.peek().is_some() {
                    err.span_suggestions(
                        expr.span,
                        "try using a conversion method",
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    /// When encountering the expected boxed value allocated in the stack, suggest allocating it
    /// in the heap by calling `Box::new()`.
    fn suggest_boxing_when_appropriate(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if self.tcx.hir().is_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return;
        }
        if !expected.is_box() || found.is_box() {
            return;
        }
        let boxed_found = self.tcx.mk_box(found);
        if let (true, Ok(snippet)) = (
            self.can_coerce(boxed_found, expected),
            self.sess().source_map().span_to_snippet(expr.span),
        ) {
            err.span_suggestion(
                expr.span,
                "store this in the heap by calling `Box::new`",
                format!("Box::new({})", snippet),
                Applicability::MachineApplicable,
            );
            err.note(
                "for more on the distinction between the stack and the heap, read \
                 https://doc.rust-lang.org/book/ch15-01-box.html, \
                 https://doc.rust-lang.org/rust-by-example/std/box.html, and \
                 https://doc.rust-lang.org/std/boxed/index.html",
            );
        }
    }

    /// When encountering an `impl Future` where `BoxFuture` is expected, suggest `Box::pin`.
    fn suggest_calling_boxed_future_when_appropriate(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        // Handle #68197.

        if self.tcx.hir().is_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return false;
        }
        let pin_did = self.tcx.lang_items().pin_type();
        match expected.kind {
            ty::Adt(def, _) if Some(def.did) != pin_did => return false,
            // This guards the `unwrap` and `mk_box` below.
            _ if pin_did.is_none() || self.tcx.lang_items().owned_box().is_none() => return false,
            _ => {}
        }
        let boxed_found = self.tcx.mk_box(found);
        let new_found = self.tcx.mk_lang_item(boxed_found, lang_items::PinTypeLangItem).unwrap();
        if let (true, Ok(snippet)) = (
            self.can_coerce(new_found, expected),
            self.sess().source_map().span_to_snippet(expr.span),
        ) {
            match found.kind {
                ty::Adt(def, _) if def.is_box() => {
                    err.help("use `Box::pin`");
                }
                _ => {
                    err.span_suggestion(
                        expr.span,
                        "you need to pin and box this expression",
                        format!("Box::pin({})", snippet),
                        Applicability::MachineApplicable,
                    );
                }
            }
            true
        } else {
            false
        }
    }

    /// A common error is to forget to add a semicolon at the end of a block, e.g.,
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
    fn suggest_missing_semicolon(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expression: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        cause_span: Span,
    ) {
        if expected.is_unit() {
            // `BlockTailExpression` only relevant if the tail expr would be
            // useful on its own.
            match expression.kind {
                ExprKind::Call(..)
                | ExprKind::MethodCall(..)
                | ExprKind::Loop(..)
                | ExprKind::Match(..)
                | ExprKind::Block(..) => {
                    err.span_suggestion(
                        cause_span.shrink_to_hi(),
                        "try adding a semicolon",
                        ";".to_string(),
                        Applicability::MachineApplicable,
                    );
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
    fn suggest_missing_return_type(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        fn_decl: &hir::FnDecl<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        can_suggest: bool,
    ) -> bool {
        // Only suggest changing the return type for methods that
        // haven't set a return type at all (and aren't `fn main()` or an impl).
        match (&fn_decl.output, found.is_suggestable(), can_suggest, expected.is_unit()) {
            (&hir::FnRetTy::DefaultReturn(span), true, true, true) => {
                err.span_suggestion(
                    span,
                    "try adding a return type",
                    format!("-> {} ", self.resolve_vars_with_obligations(found)),
                    Applicability::MachineApplicable,
                );
                true
            }
            (&hir::FnRetTy::DefaultReturn(span), false, true, true) => {
                err.span_label(span, "possibly return type missing here?");
                true
            }
            (&hir::FnRetTy::DefaultReturn(span), _, false, true) => {
                // `fn main()` must return `()`, do not suggest changing return type
                err.span_label(span, "expected `()` because of default return type");
                true
            }
            // expectation was caused by something else, not the default return
            (&hir::FnRetTy::DefaultReturn(_), _, _, false) => false,
            (&hir::FnRetTy::Return(ref ty), _, _, _) => {
                // Only point to return type if the expected type is the return type, as if they
                // are not, the expectation must have been caused by something else.
                debug!("suggest_missing_return_type: return type {:?} node {:?}", ty, ty.kind);
                let sp = ty.span;
                let ty = AstConv::ast_ty_to_ty(self, ty);
                debug!("suggest_missing_return_type: return type {:?}", ty);
                debug!("suggest_missing_return_type: expected type {:?}", ty);
                if ty.kind == expected.kind {
                    err.span_label(sp, format!("expected `{}` because of return type", expected));
                    return true;
                }
                false
            }
        }
    }

    /// A possible error is to forget to add `.await` when using futures:
    ///
    /// ```
    /// async fn make_u32() -> u32 {
    ///     22
    /// }
    ///
    /// fn take_u32(x: u32) {}
    ///
    /// async fn foo() {
    ///     let x = make_u32();
    ///     take_u32(x);
    /// }
    /// ```
    ///
    /// This routine checks if the found type `T` implements `Future<Output=U>` where `U` is the
    /// expected type. If this is the case, and we are inside of an async body, it suggests adding
    /// `.await` to the tail of the expression.
    fn suggest_missing_await(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        // `.await` is not permitted outside of `async` bodies, so don't bother to suggest if the
        // body isn't `async`.
        let item_id = self.tcx().hir().get_parent_node(self.body_id);
        if let Some(body_id) = self.tcx().hir().maybe_body_owned_by(item_id) {
            let body = self.tcx().hir().body(body_id);
            if let Some(hir::GeneratorKind::Async(_)) = body.generator_kind {
                let sp = expr.span;
                // Check for `Future` implementations by constructing a predicate to
                // prove: `<T as Future>::Output == U`
                let future_trait = self.tcx.lang_items().future_trait().unwrap();
                let item_def_id = self
                    .tcx
                    .associated_items(future_trait)
                    .in_definition_order()
                    .next()
                    .unwrap()
                    .def_id;
                let predicate =
                    ty::Predicate::Projection(ty::Binder::bind(ty::ProjectionPredicate {
                        // `<T as Future>::Output`
                        projection_ty: ty::ProjectionTy {
                            // `T`
                            substs: self.tcx.mk_substs_trait(
                                found,
                                self.fresh_substs_for_item(sp, item_def_id),
                            ),
                            // `Future::Output`
                            item_def_id,
                        },
                        ty: expected,
                    }));
                let obligation = traits::Obligation::new(self.misc(sp), self.param_env, predicate);
                debug!("suggest_missing_await: trying obligation {:?}", obligation);
                if self.infcx.predicate_may_hold(&obligation) {
                    debug!("suggest_missing_await: obligation held: {:?}", obligation);
                    if let Ok(code) = self.sess().source_map().span_to_snippet(sp) {
                        err.span_suggestion(
                            sp,
                            "consider using `.await` here",
                            format!("{}.await", code),
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        debug!("suggest_missing_await: no snippet for {:?}", sp);
                    }
                } else {
                    debug!("suggest_missing_await: obligation did not hold: {:?}", obligation)
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
    fn consider_hint_about_removing_semicolon(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        if let Some(span_semi) = self.could_remove_semicolon(blk, expected_ty) {
            err.span_suggestion(
                span_semi,
                "consider removing this semicolon",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }

    fn could_remove_semicolon(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> Option<Span> {
        // Be helpful when the user wrote `{... expr;}` and
        // taking the `;` off is enough to fix the error.
        let last_stmt = blk.stmts.last()?;
        let last_expr = match last_stmt.kind {
            hir::StmtKind::Semi(ref e) => e,
            _ => return None,
        };
        let last_expr_ty = self.node_ty(last_expr.hir_id);
        if self.can_sub(self.param_env, last_expr_ty, expected_ty).is_err() {
            return None;
        }
        let original_span = original_sp(last_stmt.span, blk.span);
        Some(original_span.with_lo(original_span.hi() - BytePos(1)))
    }

    // Instantiates the given path, which must refer to an item with the given
    // number of type parameters and type.
    pub fn instantiate_value_path(
        &self,
        segments: &[hir::PathSegment<'_>],
        self_ty: Option<Ty<'tcx>>,
        res: Res,
        span: Span,
        hir_id: hir::HirId,
    ) -> (Ty<'tcx>, Res) {
        debug!(
            "instantiate_value_path(segments={:?}, self_ty={:?}, res={:?}, hir_id={})",
            segments, self_ty, res, hir_id,
        );

        let tcx = self.tcx;

        let path_segs = match res {
            Res::Local(_) | Res::SelfCtor(_) => vec![],
            Res::Def(kind, def_id) => {
                AstConv::def_ids_for_value_path_segments(self, segments, self_ty, kind, def_id)
            }
            _ => bug!("instantiate_value_path on {:?}", res),
        };

        let mut user_self_ty = None;
        let mut is_alias_variant_ctor = false;
        match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, _), _) => {
                if let Some(self_ty) = self_ty {
                    let adt_def = self_ty.ty_adt_def().unwrap();
                    user_self_ty = Some(UserSelfTy { impl_def_id: adt_def.did, self_ty });
                    is_alias_variant_ctor = true;
                }
            }
            Res::Def(DefKind::AssocFn, def_id) | Res::Def(DefKind::AssocConst, def_id) => {
                let container = tcx.associated_item(def_id).container;
                debug!("instantiate_value_path: def_id={:?} container={:?}", def_id, container);
                match container {
                    ty::TraitContainer(trait_did) => {
                        callee::check_legal_trait_for_method_call(tcx, span, trait_did)
                    }
                    ty::ImplContainer(impl_def_id) => {
                        if segments.len() == 1 {
                            // `<T>::assoc` will end up here, and so
                            // can `T::assoc`. It this came from an
                            // inherent impl, we need to record the
                            // `T` for posterity (see `UserSelfTy` for
                            // details).
                            let self_ty = self_ty.expect("UFCS sugared assoc missing Self");
                            user_self_ty = Some(UserSelfTy { impl_def_id, self_ty });
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

        let generic_segs: FxHashSet<_> = path_segs.iter().map(|PathSeg(_, index)| index).collect();
        let generics_has_err = AstConv::prohibit_generics(
            self,
            segments.iter().enumerate().filter_map(|(index, seg)| {
                if !generic_segs.contains(&index) || is_alias_variant_ctor {
                    Some(seg)
                } else {
                    None
                }
            }),
        );

        if let Res::Local(hid) = res {
            let ty = self.local_ty(span, hid).decl_ty;
            let ty = self.normalize_associated_types_in(span, &ty);
            self.write_ty(hir_id, ty);
            return (ty, res);
        }

        if generics_has_err {
            // Don't try to infer type parameters when prohibited generic arguments were given.
            user_self_ty = None;
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
            let generics = tcx.generics_of(def_id);
            // Argument-position `impl Trait` is treated as a normal generic
            // parameter internally, but we don't allow users to specify the
            // parameter's value explicitly, so we have to do some error-
            // checking here.
            if let Err(GenericArgCountMismatch { reported: Some(ErrorReported), .. }) =
                AstConv::check_generic_arg_count_for_call(
                    tcx, span, &generics, &seg, false, // `is_method_call`
                )
            {
                infer_args_for_err.insert(index);
                self.set_tainted_by_errors(); // See issue #53251.
            }
        }

        let has_self = path_segs
            .last()
            .map(|PathSeg(def_id, _)| tcx.generics_of(*def_id).has_self)
            .unwrap_or(false);

        let (res, self_ctor_substs) = if let Res::SelfCtor(impl_def_id) = res {
            let ty = self.normalize_ty(span, tcx.at(span).type_of(impl_def_id));
            match ty.kind {
                ty::Adt(adt_def, substs) if adt_def.has_ctor() => {
                    let variant = adt_def.non_enum_variant();
                    let ctor_def_id = variant.ctor_def_id.unwrap();
                    (
                        Res::Def(DefKind::Ctor(CtorOf::Struct, variant.ctor_kind), ctor_def_id),
                        Some(substs),
                    )
                }
                _ => {
                    let mut err = tcx.sess.struct_span_err(
                        span,
                        "the `Self` constructor can only be used with tuple or unit structs",
                    );
                    if let Some(adt_def) = ty.ty_adt_def() {
                        match adt_def.adt_kind() {
                            AdtKind::Enum => {
                                err.help("did you mean to use one of the enum's variants?");
                            }
                            AdtKind::Struct | AdtKind::Union => {
                                err.span_suggestion(
                                    span,
                                    "use curly brackets",
                                    String::from("Self { /* fields */ }"),
                                    Applicability::HasPlaceholders,
                                );
                            }
                        }
                    }
                    err.emit();

                    return (tcx.types.err, res);
                }
            }
        } else {
            (res, None)
        };
        let def_id = res.def_id();

        // The things we are substituting into the type should not contain
        // escaping late-bound regions, and nor should the base type scheme.
        let ty = tcx.type_of(def_id);

        let substs = self_ctor_substs.unwrap_or_else(|| {
            AstConv::create_substs_for_generic_args(
                tcx,
                def_id,
                &[][..],
                has_self,
                self_ty,
                infer_args_for_err.is_empty(),
                // Provide the generic args, and whether types should be inferred.
                |def_id| {
                    if let Some(&PathSeg(_, index)) =
                        path_segs.iter().find(|&PathSeg(did, _)| *did == def_id)
                    {
                        // If we've encountered an `impl Trait`-related error, we're just
                        // going to infer the arguments for better error messages.
                        if !infer_args_for_err.contains(&index) {
                            // Check whether the user has provided generic arguments.
                            if let Some(ref data) = segments[index].args {
                                return (Some(data), segments[index].infer_args);
                            }
                        }
                        return (None, segments[index].infer_args);
                    }

                    (None, true)
                },
                // Provide substitutions for parameters for which (valid) arguments have been provided.
                |param, arg| match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        AstConv::ast_region_to_region(self, lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.to_ty(ty).into()
                    }
                    (GenericParamDefKind::Const, GenericArg::Const(ct)) => {
                        self.to_const(&ct.value, self.tcx.type_of(param.def_id)).into()
                    }
                    _ => unreachable!(),
                },
                // Provide substitutions for parameters for which arguments are inferred.
                |substs, param, infer_args| {
                    match param.kind {
                        GenericParamDefKind::Lifetime => {
                            self.re_infer(Some(param), span).unwrap().into()
                        }
                        GenericParamDefKind::Type { has_default, .. } => {
                            if !infer_args && has_default {
                                // If we have a default, then we it doesn't matter that we're not
                                // inferring the type arguments: we provide the default where any
                                // is missing.
                                let default = tcx.type_of(param.def_id);
                                self.normalize_ty(
                                    span,
                                    default.subst_spanned(tcx, substs.unwrap(), Some(span)),
                                )
                                .into()
                            } else {
                                // If no type arguments were provided, we have to infer them.
                                // This case also occurs as a result of some malformed input, e.g.
                                // a lifetime argument being given instead of a type parameter.
                                // Using inference instead of `Error` gives better error messages.
                                self.var_for_def(span, param)
                            }
                        }
                        GenericParamDefKind::Const => {
                            // FIXME(const_generics:defaults)
                            // No const parameters were provided, we have to infer them.
                            self.var_for_def(span, param)
                        }
                    }
                },
            )
        });
        assert!(!substs.has_escaping_bound_vars());
        assert!(!ty.has_escaping_bound_vars());

        // First, store the "user substs" for later.
        self.write_user_type_annotation_from_substs(hir_id, def_id, substs, user_self_ty);

        self.add_required_obligations(span, def_id, &substs);

        // Substitute the values for the type parameters into the type of
        // the referenced item.
        let ty_substituted = self.instantiate_type_scheme(span, &substs, &ty);

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            // In the case of `Foo<T>::method` and `<Foo<T>>::method`, if `method`
            // is inherent, there is no `Self` parameter; instead, the impl needs
            // type parameters, which we can infer by unifying the provided `Self`
            // with the substituted impl type.
            // This also occurs for an enum variant on a type alias.
            let ty = tcx.type_of(impl_def_id);

            let impl_ty = self.instantiate_type_scheme(span, &substs, &ty);
            match self.at(&self.misc(span), self.param_env).sup(impl_ty, self_ty) {
                Ok(ok) => self.register_infer_ok_obligations(ok),
                Err(_) => {
                    self.tcx.sess.delay_span_bug(span, &format!(
                        "instantiate_value_path: (UFCS) {:?} was a subtype of {:?} but now is not?",
                        self_ty,
                        impl_ty,
                    ));
                }
            }
        }

        self.check_rustc_args_require_const(def_id, hir_id, span);

        debug!("instantiate_value_path: type of {:?} is {:?}", hir_id, ty_substituted);
        self.write_substs(hir_id, substs);

        (ty_substituted, res)
    }

    /// Add all the obligations that are required, substituting and normalized appropriately.
    fn add_required_obligations(&self, span: Span, def_id: DefId, substs: &SubstsRef<'tcx>) {
        let (bounds, spans) = self.instantiate_bounds(span, def_id, &substs);

        for (i, mut obligation) in traits::predicates_for_generics(
            traits::ObligationCause::new(span, self.body_id, traits::ItemObligation(def_id)),
            self.param_env,
            &bounds,
        )
        .into_iter()
        .enumerate()
        {
            // This makes the error point at the bound, but we want to point at the argument
            if let Some(span) = spans.get(i) {
                obligation.cause.code = traits::BindingObligation(def_id, *span);
            }
            self.register_predicate(obligation);
        }
    }

    fn check_rustc_args_require_const(&self, def_id: DefId, hir_id: hir::HirId, span: Span) {
        // We're only interested in functions tagged with
        // #[rustc_args_required_const], so ignore anything that's not.
        if !self.tcx.has_attr(def_id, sym::rustc_args_required_const) {
            return;
        }

        // If our calling expression is indeed the function itself, we're good!
        // If not, generate an error that this can only be called directly.
        if let Node::Expr(expr) = self.tcx.hir().get(self.tcx.hir().get_parent_node(hir_id)) {
            if let ExprKind::Call(ref callee, ..) = expr.kind {
                if callee.hir_id == hir_id {
                    return;
                }
            }
        }

        self.tcx.sess.span_err(
            span,
            "this function can only be invoked directly, not through a function pointer",
        );
    }

    /// Resolves `typ` by a single level if `typ` is a type variable.
    /// If no resolution is possible, then an error is reported.
    /// Numeric inference variables may be left unresolved.
    pub fn structurally_resolved_type(&self, sp: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.resolve_vars_with_obligations(ty);
        if !ty.is_ty_var() {
            ty
        } else {
            if !self.is_tainted_by_errors() {
                self.need_type_info_err((**self).body_id, sp, ty, E0282)
                    .note("type must be known at this point")
                    .emit();
            }
            self.demand_suptype(sp, self.tcx.types.err, ty);
            self.tcx.types.err
        }
    }

    fn with_breakable_ctxt<F: FnOnce() -> R, R>(
        &self,
        id: hir::HirId,
        ctxt: BreakableCtxt<'tcx>,
        f: F,
    ) -> (BreakableCtxt<'tcx>, R) {
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

    /// Instantiate a QueryResponse in a probe context, without a
    /// good ObligationCause.
    fn probe_instantiate_query_response(
        &self,
        span: Span,
        original_values: &OriginalQueryValues<'tcx>,
        query_result: &Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
    ) -> InferResult<'tcx, Ty<'tcx>> {
        self.instantiate_query_response_and_region_obligations(
            &traits::ObligationCause::misc(span, self.body_id),
            self.param_env,
            original_values,
            query_result,
        )
    }

    /// Returns `true` if an expression is contained inside the LHS of an assignment expression.
    fn expr_in_place(&self, mut expr_id: hir::HirId) -> bool {
        let mut contained_in_place = false;

        while let hir::Node::Expr(parent_expr) =
            self.tcx.hir().get(self.tcx.hir().get_parent_node(expr_id))
        {
            match &parent_expr.kind {
                hir::ExprKind::Assign(lhs, ..) | hir::ExprKind::AssignOp(_, lhs, ..) => {
                    if lhs.hir_id == expr_id {
                        contained_in_place = true;
                        break;
                    }
                }
                _ => (),
            }
            expr_id = parent_expr.hir_id;
        }

        contained_in_place
    }
}

pub fn check_bounds_are_used<'tcx>(tcx: TyCtxt<'tcx>, generics: &ty::Generics, ty: Ty<'tcx>) {
    let own_counts = generics.own_counts();
    debug!(
        "check_bounds_are_used(n_tys={}, n_cts={}, ty={:?})",
        own_counts.types, own_counts.consts, ty
    );

    if own_counts.types == 0 {
        return;
    }

    // Make a vector of booleans initially `false`; set to `true` when used.
    let mut types_used = vec![false; own_counts.types];

    for leaf_ty in ty.walk() {
        if let ty::Param(ty::ParamTy { index, .. }) = leaf_ty.kind {
            debug!("found use of ty param num {}", index);
            types_used[index as usize - own_counts.lifetimes] = true;
        } else if let ty::Error = leaf_ty.kind {
            // If there is already another error, do not emit
            // an error for not using a type parameter.
            assert!(tcx.sess.has_errors());
            return;
        }
    }

    let types = generics.params.iter().filter(|param| match param.kind {
        ty::GenericParamDefKind::Type { .. } => true,
        _ => false,
    });
    for (&used, param) in types_used.iter().zip(types) {
        if !used {
            let id = tcx.hir().as_local_hir_id(param.def_id).unwrap();
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
        https://github.com/rust-lang/rust/issues/43162#issuecomment-320764675",
    );
    handler.note_without_error(&format!(
        "rustc {} running on {}",
        option_env!("CFG_VERSION").unwrap_or("unknown_version"),
        crate::session::config::host_triple(),
    ));
}

fn potentially_plural_count(count: usize, word: &str) -> String {
    format!("{} {}{}", count, word, pluralize!(count))
}
