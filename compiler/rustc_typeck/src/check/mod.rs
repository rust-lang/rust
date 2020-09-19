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
variables, are stored directly into the `tcx` typeck_results.

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
pub mod cast;
mod closure;
pub mod coercion;
mod compare_method;
pub mod demand;
pub mod dropck;
mod expr;
mod fn_ctxt;
mod gather_locals;
mod generator_interior;
pub mod intrinsic;
pub mod method;
mod op;
mod pat;
mod place_op;
mod regionck;
mod upvar;
mod wfcheck;
pub mod writeback;

pub use fn_ctxt::FnCtxt;

use crate::astconv::AstConv;
use crate::check::gather_locals::GatherLocalsVisitor;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, LOCAL_CRATE};
use rustc_hir::intravisit::Visitor;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{HirIdMap, ItemKind, Node};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use rustc_infer::infer;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{InferCtxt, InferOk, RegionVariableOrigin, TyCtxtInferExt};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::util::{Discr, IntTypeExt, Representability};
use rustc_middle::ty::WithConstness;
use rustc_middle::ty::{self, RegionKind, ToPredicate, Ty, TyCtxt, UserType};
use rustc_session::config::{self, EntryFnType};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{self, BytePos, MultiSpan, Span};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::opaque_types::OpaqueTypeDecl;
use rustc_trait_selection::traits::error_reporting::recursive_type_with_infinite_size_error;
use rustc_trait_selection::traits::error_reporting::suggestions::ReturnsVisitor;
use rustc_trait_selection::traits::{self, ObligationCauseCode, TraitEngine, TraitEngineExt};

use std::cell::{Ref, RefCell, RefMut};
use std::cmp;
use std::ops::{self, Deref};

use crate::require_c_abi_if_c_variadic;
use crate::util::common::indenter;

use self::callee::DeferredCallResolution;
use self::coercion::{CoerceMany, DynamicCoerceMany};
use self::compare_method::{compare_const_impl, compare_impl_method, compare_ty_impl};
pub use self::Expectation::*;

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

    typeck_results: MaybeInProgressTables<'a, 'tcx>,

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
        match fcx.tcx.struct_tail_without_normalization(ty).kind() {
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

/// Helper type of a temporary returned by `Inherited::build(...)`.
/// Necessary because we can't write the following bound:
/// `F: for<'b, 'tcx> where 'tcx FnOnce(Inherited<'b, 'tcx>)`.
pub struct InheritedBuilder<'tcx> {
    infcx: infer::InferCtxtBuilder<'tcx>,
    def_id: LocalDefId,
}

impl Inherited<'_, 'tcx> {
    pub fn build(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> InheritedBuilder<'tcx> {
        let hir_owner = tcx.hir().local_def_id_to_hir_id(def_id).owner;

        InheritedBuilder {
            infcx: tcx.infer_ctxt().with_fresh_in_progress_typeck_results(hir_owner),
            def_id,
        }
    }
}

impl<'tcx> InheritedBuilder<'tcx> {
    pub fn enter<F, R>(&mut self, f: F) -> R
    where
        F: for<'a> FnOnce(Inherited<'a, 'tcx>) -> R,
    {
        let def_id = self.def_id;
        self.infcx.enter(|infcx| f(Inherited::new(infcx, def_id)))
    }
}

impl Inherited<'a, 'tcx> {
    fn new(infcx: InferCtxt<'a, 'tcx>, def_id: LocalDefId) -> Self {
        let tcx = infcx.tcx;
        let item_id = tcx.hir().local_def_id_to_hir_id(def_id);
        let body_id = tcx.hir().maybe_body_owned_by(item_id);

        Inherited {
            typeck_results: MaybeInProgressTables {
                maybe_typeck_results: infcx.in_progress_typeck_results,
            },
            infcx,
            fulfillment_cx: RefCell::new(TraitEngine::new(tcx)),
            locals: RefCell::new(Default::default()),
            deferred_sized_obligations: RefCell::new(Vec::new()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_generator_interiors: RefCell::new(Vec::new()),
            opaque_types: RefCell::new(Default::default()),
            opaque_types_vars: RefCell::new(Default::default()),
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

pub fn check_wf_new(tcx: TyCtxt<'_>) {
    let visit = wfcheck::CheckTypeWellFormedVisitor::new(tcx);
    tcx.hir().krate().par_visit_all_item_likes(&visit);
}

pub fn provide(providers: &mut Providers) {
    method::provide(providers);
    *providers = Providers {
        typeck_item_bodies,
        typeck_const_arg,
        typeck,
        diagnostic_only_typeck,
        has_typeck_results,
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
/// If this function returns `Some`, then `typeck_results(def_id)` will
/// succeed; if it returns `None`, then `typeck_results(def_id)` may or
/// may not succeed. In some cases where this function returns `None`
/// (notably closures), `typeck_results(def_id)` would wind up
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
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                Some((body, None, Some(&sig.header), Some(&sig.decl)))
            }
            _ => None,
        },
        Node::ImplItem(item) => match item.kind {
            hir::ImplItemKind::Const(ref ty, body) => Some((body, Some(ty), None, None)),
            hir::ImplItemKind::Fn(ref sig, body) => {
                Some((body, None, Some(&sig.header), Some(&sig.decl)))
            }
            _ => None,
        },
        Node::AnonConst(constant) => Some((constant.body, None, None, None)),
        _ => None,
    }
}

fn has_typeck_results(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // Closures' typeck results come from their outermost function,
    // as they are part of the same "inference environment".
    let outer_def_id = tcx.closure_base_def_id(def_id);
    if outer_def_id != def_id {
        return tcx.has_typeck_results(outer_def_id);
    }

    if let Some(def_id) = def_id.as_local() {
        let id = tcx.hir().local_def_id_to_hir_id(def_id);
        primary_body_of(tcx, id).is_some()
    } else {
        false
    }
}

fn used_trait_imports(tcx: TyCtxt<'_>, def_id: LocalDefId) -> &FxHashSet<LocalDefId> {
    &*tcx.typeck(def_id).used_trait_imports
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
            match *ty.kind() {
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
                                    if let ty::Infer(_) = old_ty.kind() {
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

fn typeck_const_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    (did, param_did): (LocalDefId, DefId),
) -> &ty::TypeckResults<'tcx> {
    let fallback = move || tcx.type_of(param_did);
    typeck_with_fallback(tcx, did, fallback)
}

fn typeck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &ty::TypeckResults<'tcx> {
    if let Some(param_did) = tcx.opt_const_param_of(def_id) {
        tcx.typeck_const_arg((def_id, param_did))
    } else {
        let fallback = move || tcx.type_of(def_id.to_def_id());
        typeck_with_fallback(tcx, def_id, fallback)
    }
}

/// Used only to get `TypeckResults` for type inference during error recovery.
/// Currently only used for type inference of `static`s and `const`s to avoid type cycle errors.
fn diagnostic_only_typeck<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &ty::TypeckResults<'tcx> {
    let fallback = move || {
        let span = tcx.hir().span(tcx.hir().local_def_id_to_hir_id(def_id));
        tcx.ty_error_with_message(span, "diagnostic only typeck table used")
    };
    typeck_with_fallback(tcx, def_id, fallback)
}

fn typeck_with_fallback<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    fallback: impl Fn() -> Ty<'tcx> + 'tcx,
) -> &'tcx ty::TypeckResults<'tcx> {
    // Closures' typeck results come from their outermost function,
    // as they are part of the same "inference environment".
    let outer_def_id = tcx.closure_base_def_id(def_id.to_def_id()).expect_local();
    if outer_def_id != def_id {
        return tcx.typeck(outer_def_id);
    }

    let id = tcx.hir().local_def_id_to_hir_id(def_id);
    let span = tcx.hir().span(id);

    // Figure out what primary body this item has.
    let (body_id, body_ty, fn_header, fn_decl) = primary_body_of(tcx, id).unwrap_or_else(|| {
        span_bug!(span, "can't type-check body of {:?}", def_id);
    });
    let body = tcx.hir().body(body_id);

    let typeck_results = Inherited::build(tcx, def_id).enter(|inh| {
        let param_env = tcx.param_env(def_id);
        let fcx = if let (Some(header), Some(decl)) = (fn_header, fn_decl) {
            let fn_sig = if crate::collect::get_infer_ret_ty(&decl.output).is_some() {
                let fcx = FnCtxt::new(&inh, param_env, body.value.hir_id);
                AstConv::ty_of_fn(
                    &fcx,
                    header.unsafety,
                    header.abi,
                    decl,
                    &hir::Generics::empty(),
                    None,
                )
            } else {
                tcx.fn_sig(def_id)
            };

            check_abi(tcx, span, fn_sig.abi());

            // Compute the fty from point of view of inside the fn.
            let fn_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), &fn_sig);
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
            GatherLocalsVisitor::new(&fcx, id).visit_body(body);

            fcx.check_expr_coercable_to_type(&body.value, revealed_ty, None);

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
        fcx.resolve_generator_interiors(def_id.to_def_id());

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

    // Consistency check our TypeckResults instance can hold all ItemLocalIds
    // it will need to hold.
    assert_eq!(typeck_results.hir_owner, id.owner);

    typeck_results
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

    // Create the function context. This is either derived from scratch or,
    // in the case of closures, based on the outer context.
    let mut fcx = FnCtxt::new(inherited, param_env, body.value.hir_id);
    *fcx.ps.borrow_mut() = UnsafetyState::function(fn_sig.unsafety, fn_id);

    let tcx = fcx.tcx;
    let sess = tcx.sess;
    let hir = tcx.hir();

    let declared_ret_ty = fn_sig.output();

    let revealed_ret_ty =
        fcx.instantiate_opaque_types_from_value(fn_id, &declared_ret_ty, decl.output.span());
    debug!("check_fn: declared_ret_ty: {}, revealed_ret_ty: {}", declared_ret_ty, revealed_ret_ty);
    fcx.ret_coercion = Some(RefCell::new(CoerceMany::new(revealed_ret_ty)));
    fcx.ret_type_span = Some(decl.output.span());
    if let ty::Opaque(..) = declared_ret_ty.kind() {
        fcx.ret_coercion_impl_trait = Some(declared_ret_ty);
    }
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

    let outer_def_id = tcx.closure_base_def_id(hir.local_def_id(fn_id).to_def_id()).expect_local();
    let outer_hir_id = hir.local_def_id_to_hir_id(outer_def_id);
    GatherLocalsVisitor::new(&fcx, outer_hir_id).visit_body(body);

    // C-variadic fns also have a `VaList` input that's not listed in `fn_sig`
    // (as it's created inside the body itself, not passed in from outside).
    let maybe_va_list = if fn_sig.c_variadic {
        let span = body.params.last().unwrap().span;
        let va_list_did = tcx.require_lang_item(LangItem::VaList, Some(span));
        let region = fcx.next_region_var(RegionVariableOrigin::MiscVariable(span));

        Some(tcx.type_of(va_list_did).subst(tcx, &[region.into()]))
    } else {
        None
    };

    // Add formal parameters.
    let inputs_hir = hir.fn_decl_by_hir_id(fn_id).map(|decl| &decl.inputs);
    let inputs_fn = fn_sig.inputs().iter().copied();
    for (idx, (param_ty, param)) in inputs_fn.chain(maybe_va_list).zip(body.params).enumerate() {
        // Check the pattern.
        let ty_span = try { inputs_hir?.get(idx)?.span };
        fcx.check_pat_top(&param.pat, param_ty, ty_span, false);

        // Check that argument is Sized.
        // The check for a non-trivial pattern is a hack to avoid duplicate warnings
        // for simple cases like `fn foo(x: Trait)`,
        // where we would error once on the parameter as a whole, and once on the binding `x`.
        if param.pat.simple_ident().is_none() && !tcx.features().unsized_locals {
            fcx.require_type_is_sized(param_ty, param.pat.span, traits::SizedArgumentType(ty_span));
        }

        fcx.write_ty(param.hir_id, param_ty);
    }

    inherited.typeck_results.borrow_mut().liberated_fn_sigs_mut().insert(fn_id, fn_sig);

    fcx.in_tail_expr = true;
    if let ty::Dynamic(..) = declared_ret_ty.kind() {
        // FIXME: We need to verify that the return type is `Sized` after the return expression has
        // been evaluated so that we have types available for all the nodes being returned, but that
        // requires the coerced evaluated type to be stored. Moving `check_return_expr` before this
        // causes unsized errors caused by the `declared_ret_ty` to point at the return expression,
        // while keeping the current ordering we will ignore the tail expression's type because we
        // don't know it yet. We can't do `check_expr_kind` while keeping `check_return_expr`
        // because we will trigger "unreachable expression" lints unconditionally.
        // Because of all of this, we perform a crude check to know whether the simplest `!Sized`
        // case that a newcomer might make, returning a bare trait, and in that case we populate
        // the tail expression's type so that the suggestion will be correct, but ignore all other
        // possible cases.
        fcx.check_expr(&body.value);
        fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
        tcx.sess.delay_span_bug(decl.output.span(), "`!Sized` return type");
    } else {
        fcx.require_type_is_sized(declared_ret_ty, decl.output.span(), traits::SizedReturnType);
        fcx.check_return_expr(&body.value);
    }
    fcx.in_tail_expr = false;

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
            let main_id = hir.local_def_id_to_hir_id(def_id);
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
                    trait_ref.without_const().to_predicate(tcx),
                ));
            }
        }
    }

    // Check that a function marked as `#[panic_handler]` has signature `fn(&PanicInfo) -> !`
    if let Some(panic_impl_did) = tcx.lang_items().panic_impl() {
        if panic_impl_did == hir.local_def_id(fn_id).to_def_id() {
            if let Some(panic_info_did) = tcx.lang_items().panic_info() {
                if *declared_ret_ty.kind() != ty::Never {
                    sess.span_err(decl.output.span(), "return type should be `!`");
                }

                let inputs = fn_sig.inputs();
                let span = hir.span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_panic_info = match *inputs[0].kind() {
                        ty::Ref(region, ty, mutbl) => match *ty.kind() {
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
                    let span = sess.source_map().guess_head_span(span);
                    sess.span_err(span, "function should have one argument");
                }
            } else {
                sess.err("language item required, but not found: `panic_info`");
            }
        }
    }

    // Check that a function marked as `#[alloc_error_handler]` has signature `fn(Layout) -> !`
    if let Some(alloc_error_handler_did) = tcx.lang_items().oom() {
        if alloc_error_handler_did == hir.local_def_id(fn_id).to_def_id() {
            if let Some(alloc_layout_did) = tcx.lang_items().alloc_layout() {
                if *declared_ret_ty.kind() != ty::Never {
                    sess.span_err(decl.output.span(), "return type should be `!`");
                }

                let inputs = fn_sig.inputs();
                let span = hir.span(fn_id);
                if inputs.len() == 1 {
                    let arg_is_alloc_layout = match inputs[0].kind() {
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
                    let span = sess.source_map().guess_head_span(span);
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

    check_transparent(tcx, span, def);
    check_packed(tcx, span, def);
}

fn check_union(tcx: TyCtxt<'_>, id: hir::HirId, span: Span) {
    let def_id = tcx.hir().local_def_id(id);
    let def = tcx.adt_def(def_id);
    def.destructor(tcx); // force the destructor to be evaluated
    check_representable(tcx, span, def_id);
    check_transparent(tcx, span, def);
    check_union_fields(tcx, span, def_id);
    check_packed(tcx, span, def);
}

/// When the `#![feature(untagged_unions)]` gate is active,
/// check that the fields of the `union` does not contain fields that need dropping.
fn check_union_fields(tcx: TyCtxt<'_>, span: Span, item_def_id: LocalDefId) -> bool {
    let item_type = tcx.type_of(item_def_id);
    if let ty::Adt(def, substs) = item_type.kind() {
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
        span_bug!(span, "unions must be ty::Adt, but got {:?}", item_type.kind());
    }
    true
}

/// Checks that an opaque type does not contain cycles and does not use `Self` or `T::Foo`
/// projections that would result in "inheriting lifetimes".
fn check_opaque<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
    check_opaque_for_inheriting_lifetimes(tcx, def_id, span);
    check_opaque_for_cycles(tcx, def_id, substs, span, origin);
}

/// Checks that an opaque type does not use `Self` or `T::Foo` projections that would result
/// in "inheriting lifetimes".
fn check_opaque_for_inheriting_lifetimes(tcx: TyCtxt<'tcx>, def_id: LocalDefId, span: Span) {
    let item = tcx.hir().expect_item(tcx.hir().local_def_id_to_hir_id(def_id));
    debug!(
        "check_opaque_for_inheriting_lifetimes: def_id={:?} span={:?} item={:?}",
        def_id, span, item
    );

    #[derive(Debug)]
    struct ProhibitOpaqueVisitor<'tcx> {
        opaque_identity_ty: Ty<'tcx>,
        generics: &'tcx ty::Generics,
        ty: Option<Ty<'tcx>>,
    };

    impl<'tcx> ty::fold::TypeVisitor<'tcx> for ProhibitOpaqueVisitor<'tcx> {
        fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
            debug!("check_opaque_for_inheriting_lifetimes: (visit_ty) t={:?}", t);
            if t != self.opaque_identity_ty && t.super_visit_with(self) {
                self.ty = Some(t);
                return true;
            }
            false
        }

        fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
            debug!("check_opaque_for_inheriting_lifetimes: (visit_region) r={:?}", r);
            if let RegionKind::ReEarlyBound(ty::EarlyBoundRegion { index, .. }) = r {
                return *index < self.generics.parent_count as u32;
            }

            r.super_visit_with(self)
        }

        fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> bool {
            if let ty::ConstKind::Unevaluated(..) = c.val {
                // FIXME(#72219) We currenctly don't detect lifetimes within substs
                // which would violate this check. Even though the particular substitution is not used
                // within the const, this should still be fixed.
                return false;
            }
            c.super_visit_with(self)
        }
    }

    if let ItemKind::OpaqueTy(hir::OpaqueTy {
        origin: hir::OpaqueTyOrigin::AsyncFn | hir::OpaqueTyOrigin::FnReturn,
        ..
    }) = item.kind
    {
        let mut visitor = ProhibitOpaqueVisitor {
            opaque_identity_ty: tcx.mk_opaque(
                def_id.to_def_id(),
                InternalSubsts::identity_for_item(tcx, def_id.to_def_id()),
            ),
            generics: tcx.generics_of(def_id),
            ty: None,
        };
        let prohibit_opaque = tcx
            .predicates_of(def_id)
            .predicates
            .iter()
            .any(|(predicate, _)| predicate.visit_with(&mut visitor));
        debug!(
            "check_opaque_for_inheriting_lifetimes: prohibit_opaque={:?}, visitor={:?}",
            prohibit_opaque, visitor
        );

        if prohibit_opaque {
            let is_async = match item.kind {
                ItemKind::OpaqueTy(hir::OpaqueTy { origin, .. }) => match origin {
                    hir::OpaqueTyOrigin::AsyncFn => true,
                    _ => false,
                },
                _ => unreachable!(),
            };

            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0760,
                "`{}` return type cannot contain a projection or `Self` that references lifetimes from \
             a parent scope",
                if is_async { "async fn" } else { "impl Trait" },
            );

            if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(span) {
                if snippet == "Self" {
                    if let Some(ty) = visitor.ty {
                        err.span_suggestion(
                            span,
                            "consider spelling out the type instead",
                            format!("{:?}", ty),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
            err.emit();
        }
    }
}

/// Given a `DefId` for an opaque type in return position, find its parent item's return
/// expressions.
fn get_owner_return_paths(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Option<(hir::HirId, ReturnsVisitor<'tcx>)> {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let id = tcx.hir().get_parent_item(hir_id);
    tcx.hir()
        .find(id)
        .map(|n| (id, n))
        .and_then(|(hir_id, node)| node.body_id().map(|b| (hir_id, b)))
        .map(|(hir_id, body_id)| {
            let body = tcx.hir().body(body_id);
            let mut visitor = ReturnsVisitor::default();
            visitor.visit_body(body);
            (hir_id, visitor)
        })
}

/// Emit an error for recursive opaque types.
///
/// If this is a return `impl Trait`, find the item's return expressions and point at them. For
/// direct recursion this is enough, but for indirect recursion also point at the last intermediary
/// `impl Trait`.
///
/// If all the return expressions evaluate to `!`, then we explain that the error will go away
/// after changing it. This can happen when a user uses `panic!()` or similar as a placeholder.
fn opaque_type_cycle_error(tcx: TyCtxt<'tcx>, def_id: LocalDefId, span: Span) {
    let mut err = struct_span_err!(tcx.sess, span, E0720, "cannot resolve opaque type");

    let mut label = false;
    if let Some((hir_id, visitor)) = get_owner_return_paths(tcx, def_id) {
        let typeck_results = tcx.typeck(tcx.hir().local_def_id(hir_id));
        if visitor
            .returns
            .iter()
            .filter_map(|expr| typeck_results.node_type_opt(expr.hir_id))
            .all(|ty| matches!(ty.kind(), ty::Never))
        {
            let spans = visitor
                .returns
                .iter()
                .filter(|expr| typeck_results.node_type_opt(expr.hir_id).is_some())
                .map(|expr| expr.span)
                .collect::<Vec<Span>>();
            let span_len = spans.len();
            if span_len == 1 {
                err.span_label(spans[0], "this returned value is of `!` type");
            } else {
                let mut multispan: MultiSpan = spans.clone().into();
                for span in spans {
                    multispan
                        .push_span_label(span, "this returned value is of `!` type".to_string());
                }
                err.span_note(multispan, "these returned values have a concrete \"never\" type");
            }
            err.help("this error will resolve once the item's body returns a concrete type");
        } else {
            let mut seen = FxHashSet::default();
            seen.insert(span);
            err.span_label(span, "recursive opaque type");
            label = true;
            for (sp, ty) in visitor
                .returns
                .iter()
                .filter_map(|e| typeck_results.node_type_opt(e.hir_id).map(|t| (e.span, t)))
                .filter(|(_, ty)| !matches!(ty.kind(), ty::Never))
            {
                struct VisitTypes(Vec<DefId>);
                impl<'tcx> ty::fold::TypeVisitor<'tcx> for VisitTypes {
                    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
                        match *t.kind() {
                            ty::Opaque(def, _) => {
                                self.0.push(def);
                                false
                            }
                            _ => t.super_visit_with(self),
                        }
                    }
                }
                let mut visitor = VisitTypes(vec![]);
                ty.visit_with(&mut visitor);
                for def_id in visitor.0 {
                    let ty_span = tcx.def_span(def_id);
                    if !seen.contains(&ty_span) {
                        err.span_label(ty_span, &format!("returning this opaque type `{}`", ty));
                        seen.insert(ty_span);
                    }
                    err.span_label(sp, &format!("returning here with type `{}`", ty));
                }
            }
        }
    }
    if !label {
        err.span_label(span, "cannot resolve opaque type");
    }
    err.emit();
}

/// Emit an error for recursive opaque types in a `let` binding.
fn binding_opaque_type_cycle_error(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    span: Span,
    partially_expanded_type: Ty<'tcx>,
) {
    let mut err = struct_span_err!(tcx.sess, span, E0720, "cannot resolve opaque type");
    err.span_label(span, "cannot resolve opaque type");
    // Find the the owner that declared this `impl Trait` type.
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let mut prev_hir_id = hir_id;
    let mut hir_id = tcx.hir().get_parent_node(hir_id);
    while let Some(node) = tcx.hir().find(hir_id) {
        match node {
            hir::Node::Local(hir::Local {
                pat,
                init: None,
                ty: Some(ty),
                source: hir::LocalSource::Normal,
                ..
            }) => {
                err.span_label(pat.span, "this binding might not have a concrete type");
                err.span_suggestion_verbose(
                    ty.span.shrink_to_hi(),
                    "set the binding to a value for a concrete type to be resolved",
                    " = /* value */".to_string(),
                    Applicability::HasPlaceholders,
                );
            }
            hir::Node::Local(hir::Local {
                init: Some(expr),
                source: hir::LocalSource::Normal,
                ..
            }) => {
                let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
                let typeck_results =
                    tcx.typeck(tcx.hir().local_def_id(tcx.hir().get_parent_item(hir_id)));
                if let Some(ty) = typeck_results.node_type_opt(expr.hir_id) {
                    err.span_label(
                        expr.span,
                        &format!(
                            "this is of type `{}`, which doesn't constrain \
                             `{}` enough to arrive to a concrete type",
                            ty, partially_expanded_type
                        ),
                    );
                }
            }
            _ => {}
        }
        if prev_hir_id == hir_id {
            break;
        }
        prev_hir_id = hir_id;
        hir_id = tcx.hir().get_parent_node(hir_id);
    }
    err.emit();
}

fn async_opaque_type_cycle_error(tcx: TyCtxt<'tcx>, span: Span) {
    struct_span_err!(tcx.sess, span, E0733, "recursion in an `async fn` requires boxing")
        .span_label(span, "recursive `async fn`")
        .note("a recursive `async fn` must be rewritten to return a boxed `dyn Future`")
        .emit();
}

/// Checks that an opaque type does not contain cycles.
fn check_opaque_for_cycles<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    substs: SubstsRef<'tcx>,
    span: Span,
    origin: &hir::OpaqueTyOrigin,
) {
    if let Err(partially_expanded_type) = tcx.try_expand_impl_trait_type(def_id.to_def_id(), substs)
    {
        match origin {
            hir::OpaqueTyOrigin::AsyncFn => async_opaque_type_cycle_error(tcx, span),
            hir::OpaqueTyOrigin::Binding => {
                binding_opaque_type_cycle_error(tcx, def_id, span, partially_expanded_type)
            }
            _ => opaque_type_cycle_error(tcx, def_id, span),
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
        tcx.def_path_str(tcx.hir().local_def_id(it.hir_id).to_def_id())
    );
    let _indenter = indenter();
    match it.kind {
        // Consts can play a role in type-checking, so they are included here.
        hir::ItemKind::Static(..) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            tcx.ensure().typeck(def_id);
            maybe_check_static_with_link_section(tcx, def_id, it.span);
        }
        hir::ItemKind::Const(..) => {
            tcx.ensure().typeck(tcx.hir().local_def_id(it.hir_id));
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
            check_on_unimplemented(tcx, def_id.to_def_id(), it);

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
            // HACK(jynelson): trying to infer the type of `impl trait` breaks documenting
            // `async-std` (and `pub async fn` in general).
            // Since rustdoc doesn't care about the concrete type behind `impl Trait`, just don't look at it!
            // See https://github.com/rust-lang/rust/issues/75100
            if !tcx.sess.opts.actually_rustdoc {
                let def_id = tcx.hir().local_def_id(it.hir_id);

                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                check_opaque(tcx, def_id, substs, it.span, &origin);
            }
        }
        hir::ItemKind::TyAlias(..) => {
            let def_id = tcx.hir().local_def_id(it.hir_id);
            let pty_ty = tcx.type_of(def_id);
            let generics = tcx.generics_of(def_id);
            check_type_params_are_used(tcx, &generics, pty_ty);
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

fn maybe_check_static_with_link_section(tcx: TyCtxt<'_>, id: LocalDefId, span: Span) {
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
    match tcx.eval_static_initializer(id.to_def_id()) {
        Ok(alloc) => {
            if alloc.relocations().len() != 0 {
                let msg = "statics with a custom `#[link_section]` must be a \
                           simple list of bytes on the wasm target with no \
                           extra levels of indirection such as references";
                tcx.sess.span_err(span, msg);
            }
        }
        Err(_) => {}
    }
}

fn check_on_unimplemented(tcx: TyCtxt<'_>, trait_def_id: DefId, item: &hir::Item<'_>) {
    let item_def_id = tcx.hir().local_def_id(item.hir_id);
    // an error would be reported if this fails.
    let _ = traits::OnUnimplementedDirective::of_item(tcx, trait_def_id, item_def_id.to_def_id());
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
        hir::ImplItemKind::Fn(..) => ty::AssocKind::Fn,
        hir::ImplItemKind::TyAlias(_) => ty::AssocKind::Type,
    };

    let ancestors = match trait_def.ancestors(tcx, impl_id) {
        Ok(ancestors) => ancestors,
        Err(_) => return,
    };
    let mut ancestor_impls = ancestors
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
                if tcx.impl_defaultness(parent_impl.def_id()).is_default() {
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
    impl_id: LocalDefId,
    impl_trait_ref: ty::TraitRef<'tcx>,
    impl_item_refs: &[hir::ImplItemRef<'_>],
) {
    let impl_span = tcx.sess.source_map().guess_head_span(full_impl_span);

    // If the trait reference itself is erroneous (so the compilation is going
    // to fail), skip checking the items here -- the `impl_item` table in `tcx`
    // isn't populated for such impls.
    if impl_trait_ref.references_error() {
        return;
    }

    // Negative impls are not expected to have any items
    match tcx.impl_polarity(impl_id) {
        ty::ImplPolarity::Reservation | ty::ImplPolarity::Positive => {}
        ty::ImplPolarity::Negative => {
            if let [first_item_ref, ..] = impl_item_refs {
                let first_item_span = tcx.hir().impl_item(first_item_ref.id).span;
                struct_span_err!(
                    tcx.sess,
                    first_item_span,
                    E0749,
                    "negative impls cannot have any items"
                )
                .emit();
            }
            return;
        }
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
                hir::ImplItemKind::Fn(..) => {
                    let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssocKind::Fn {
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
                hir::ImplItemKind::TyAlias(_) => {
                    let opt_trait_span = tcx.hir().span_if_local(ty_trait_item.def_id);
                    if ty_trait_item.kind == ty::AssocKind::Type {
                        compare_ty_impl(
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

            check_specialization_validity(
                tcx,
                trait_def,
                &ty_trait_item,
                impl_id.to_def_id(),
                impl_item,
            );
        }
    }

    // Check for missing items from trait
    let mut missing_items = Vec::new();
    if let Ok(ancestors) = trait_def.ancestors(tcx, impl_id.to_def_id()) {
        for trait_item in tcx.associated_items(impl_trait_ref.def_id).in_definition_order() {
            let is_implemented = ancestors
                .leaf_def(tcx, trait_item.ident, trait_item.kind)
                .map(|node_item| !node_item.defining_node.is_from_trait())
                .unwrap_or(false);

            if !is_implemented && tcx.impl_defaultness(impl_id).is_final() {
                if !trait_item.defaultness.has_value() {
                    missing_items.push(*trait_item);
                }
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
fn bounds_from_generic_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: ty::GenericPredicates<'tcx>,
) -> (String, String) {
    let mut types: FxHashMap<Ty<'tcx>, Vec<DefId>> = FxHashMap::default();
    let mut projections = vec![];
    for (predicate, _) in predicates.predicates {
        debug!("predicate {:?}", predicate);
        match predicate.skip_binders() {
            ty::PredicateAtom::Trait(trait_predicate, _) => {
                let entry = types.entry(trait_predicate.self_ty()).or_default();
                let def_id = trait_predicate.def_id();
                if Some(def_id) != tcx.lang_items().sized_trait() {
                    // Type params are `Sized` by default, do not add that restriction to the list
                    // if it is a positive requirement.
                    entry.push(trait_predicate.def_id());
                }
            }
            ty::PredicateAtom::Projection(projection_pred) => {
                projections.push(ty::Binder::bind(projection_pred));
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
                .filter_map(|t| match t.kind() {
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
fn fn_sig_suggestion<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig: ty::FnSig<'tcx>,
    ident: Ident,
    predicates: ty::GenericPredicates<'tcx>,
    assoc: &ty::AssocItem,
) -> String {
    let args = sig
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            Some(match ty.kind() {
                ty::Param(_) if assoc.fn_has_self_parameter && i == 0 => "self".to_string(),
                ty::Ref(reg, ref_ty, mutability) if i == 0 => {
                    let reg = match &format!("{}", reg)[..] {
                        "'_" | "" => String::new(),
                        reg => format!("{} ", reg),
                    };
                    if assoc.fn_has_self_parameter {
                        match ref_ty.kind() {
                            ty::Param(param) if param.name == kw::SelfUpper => {
                                format!("&{}{}self", reg, mutability.prefix_str())
                            }

                            _ => format!("self: {}", ty),
                        }
                    } else {
                        format!("_: {}", ty)
                    }
                }
                _ => {
                    if assoc.fn_has_self_parameter && i == 0 {
                        format!("self: {}", ty)
                    } else {
                        format!("_: {}", ty)
                    }
                }
            })
        })
        .chain(std::iter::once(if sig.c_variadic { Some("...".to_string()) } else { None }))
        .filter_map(|arg| arg)
        .collect::<Vec<String>>()
        .join(", ");
    let output = sig.output();
    let output = if !output.is_unit() { format!(" -> {}", output) } else { String::new() };

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
        ty::AssocKind::Fn => {
            // We skip the binder here because the binder would deanonymize all
            // late-bound regions, and we don't want method signatures to show up
            // `as for<'r> fn(&'r MyType)`.  Pretty-printing handles late-bound
            // regions just fine, showing `fn(&MyType)`.
            fn_sig_suggestion(
                tcx,
                tcx.fn_sig(assoc.def_id).skip_binder(),
                assoc.ident,
                tcx.predicates_of(assoc.def_id),
                assoc,
            )
        }
        ty::AssocKind::Type => format!("type {} = Type;", assoc.ident),
        ty::AssocKind::Const => {
            let ty = tcx.type_of(assoc.def_id);
            let val = expr::ty_kind_suggestion(ty).unwrap_or("value");
            format!("const {}: {} = {};", assoc.ident, ty, val)
        }
    }
}

/// Checks whether a type can be represented in memory. In particular, it
/// identifies types that contain themselves without indirection through a
/// pointer, which would mean their size is unbounded.
fn check_representable(tcx: TyCtxt<'_>, sp: Span, item_def_id: LocalDefId) -> bool {
    let rty = tcx.type_of(item_def_id);

    // Check that it is possible to represent this type. This call identifies
    // (1) types that contain themselves and (2) types that contain a different
    // recursive type. It is only necessary to throw an error on those that
    // contain themselves. For case 2, there must be an inner type that will be
    // caught by case 1.
    match rty.is_representable(tcx, sp) {
        Representability::SelfRecursive(spans) => {
            recursive_type_with_infinite_size_error(tcx, item_def_id.to_def_id(), spans);
            return false;
        }
        Representability::Representable | Representability::ContainsRecursive => (),
    }
    true
}

pub fn check_simd(tcx: TyCtxt<'_>, sp: Span, def_id: LocalDefId) {
    let t = tcx.type_of(def_id);
    if let ty::Adt(def, substs) = t.kind() {
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
            match e.kind() {
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

fn check_packed(tcx: TyCtxt<'_>, sp: Span, def: &ty::AdtDef) {
    let repr = def.repr;
    if repr.packed() {
        for attr in tcx.get_attrs(def.did).iter() {
            for r in attr::find_repr_attrs(&tcx.sess, attr) {
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
            if let Some(def_spans) = check_packed_inner(tcx, def.did, &mut vec![]) {
                let mut err = struct_span_err!(
                    tcx.sess,
                    sp,
                    E0588,
                    "packed type cannot transitively contain a `#[repr(align)]` type"
                );

                err.span_note(
                    tcx.def_span(def_spans[0].0),
                    &format!(
                        "`{}` has a `#[repr(align)]` attribute",
                        tcx.item_name(def_spans[0].0)
                    ),
                );

                if def_spans.len() > 2 {
                    let mut first = true;
                    for (adt_def, span) in def_spans.iter().skip(1).rev() {
                        let ident = tcx.item_name(*adt_def);
                        err.span_note(
                            *span,
                            &if first {
                                format!(
                                    "`{}` contains a field of type `{}`",
                                    tcx.type_of(def.did),
                                    ident
                                )
                            } else {
                                format!("...which contains a field of type `{}`", ident)
                            },
                        );
                        first = false;
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
    if let ty::Adt(def, substs) = tcx.type_of(def_id).kind() {
        if def.is_struct() || def.is_union() {
            if def.repr.align.is_some() {
                return Some(vec![(def.did, DUMMY_SP)]);
            }

            stack.push(def_id);
            for field in &def.non_enum_variant().fields {
                if let ty::Adt(def, _) = field.ty(tcx, substs).kind() {
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

fn check_transparent<'tcx>(tcx: TyCtxt<'tcx>, sp: Span, adt: &'tcx ty::AdtDef) {
    if !adt.repr.transparent() {
        return;
    }
    let sp = tcx.sess.source_map().guess_head_span(sp);

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
        bad_variant_count(tcx, adt, sp, adt.did);
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
        let attributes = tcx.get_attrs(def_id.to_def_id());
        if let Some(attr) = tcx.sess.find_by_name(&attributes, sym::repr) {
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
            tcx.ensure().typeck(tcx.hir().local_def_id(e.hir_id));
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
            let variant_i_hir_id = tcx.hir().local_def_id_to_hir_id(variant_did.expect_local());
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
    check_transparent(tcx, sp, def);
}

fn report_unexpected_variant_res(tcx: TyCtxt<'_>, res: Res, span: Span) {
    struct_span_err!(
        tcx.sess,
        span,
        E0533,
        "expected unit struct, unit variant or constant, found {}{}",
        res.descr(),
        tcx.sess.source_map().span_to_snippet(span).map_or(String::new(), |s| format!(" `{}`", s)),
    )
    .emit();
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

fn check_type_params_are_used<'tcx>(tcx: TyCtxt<'tcx>, generics: &ty::Generics, ty: Ty<'tcx>) {
    debug!("check_type_params_are_used(generics={:?}, ty={:?})", generics, ty);

    assert_eq!(generics.parent, None);

    if generics.own_counts().types == 0 {
        return;
    }

    let mut params_used = BitSet::new_empty(generics.params.len());

    if ty.references_error() {
        // If there is already another error, do not emit
        // an error for not using a type parameter.
        assert!(tcx.sess.has_errors());
        return;
    }

    for leaf in ty.walk() {
        if let GenericArgKind::Type(leaf_ty) = leaf.unpack() {
            if let ty::Param(param) = leaf_ty.kind() {
                debug!("found use of ty param {:?}", param);
                params_used.insert(param.index);
            }
        }
    }

    for param in &generics.params {
        if !params_used.contains(param.index) {
            if let ty::GenericParamDefKind::Type { .. } = param.kind {
                let span = tcx.def_span(param.def_id);
                struct_span_err!(
                    tcx.sess,
                    span,
                    E0091,
                    "type parameter `{}` is unused",
                    param.name,
                )
                .span_label(span, "unused type parameter")
                .emit();
            }
        }
    }
}

/// A wrapper for `InferCtxt`'s `in_progress_typeck_results` field.
#[derive(Copy, Clone)]
struct MaybeInProgressTables<'a, 'tcx> {
    maybe_typeck_results: Option<&'a RefCell<ty::TypeckResults<'tcx>>>,
}

impl<'a, 'tcx> MaybeInProgressTables<'a, 'tcx> {
    fn borrow(self) -> Ref<'a, ty::TypeckResults<'tcx>> {
        match self.maybe_typeck_results {
            Some(typeck_results) => typeck_results.borrow(),
            None => bug!(
                "MaybeInProgressTables: inh/fcx.typeck_results.borrow() with no typeck results"
            ),
        }
    }

    fn borrow_mut(self) -> RefMut<'a, ty::TypeckResults<'tcx>> {
        match self.maybe_typeck_results {
            Some(typeck_results) => typeck_results.borrow_mut(),
            None => bug!(
                "MaybeInProgressTables: inh/fcx.typeck_results.borrow_mut() with no typeck results"
            ),
        }
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

fn check_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CheckItemTypesVisitor { tcx });
}

fn check_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_item_well_formed(tcx, def_id);
}

fn check_trait_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_trait_item(tcx, def_id);
}

fn check_impl_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_impl_item(tcx, def_id);
}

fn typeck_item_bodies(tcx: TyCtxt<'_>, crate_num: CrateNum) {
    debug_assert!(crate_num == LOCAL_CRATE);
    tcx.par_body_owners(|body_owner_def_id| {
        tcx.ensure().typeck(body_owner_def_id);
    });
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
        config::host_triple(),
    ));
}

fn potentially_plural_count(count: usize, word: &str) -> String {
    format!("{} {}{}", count, word, pluralize!(count))
}
