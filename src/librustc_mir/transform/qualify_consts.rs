//! A pass that qualifies constness of temporaries in constants,
//! static initializers and functions and also drives promotion.
//!
//! The Qualif flags below can be used to also provide better
//! diagnostics as to why a constant rvalue wasn't promoted.

use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;
use rustc_target::spec::abi::Abi;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::traits::{self, TraitEngine};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable};
use rustc::ty::cast::CastTy;
use rustc::ty::query::Providers;
use rustc::mir::*;
use rustc::mir::interpret::ConstValue;
use rustc::mir::traversal::ReversePostorder;
use rustc::mir::visit::{PlaceContext, Visitor, MutatingUseContext, NonMutatingUseContext};
use rustc::middle::lang_items;
use rustc::session::config::nightly_options;
use syntax::ast::LitKind;
use syntax::feature_gate::{emit_feature_err, GateIssue};
use syntax::symbol::sym;
use syntax_pos::{Span, DUMMY_SP};

use std::borrow::Cow;
use std::cell::Cell;
use std::fmt;
use std::ops::{Deref, Index, IndexMut};
use std::usize;

use rustc::hir::HirId;
use crate::transform::{MirPass, MirSource};
use super::promote_consts::{self, Candidate, TempState};
use crate::transform::check_consts::ops::{self, NonConstOp};

/// What kind of item we are in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    /// A `static` item.
    Static,
    /// A `static mut` item.
    StaticMut,
    /// A `const fn` item.
    ConstFn,
    /// A `const` item or an anonymous constant (e.g. in array lengths).
    Const,
    /// Other type of `fn`.
    NonConstFn,
}

impl Mode {
    /// Determine whether we have to do full const-checking because syntactically, we
    /// are required to be "const".
    #[inline]
    fn requires_const_checking(self) -> bool {
        self != Mode::NonConstFn
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Mode::Const => write!(f, "constant"),
            Mode::Static | Mode::StaticMut => write!(f, "static"),
            Mode::ConstFn => write!(f, "constant function"),
            Mode::NonConstFn => write!(f, "function")
        }
    }
}

const QUALIF_COUNT: usize = 4;

// FIXME(eddyb) once we can use const generics, replace this array with
// something like `IndexVec` but for fixed-size arrays (`IndexArray`?).
#[derive(Copy, Clone, Default)]
struct PerQualif<T>([T; QUALIF_COUNT]);

impl<T: Clone> PerQualif<T> {
    fn new(x: T) -> Self {
        PerQualif([x.clone(), x.clone(), x.clone(), x])
    }
}

impl<T> PerQualif<T> {
    fn as_mut(&mut self) -> PerQualif<&mut T> {
        let [x0, x1, x2, x3] = &mut self.0;
        PerQualif([x0, x1, x2, x3])
    }

    fn zip<U>(self, other: PerQualif<U>) -> PerQualif<(T, U)> {
        let [x0, x1, x2, x3] = self.0;
        let [y0, y1, y2, y3] = other.0;
        PerQualif([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
    }
}

impl PerQualif<bool> {
    fn encode_to_bits(self) -> u8 {
        self.0.iter().enumerate().fold(0, |bits, (i, &qualif)| {
            bits | ((qualif as u8) << i)
        })
    }

    fn decode_from_bits(bits: u8) -> Self {
        let mut qualifs = Self::default();
        for (i, qualif) in qualifs.0.iter_mut().enumerate() {
            *qualif = (bits & (1 << i)) != 0;
        }
        qualifs
    }
}

impl<Q: Qualif, T> Index<Q> for PerQualif<T> {
    type Output = T;

    fn index(&self, _: Q) -> &T {
        &self.0[Q::IDX]
    }
}

impl<Q: Qualif, T> IndexMut<Q> for PerQualif<T> {
    fn index_mut(&mut self, _: Q) -> &mut T {
        &mut self.0[Q::IDX]
    }
}

struct ConstCx<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mode: Mode,
    body: &'a Body<'tcx>,

    per_local: PerQualif<BitSet<Local>>,
}

impl<'a, 'tcx> ConstCx<'a, 'tcx> {
    fn is_const_panic_fn(&self, def_id: DefId) -> bool {
        Some(def_id) == self.tcx.lang_items().panic_fn() ||
        Some(def_id) == self.tcx.lang_items().begin_panic_fn()
    }
}

#[derive(Copy, Clone, Debug)]
enum ValueSource<'a, 'tcx> {
    Rvalue(&'a Rvalue<'tcx>),
    DropAndReplace(&'a Operand<'tcx>),
    Call {
        callee: &'a Operand<'tcx>,
        args: &'a [Operand<'tcx>],
        return_ty: Ty<'tcx>,
    },
}

/// A "qualif"(-ication) is a way to look for something "bad" in the MIR that would disqualify some
/// code for promotion or prevent it from evaluating at compile time. So `return true` means
/// "I found something bad, no reason to go on searching". `false` is only returned if we
/// definitely cannot find anything bad anywhere.
///
/// The default implementations proceed structurally.
trait Qualif {
    const IDX: usize;

    /// Return the qualification that is (conservatively) correct for any value
    /// of the type, or `None` if the qualification is not value/type-based.
    fn in_any_value_of_ty(_cx: &ConstCx<'_, 'tcx>, _ty: Ty<'tcx>) -> Option<bool> {
        None
    }

    /// Return a mask for the qualification, given a type. This is `false` iff
    /// no value of that type can have the qualification.
    fn mask_for_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        Self::in_any_value_of_ty(cx, ty).unwrap_or(true)
    }

    fn in_local(cx: &ConstCx<'_, '_>, local: Local) -> bool {
        cx.per_local.0[Self::IDX].contains(local)
    }

    fn in_static(_cx: &ConstCx<'_, 'tcx>, _static: &Static<'tcx>) -> bool {
        // FIXME(eddyb) should we do anything here for value properties?
        false
    }

    fn in_projection_structurally(
        cx: &ConstCx<'_, 'tcx>,
        place: PlaceRef<'_, 'tcx>,
    ) -> bool {
        if let [proj_base @ .., elem] = place.projection {
            let base_qualif = Self::in_place(cx, PlaceRef {
                base: place.base,
                projection: proj_base,
            });
            let qualif = base_qualif && Self::mask_for_ty(
                cx,
                Place::ty_from(place.base, proj_base, cx.body, cx.tcx)
                    .projection_ty(cx.tcx, elem)
                    .ty,
            );
            match elem {
                ProjectionElem::Deref |
                ProjectionElem::Subslice { .. } |
                ProjectionElem::Field(..) |
                ProjectionElem::ConstantIndex { .. } |
                ProjectionElem::Downcast(..) => qualif,

                // FIXME(eddyb) shouldn't this be masked *after* including the
                // index local? Then again, it's `usize` which is neither
                // `HasMutInterior` nor `NeedsDrop`.
                ProjectionElem::Index(local) => qualif || Self::in_local(cx, *local),
            }
        } else {
            bug!("This should be called if projection is not empty");
        }
    }

    fn in_projection(
        cx: &ConstCx<'_, 'tcx>,
        place: PlaceRef<'_, 'tcx>,
    ) -> bool {
        Self::in_projection_structurally(cx, place)
    }

    fn in_place(cx: &ConstCx<'_, 'tcx>, place: PlaceRef<'_, 'tcx>) -> bool {
        match place {
            PlaceRef {
                base: PlaceBase::Local(local),
                projection: [],
            } => Self::in_local(cx, *local),
            PlaceRef {
                base: PlaceBase::Static(box Static {
                    kind: StaticKind::Promoted(..),
                    ..
                }),
                projection: [],
            } => bug!("qualifying already promoted MIR"),
            PlaceRef {
                base: PlaceBase::Static(static_),
                projection: [],
            } => {
                Self::in_static(cx, static_)
            },
            PlaceRef {
                base: _,
                projection: [.., _],
            } => Self::in_projection(cx, place),
        }
    }

    fn in_operand(cx: &ConstCx<'_, 'tcx>, operand: &Operand<'tcx>) -> bool {
        match *operand {
            Operand::Copy(ref place) |
            Operand::Move(ref place) => Self::in_place(cx, place.as_ref()),

            Operand::Constant(ref constant) => {
                if let ConstValue::Unevaluated(def_id, _) = constant.literal.val {
                    // Don't peek inside trait associated constants.
                    if cx.tcx.trait_of_item(def_id).is_some() {
                        Self::in_any_value_of_ty(cx, constant.literal.ty).unwrap_or(false)
                    } else {
                        let (bits, _) = cx.tcx.at(constant.span).mir_const_qualif(def_id);

                        let qualif = PerQualif::decode_from_bits(bits).0[Self::IDX];

                        // Just in case the type is more specific than
                        // the definition, e.g., impl associated const
                        // with type parameters, take it into account.
                        qualif && Self::mask_for_ty(cx, constant.literal.ty)
                    }
                } else {
                    false
                }
            }
        }
    }

    fn in_rvalue_structurally(cx: &ConstCx<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> bool {
        match *rvalue {
            Rvalue::NullaryOp(..) => false,

            Rvalue::Discriminant(ref place) |
            Rvalue::Len(ref place) => Self::in_place(cx, place.as_ref()),

            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::UnaryOp(_, ref operand) |
            Rvalue::Cast(_, ref operand, _) => Self::in_operand(cx, operand),

            Rvalue::BinaryOp(_, ref lhs, ref rhs) |
            Rvalue::CheckedBinaryOp(_, ref lhs, ref rhs) => {
                Self::in_operand(cx, lhs) || Self::in_operand(cx, rhs)
            }

            Rvalue::Ref(_, _, ref place) => {
                // Special-case reborrows to be more like a copy of the reference.
                if let &[ref proj_base @ .., elem] = place.projection.as_ref() {
                    if ProjectionElem::Deref == elem {
                        let base_ty = Place::ty_from(&place.base, proj_base, cx.body, cx.tcx).ty;
                        if let ty::Ref(..) = base_ty.kind {
                            return Self::in_place(cx, PlaceRef {
                                base: &place.base,
                                projection: proj_base,
                            });
                        }
                    }
                }

                Self::in_place(cx, place.as_ref())
            }

            Rvalue::Aggregate(_, ref operands) => {
                operands.iter().any(|o| Self::in_operand(cx, o))
            }
        }
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> bool {
        Self::in_rvalue_structurally(cx, rvalue)
    }

    fn in_call(
        cx: &ConstCx<'_, 'tcx>,
        _callee: &Operand<'tcx>,
        _args: &[Operand<'tcx>],
        return_ty: Ty<'tcx>,
    ) -> bool {
        // Be conservative about the returned value of a const fn.
        Self::in_any_value_of_ty(cx, return_ty).unwrap_or(false)
    }

    fn in_value(cx: &ConstCx<'_, 'tcx>, source: ValueSource<'_, 'tcx>) -> bool {
        match source {
            ValueSource::Rvalue(rvalue) => Self::in_rvalue(cx, rvalue),
            ValueSource::DropAndReplace(source) => Self::in_operand(cx, source),
            ValueSource::Call { callee, args, return_ty } => {
                Self::in_call(cx, callee, args, return_ty)
            }
        }
    }
}

/// Constant containing interior mutability (`UnsafeCell<T>`).
/// This must be ruled out to make sure that evaluating the constant at compile-time
/// and at *any point* during the run-time would produce the same result. In particular,
/// promotion of temporaries must not change program behavior; if the promoted could be
/// written to, that would be a problem.
struct HasMutInterior;

impl Qualif for HasMutInterior {
    const IDX: usize = 0;

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<bool> {
        Some(!ty.is_freeze(cx.tcx, cx.param_env, DUMMY_SP))
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> bool {
        match *rvalue {
            // Returning `true` for `Rvalue::Ref` indicates the borrow isn't
            // allowed in constants (and the `Checker` will error), and/or it
            // won't be promoted, due to `&mut ...` or interior mutability.
            Rvalue::Ref(_, kind, ref place) => {
                let ty = place.ty(cx.body, cx.tcx).ty;

                if let BorrowKind::Mut { .. } = kind {
                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences. However, only &mut []
                    // is allowed right now, and only in functions.
                    if cx.mode == Mode::StaticMut {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        match ty.kind {
                            ty::Array(..) | ty::Slice(_) => {}
                            _ => return true,
                        }
                    } else if let ty::Array(_, len) = ty.kind {
                        // FIXME(eddyb) the `cx.mode == Mode::NonConstFn` condition
                        // seems unnecessary, given that this is merely a ZST.
                        match len.try_eval_usize(cx.tcx, cx.param_env) {
                            Some(0) if cx.mode == Mode::NonConstFn => {},
                            _ => return true,
                        }
                    } else {
                        return true;
                    }
                }
            }

            Rvalue::Aggregate(ref kind, _) => {
                if let AggregateKind::Adt(def, ..) = **kind {
                    if Some(def.did) == cx.tcx.lang_items().unsafe_cell_type() {
                        let ty = rvalue.ty(cx.body, cx.tcx);
                        assert_eq!(Self::in_any_value_of_ty(cx, ty), Some(true));
                        return true;
                    }
                }
            }

            _ => {}
        }

        Self::in_rvalue_structurally(cx, rvalue)
    }
}

/// Constant containing an ADT that implements `Drop`.
/// This must be ruled out (a) because we cannot run `Drop` during compile-time
/// as that might not be a `const fn`, and (b) because implicit promotion would
/// remove side-effects that occur as part of dropping that value.
struct NeedsDrop;

impl Qualif for NeedsDrop {
    const IDX: usize = 1;

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<bool> {
        Some(ty.needs_drop(cx.tcx, cx.param_env))
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> bool {
        if let Rvalue::Aggregate(ref kind, _) = *rvalue {
            if let AggregateKind::Adt(def, ..) = **kind {
                if def.has_dtor(cx.tcx) {
                    return true;
                }
            }
        }

        Self::in_rvalue_structurally(cx, rvalue)
    }
}

/// Not promotable at all - non-`const fn` calls, `asm!`,
/// pointer comparisons, ptr-to-int casts, etc.
/// Inside a const context all constness rules apply, so promotion simply has to follow the regular
/// constant rules (modulo interior mutability or `Drop` rules which are handled `HasMutInterior`
/// and `NeedsDrop` respectively). Basically this duplicates the checks that the const-checking
/// visitor enforces by emitting errors when working in const context.
struct IsNotPromotable;

impl Qualif for IsNotPromotable {
    const IDX: usize = 2;

    fn in_static(cx: &ConstCx<'_, 'tcx>, static_: &Static<'tcx>) -> bool {
        match static_.kind {
            StaticKind::Promoted(_, _) => unreachable!(),
            StaticKind::Static => {
                // Only allow statics (not consts) to refer to other statics.
                // FIXME(eddyb) does this matter at all for promotion?
                let allowed = cx.mode == Mode::Static || cx.mode == Mode::StaticMut;

                !allowed ||
                    cx.tcx.get_attrs(static_.def_id).iter().any(
                        |attr| attr.check_name(sym::thread_local)
                    )
            }
        }
    }

    fn in_projection(
        cx: &ConstCx<'_, 'tcx>,
        place: PlaceRef<'_, 'tcx>,
    ) -> bool {
        if let [proj_base @ .., elem] = place.projection {
            match elem {
                ProjectionElem::Deref |
                ProjectionElem::Downcast(..) => return true,

                ProjectionElem::ConstantIndex {..} |
                ProjectionElem::Subslice {..} |
                ProjectionElem::Index(_) => {}

                ProjectionElem::Field(..) => {
                    if cx.mode == Mode::NonConstFn {
                        let base_ty = Place::ty_from(place.base, proj_base, cx.body, cx.tcx).ty;
                        if let Some(def) = base_ty.ty_adt_def() {
                            // No promotion of union field accesses.
                            if def.is_union() {
                                return true;
                            }
                        }
                    }
                }
            }

            Self::in_projection_structurally(cx, place)
        } else {
            bug!("This should be called if projection is not empty");
        }
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> bool {
        match *rvalue {
            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) if cx.mode == Mode::NonConstFn => {
                let operand_ty = operand.ty(cx.body, cx.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) => {
                        // in normal functions, mark such casts as not promotable
                        return true;
                    }
                    _ => {}
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) if cx.mode == Mode::NonConstFn => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(cx.body, cx.tcx).kind {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    // raw pointer operations are not allowed inside promoteds
                    return true;
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => return true,

            _ => {}
        }

        Self::in_rvalue_structurally(cx, rvalue)
    }

    fn in_call(
        cx: &ConstCx<'_, 'tcx>,
        callee: &Operand<'tcx>,
        args: &[Operand<'tcx>],
        _return_ty: Ty<'tcx>,
    ) -> bool {
        let fn_ty = callee.ty(cx.body, cx.tcx);
        match fn_ty.kind {
            ty::FnDef(def_id, _) => {
                match cx.tcx.fn_sig(def_id).abi() {
                    Abi::RustIntrinsic |
                    Abi::PlatformIntrinsic => {
                        assert!(!cx.tcx.is_const_fn(def_id));
                        match &*cx.tcx.item_name(def_id).as_str() {
                            | "size_of"
                            | "min_align_of"
                            | "needs_drop"
                            | "type_id"
                            | "bswap"
                            | "bitreverse"
                            | "ctpop"
                            | "cttz"
                            | "cttz_nonzero"
                            | "ctlz"
                            | "ctlz_nonzero"
                            | "wrapping_add"
                            | "wrapping_sub"
                            | "wrapping_mul"
                            | "unchecked_shl"
                            | "unchecked_shr"
                            | "rotate_left"
                            | "rotate_right"
                            | "add_with_overflow"
                            | "sub_with_overflow"
                            | "mul_with_overflow"
                            | "saturating_add"
                            | "saturating_sub"
                            | "transmute"
                            | "simd_insert"
                            | "simd_extract"
                            | "ptr_offset_from"
                            => return true,

                            _ => {}
                        }
                    }
                    _ => {
                        let is_const_fn =
                            cx.tcx.is_const_fn(def_id) ||
                            cx.tcx.is_unstable_const_fn(def_id).is_some() ||
                            cx.is_const_panic_fn(def_id);
                        if !is_const_fn {
                            return true;
                        }
                    }
                }
            }
            _ => return true,
        }

        Self::in_operand(cx, callee) || args.iter().any(|arg| Self::in_operand(cx, arg))
    }
}

/// Refers to temporaries which cannot be promoted *implicitly*.
/// Explicit promotion happens e.g. for constant arguments declared via `rustc_args_required_const`.
/// Implicit promotion has almost the same rules, except that disallows `const fn` except for
/// those marked `#[rustc_promotable]`. This is to avoid changing a legitimate run-time operation
/// into a failing compile-time operation e.g. due to addresses being compared inside the function.
struct IsNotImplicitlyPromotable;

impl Qualif for IsNotImplicitlyPromotable {
    const IDX: usize = 3;

    fn in_call(
        cx: &ConstCx<'_, 'tcx>,
        callee: &Operand<'tcx>,
        args: &[Operand<'tcx>],
        _return_ty: Ty<'tcx>,
    ) -> bool {
        if cx.mode == Mode::NonConstFn {
            if let ty::FnDef(def_id, _) = callee.ty(cx.body, cx.tcx).kind {
                // Never promote runtime `const fn` calls of
                // functions without `#[rustc_promotable]`.
                if !cx.tcx.is_promotable_const_fn(def_id) {
                    return true;
                }
            }
        }

        Self::in_operand(cx, callee) || args.iter().any(|arg| Self::in_operand(cx, arg))
    }
}

// Ensure the `IDX` values are sequential (`0..QUALIF_COUNT`).
macro_rules! static_assert_seq_qualifs {
    ($i:expr => $first:ident $(, $rest:ident)*) => {
        static_assert!({
            static_assert_seq_qualifs!($i + 1 => $($rest),*);

            $first::IDX == $i
        });
    };
    ($i:expr =>) => {
        static_assert!(QUALIF_COUNT == $i);
    };
}
static_assert_seq_qualifs!(
    0 => HasMutInterior, NeedsDrop, IsNotPromotable, IsNotImplicitlyPromotable
);

impl ConstCx<'_, 'tcx> {
    fn qualifs_in_any_value_of_ty(&self, ty: Ty<'tcx>) -> PerQualif<bool> {
        let mut qualifs = PerQualif::default();
        qualifs[HasMutInterior] = HasMutInterior::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs[NeedsDrop] = NeedsDrop::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs[IsNotPromotable] = IsNotPromotable::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs[IsNotImplicitlyPromotable] =
            IsNotImplicitlyPromotable::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs
    }

    fn qualifs_in_local(&self, local: Local) -> PerQualif<bool> {
        let mut qualifs = PerQualif::default();
        qualifs[HasMutInterior] = HasMutInterior::in_local(self, local);
        qualifs[NeedsDrop] = NeedsDrop::in_local(self, local);
        qualifs[IsNotPromotable] = IsNotPromotable::in_local(self, local);
        qualifs[IsNotImplicitlyPromotable] = IsNotImplicitlyPromotable::in_local(self, local);
        qualifs
    }

    fn qualifs_in_value(&self, source: ValueSource<'_, 'tcx>) -> PerQualif<bool> {
        let mut qualifs = PerQualif::default();
        qualifs[HasMutInterior] = HasMutInterior::in_value(self, source);
        qualifs[NeedsDrop] = NeedsDrop::in_value(self, source);
        qualifs[IsNotPromotable] = IsNotPromotable::in_value(self, source);
        qualifs[IsNotImplicitlyPromotable] = IsNotImplicitlyPromotable::in_value(self, source);
        qualifs
    }
}

/// Checks MIR for being admissible as a compile-time constant, using `ConstCx`
/// for value qualifications, and accumulates writes of
/// rvalue/call results to locals, in `local_qualif`.
/// It also records candidates for promotion in `promotion_candidates`,
/// both in functions and const/static items.
struct Checker<'a, 'tcx> {
    cx: ConstCx<'a, 'tcx>,

    span: Span,
    def_id: DefId,
    rpo: ReversePostorder<'a, 'tcx>,

    temp_promotion_state: IndexVec<Local, TempState>,
    promotion_candidates: Vec<Candidate>,
    unchecked_promotion_candidates: Vec<Candidate>,

    /// If `true`, do not emit errors to the user, merely collect them in `errors`.
    suppress_errors: bool,
    errors: Vec<(Span, String)>,
}

macro_rules! unleash_miri {
    ($this:expr) => {{
        if $this.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
            if $this.mode.requires_const_checking() && !$this.suppress_errors {
                $this.tcx.sess.span_warn($this.span, "skipping const checks");
            }
            return;
        }
    }}
}

impl Deref for Checker<'a, 'tcx> {
    type Target = ConstCx<'a, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.cx
    }
}

impl<'a, 'tcx> Checker<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: DefId, body: &'a Body<'tcx>, mode: Mode) -> Self {
        assert!(def_id.is_local());
        let mut rpo = traversal::reverse_postorder(body);
        let (temps, unchecked_promotion_candidates) =
            promote_consts::collect_temps_and_candidates(tcx, body, &mut rpo);
        rpo.reset();

        let param_env = tcx.param_env(def_id);

        let mut cx = ConstCx {
            tcx,
            param_env,
            mode,
            body,
            per_local: PerQualif::new(BitSet::new_empty(body.local_decls.len())),
        };

        for (local, decl) in body.local_decls.iter_enumerated() {
            if let LocalKind::Arg = body.local_kind(local) {
                let qualifs = cx.qualifs_in_any_value_of_ty(decl.ty);
                for (per_local, qualif) in &mut cx.per_local.as_mut().zip(qualifs).0 {
                    if *qualif {
                        per_local.insert(local);
                    }
                }
            }
            if !temps[local].is_promotable() {
                cx.per_local[IsNotPromotable].insert(local);
            }
            if let LocalKind::Var = body.local_kind(local) {
                // Sanity check to prevent implicit and explicit promotion of
                // named locals
                assert!(cx.per_local[IsNotPromotable].contains(local));
            }
        }

        Checker {
            cx,
            span: body.span,
            def_id,
            rpo,
            temp_promotion_state: temps,
            promotion_candidates: vec![],
            unchecked_promotion_candidates,
            errors: vec![],
            suppress_errors: false,
        }
    }

    // FIXME(eddyb) we could split the errors into meaningful
    // categories, but enabling full miri would make that
    // slightly pointless (even with feature-gating).
    fn not_const(&mut self, op: impl NonConstOp) {
        unleash_miri!(self);
        if self.mode.requires_const_checking() && !self.suppress_errors {
            self.record_error(op);
            let mut err = struct_span_err!(
                self.tcx.sess,
                self.span,
                E0019,
                "{} contains unimplemented expression type",
                self.mode
            );
            if self.tcx.sess.teach(&err.get_code().unwrap()) {
                err.note("A function call isn't allowed in the const's initialization expression \
                          because the expression's value must be known at compile-time.");
                err.note("Remember: you can't use a function call inside a const's initialization \
                          expression! However, you can use it anywhere else.");
            }
            err.emit();
        }
    }

    fn record_error(&mut self, op: impl NonConstOp) {
        self.record_error_spanned(op, self.span);
    }

    fn record_error_spanned(&mut self, op: impl NonConstOp, span: Span) {
        self.errors.push((span, format!("{:?}", op)));
    }

    /// Assigns an rvalue/call qualification to the given destination.
    fn assign(&mut self, dest: &Place<'tcx>, source: ValueSource<'_, 'tcx>, location: Location) {
        trace!("assign: {:?} <- {:?}", dest, source);

        let mut qualifs = self.qualifs_in_value(source);

        match source {
            ValueSource::Rvalue(&Rvalue::Ref(_, kind, ref place)) => {
                // Getting `true` from `HasMutInterior::in_rvalue` means
                // the borrowed place is disallowed from being borrowed,
                // due to either a mutable borrow (with some exceptions),
                // or an shared borrow of a value with interior mutability.
                // Then `HasMutInterior` is replaced with `IsNotPromotable`,
                // to avoid duplicate errors (e.g. from reborrowing).
                if qualifs[HasMutInterior] {
                    qualifs[HasMutInterior] = false;
                    qualifs[IsNotPromotable] = true;

                    debug!("suppress_errors: {}", self.suppress_errors);
                    if self.mode.requires_const_checking() && !self.suppress_errors {
                        if !self.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
                            self.record_error(ops::MutBorrow(kind));
                            if let BorrowKind::Mut { .. } = kind {
                                let mut err = struct_span_err!(self.tcx.sess,  self.span, E0017,
                                                               "references in {}s may only refer \
                                                                to immutable values", self.mode);
                                err.span_label(self.span, format!("{}s require immutable values",
                                                                    self.mode));
                                if self.tcx.sess.teach(&err.get_code().unwrap()) {
                                    err.note("References in statics and constants may only refer \
                                              to immutable values.\n\n\
                                              Statics are shared everywhere, and if they refer to \
                                              mutable data one might violate memory safety since \
                                              holding multiple mutable references to shared data \
                                              is not allowed.\n\n\
                                              If you really want global mutable state, try using \
                                              static mut or a global UnsafeCell.");
                                }
                                err.emit();
                            } else {
                                span_err!(self.tcx.sess, self.span, E0492,
                                          "cannot borrow a constant which may contain \
                                           interior mutability, create a static instead");
                            }
                        }
                    }
                } else if let BorrowKind::Mut { .. } | BorrowKind::Shared = kind {
                    // Don't promote BorrowKind::Shallow borrows, as they don't
                    // reach codegen.
                    // FIXME(eddyb) the two other kinds of borrow (`Shallow` and `Unique`)
                    // aren't promoted here but *could* be promoted as part of a larger
                    // value because `IsNotPromotable` isn't being set for them,
                    // need to figure out what is the intended behavior.

                    // We might have a candidate for promotion.
                    let candidate = Candidate::Ref(location);
                    // Start by traversing to the "base", with non-deref projections removed.
                    let deref_proj =
                        place.projection.iter().rev().find(|&elem| *elem == ProjectionElem::Deref);

                    debug!(
                        "qualify_consts: promotion candidate: place={:?} {:?}",
                        place.base, deref_proj
                    );
                    // We can only promote interior borrows of promotable temps (non-temps
                    // don't get promoted anyway).
                    // (If we bailed out of the loop due to a `Deref` above, we will definitely
                    // not enter the conditional here.)
                    if let (PlaceBase::Local(local), None) = (&place.base, deref_proj) {
                        if self.body.local_kind(*local) == LocalKind::Temp {
                            debug!("qualify_consts: promotion candidate: local={:?}", local);
                            // The borrowed place doesn't have `HasMutInterior`
                            // (from `in_rvalue`), so we can safely ignore
                            // `HasMutInterior` from the local's qualifications.
                            // This allows borrowing fields which don't have
                            // `HasMutInterior`, from a type that does, e.g.:
                            // `let _: &'static _ = &(Cell::new(1), 2).1;`
                            let mut local_qualifs = self.qualifs_in_local(*local);
                            // Any qualifications, except HasMutInterior (see above), disqualify
                            // from promotion.
                            // This is, in particular, the "implicit promotion" version of
                            // the check making sure that we don't run drop glue during const-eval.
                            local_qualifs[HasMutInterior] = false;
                            if !local_qualifs.0.iter().any(|&qualif| qualif) {
                                debug!("qualify_consts: promotion candidate: {:?}", candidate);
                                self.promotion_candidates.push(candidate);
                            }
                        }
                    }
                }
            },
            ValueSource::Rvalue(&Rvalue::Repeat(ref operand, _)) => {
                debug!("assign: self.cx.mode={:?} self.def_id={:?} location={:?} operand={:?}",
                       self.cx.mode, self.def_id, location, operand);
                if self.should_promote_repeat_expression(operand) &&
                        self.tcx.features().const_in_array_repeat_expressions {
                    self.promotion_candidates.push(Candidate::Repeat(location));
                }
            },
            _ => {},
        }

        let mut dest_projection = &dest.projection[..];
        let index = loop {
            match (&dest.base, dest_projection) {
                // We treat all locals equal in constants
                (&PlaceBase::Local(index), []) => break index,
                // projections are transparent for assignments
                // we qualify the entire destination at once, even if just a field would have
                // stricter qualification
                (base, [proj_base @ .., _]) => {
                    // Catch more errors in the destination. `visit_place` also checks various
                    // projection rules like union field access and raw pointer deref
                    let context = PlaceContext::MutatingUse(MutatingUseContext::Store);
                    self.visit_place_base(base, context, location);
                    self.visit_projection(base, dest_projection, context, location);
                    dest_projection = proj_base;
                },
                (&PlaceBase::Static(box Static {
                    kind: StaticKind::Promoted(..),
                    ..
                }), []) => bug!("promoteds don't exist yet during promotion"),
                (&PlaceBase::Static(box Static{ kind: _, .. }), []) => {
                    // Catch more errors in the destination. `visit_place` also checks that we
                    // do not try to access statics from constants or try to mutate statics
                    let context = PlaceContext::MutatingUse(MutatingUseContext::Store);
                    self.visit_place_base(&dest.base, context, location);
                    return;
                }
            }
        };

        let kind = self.body.local_kind(index);
        debug!("store to {:?} {:?}", kind, index);

        // Only handle promotable temps in non-const functions.
        if self.mode == Mode::NonConstFn {
            if kind != LocalKind::Temp ||
               !self.temp_promotion_state[index].is_promotable() {
                return;
            }
        }

        // this is overly restrictive, because even full assignments do not clear the qualif
        // While we could special case full assignments, this would be inconsistent with
        // aggregates where we overwrite all fields via assignments, which would not get
        // that feature.
        for (per_local, qualif) in &mut self.cx.per_local.as_mut().zip(qualifs).0 {
            if *qualif {
                per_local.insert(index);
            }
        }

        // Ensure the `IsNotPromotable` qualification is preserved.
        // NOTE(eddyb) this is actually unnecessary right now, as
        // we never replace the local's qualif, but we might in
        // the future, and so it serves to catch changes that unset
        // important bits (in which case, asserting `contains` could
        // be replaced with calling `insert` to re-set the bit).
        if kind == LocalKind::Temp {
            if !self.temp_promotion_state[index].is_promotable() {
                assert!(self.cx.per_local[IsNotPromotable].contains(index));
            }
        }
    }

    /// Check a whole const, static initializer or const fn.
    fn check_const(&mut self) -> (u8, &'tcx BitSet<Local>) {
        use crate::transform::check_consts as new_checker;

        debug!("const-checking {} {:?}", self.mode, self.def_id);

        // FIXME: Also use the new validator when features that require it (e.g. `const_if`) are
        // enabled.
        let use_new_validator = self.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you;
        if use_new_validator {
            debug!("Using dataflow-based const validator");
        }

        let item = new_checker::Item::new(self.tcx, self.def_id, self.body);
        let mut validator = new_checker::validation::Validator::new(&item);

        validator.suppress_errors = !use_new_validator;
        self.suppress_errors = use_new_validator;

        let body = self.body;

        let mut seen_blocks = BitSet::new_empty(body.basic_blocks().len());
        let mut bb = START_BLOCK;
        let mut has_controlflow_error = false;
        loop {
            seen_blocks.insert(bb.index());

            self.visit_basic_block_data(bb, &body[bb]);
            validator.visit_basic_block_data(bb, &body[bb]);

            let target = match body[bb].terminator().kind {
                TerminatorKind::Goto { target } |
                TerminatorKind::FalseUnwind { real_target: target, .. } |
                TerminatorKind::Drop { target, .. } |
                TerminatorKind::DropAndReplace { target, .. } |
                TerminatorKind::Assert { target, .. } |
                TerminatorKind::Call { destination: Some((_, target)), .. } => {
                    Some(target)
                }

                // Non-terminating calls cannot produce any value.
                TerminatorKind::Call { destination: None, .. } => {
                    break;
                }

                TerminatorKind::SwitchInt {..} |
                TerminatorKind::Resume |
                TerminatorKind::Abort |
                TerminatorKind::GeneratorDrop |
                TerminatorKind::Yield { .. } |
                TerminatorKind::Unreachable |
                TerminatorKind::FalseEdges { .. } => None,

                TerminatorKind::Return => {
                    break;
                }
            };

            match target {
                // No loops allowed.
                Some(target) if !seen_blocks.contains(target.index()) => {
                    bb = target;
                }
                _ => {
                    has_controlflow_error = true;
                    self.not_const(ops::Loop);
                    validator.check_op(ops::Loop);
                    break;
                }
            }
        }

        // The new validation pass should agree with the old when running on simple const bodies
        // (e.g. no `if` or `loop`).
        if !use_new_validator {
            let mut new_errors = validator.take_errors();

            // FIXME: each checker sometimes emits the same error with the same span twice in a row.
            self.errors.dedup();
            new_errors.dedup();

            if self.errors != new_errors {
                validator_mismatch(
                    self.tcx,
                    body,
                    std::mem::replace(&mut self.errors, vec![]),
                    new_errors,
                );
            }
        }

        // Collect all the temps we need to promote.
        let mut promoted_temps = BitSet::new_empty(self.temp_promotion_state.len());

        // HACK(eddyb) don't try to validate promotion candidates if any
        // parts of the control-flow graph were skipped due to an error.
        let promotion_candidates = if has_controlflow_error {
            let unleash_miri = self
                .tcx
                .sess
                .opts
                .debugging_opts
                .unleash_the_miri_inside_of_you;
            if !unleash_miri {
                self.tcx.sess.delay_span_bug(
                    body.span,
                    "check_const: expected control-flow error(s)",
                );
            }
            self.promotion_candidates.clone()
        } else {
            self.valid_promotion_candidates()
        };

        debug!("qualify_const: promotion_candidates={:?}", promotion_candidates);
        for candidate in promotion_candidates {
            match candidate {
                Candidate::Ref(Location { block: bb, statement_index: stmt_idx }) => {
                    if let StatementKind::Assign(box( _, Rvalue::Ref(_, _, place)))
                        = &self.body[bb].statements[stmt_idx].kind
                    {
                        if let PlaceBase::Local(local) = place.base {
                            promoted_temps.insert(local);
                        }
                    }
                }

                // Only rvalue-static promotion requires extending the lifetime of the promoted
                // local.
                Candidate::Argument { .. } | Candidate::Repeat(_) => {}
            }
        }

        let mut qualifs = self.qualifs_in_local(RETURN_PLACE);

        // Account for errors in consts by using the
        // conservative type qualification instead.
        if qualifs[IsNotPromotable] {
            qualifs = self.qualifs_in_any_value_of_ty(body.return_ty());
        }

        (qualifs.encode_to_bits(), self.tcx.arena.alloc(promoted_temps))
    }

    /// Get the subset of `unchecked_promotion_candidates` that are eligible
    /// for promotion.
    // FIXME(eddyb) replace the old candidate gathering with this.
    fn valid_promotion_candidates(&self) -> Vec<Candidate> {
        // Sanity-check the promotion candidates.
        let candidates = promote_consts::validate_candidates(
            self.tcx,
            self.body,
            self.def_id,
            &self.temp_promotion_state,
            &self.unchecked_promotion_candidates,
        );

        if candidates != self.promotion_candidates {
            let report = |msg, candidate| {
                let span = match candidate {
                    Candidate::Ref(loc) |
                    Candidate::Repeat(loc) => self.body.source_info(loc).span,
                    Candidate::Argument { bb, .. } => {
                        self.body[bb].terminator().source_info.span
                    }
                };
                self.tcx.sess.span_err(span, &format!("{}: {:?}", msg, candidate));
            };

            for &c in &self.promotion_candidates {
                if !candidates.contains(&c) {
                    report("invalidated old candidate", c);
                }
            }

            for &c in &candidates {
                if !self.promotion_candidates.contains(&c) {
                    report("extra new candidate", c);
                }
            }

            bug!("promotion candidate validation mismatches (see above)");
        }

        candidates
    }

    /// Returns `true` if the operand of a repeat expression is promotable.
    fn should_promote_repeat_expression(&self, operand: &Operand<'tcx>) -> bool {
        let not_promotable = IsNotImplicitlyPromotable::in_operand(self, operand) ||
                             IsNotPromotable::in_operand(self, operand);
        debug!("should_promote_repeat_expression: operand={:?} not_promotable={:?}",
               operand, not_promotable);
        !not_promotable
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Checker<'a, 'tcx> {
    fn visit_place_base(
        &mut self,
        place_base: &PlaceBase<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        self.super_place_base(place_base, context, location);
        match place_base {
            PlaceBase::Local(_) => {}
            PlaceBase::Static(box Static{ kind: StaticKind::Promoted(_, _), .. }) => {
                unreachable!()
            }
            PlaceBase::Static(box Static{ kind: StaticKind::Static, def_id, .. }) => {
                if self.tcx
                        .get_attrs(*def_id)
                        .iter()
                        .any(|attr| attr.check_name(sym::thread_local)) {
                    if self.mode.requires_const_checking() && !self.suppress_errors {
                        self.record_error(ops::ThreadLocalAccess);
                        span_err!(self.tcx.sess, self.span, E0625,
                                    "thread-local statics cannot be \
                                    accessed at compile-time");
                    }
                    return;
                }

                // Only allow statics (not consts) to refer to other statics.
                if self.mode == Mode::Static || self.mode == Mode::StaticMut {
                    if self.mode == Mode::Static
                        && context.is_mutating_use()
                        && !self.suppress_errors
                    {
                        // this is not strictly necessary as miri will also bail out
                        // For interior mutability we can't really catch this statically as that
                        // goes through raw pointers and intermediate temporaries, so miri has
                        // to catch this anyway
                        self.tcx.sess.span_err(
                            self.span,
                            "cannot mutate statics in the initializer of another static",
                        );
                    }
                    return;
                }
                unleash_miri!(self);

                if self.mode.requires_const_checking() && !self.suppress_errors {
                    self.record_error(ops::StaticAccess);
                    let mut err = struct_span_err!(self.tcx.sess, self.span, E0013,
                                                    "{}s cannot refer to statics, use \
                                                    a constant instead", self.mode);
                    if self.tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(
                            "Static and const variables can refer to other const variables. \
                                But a const variable cannot refer to a static variable."
                        );
                        err.help(
                            "To fix this, the value can be extracted as a const and then used."
                        );
                    }
                    err.emit()
                }
            }
        }
    }

    fn visit_projection_elem(
        &mut self,
        place_base: &PlaceBase<'tcx>,
        proj_base: &[PlaceElem<'tcx>],
        elem: &PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        debug!(
            "visit_projection_elem: place_base={:?} proj_base={:?} elem={:?} \
            context={:?} location={:?}",
            place_base,
            proj_base,
            elem,
            context,
            location,
        );

        self.super_projection_elem(place_base, proj_base, elem, context, location);

        match elem {
            ProjectionElem::Deref => {
                if context.is_mutating_use() {
                    // `not_const` errors out in const contexts
                    self.not_const(ops::MutDeref)
                }
                let base_ty = Place::ty_from(place_base, proj_base, self.body, self.tcx).ty;
                match self.mode {
                    Mode::NonConstFn => {}
                    _ if self.suppress_errors => {}
                    _ => {
                        if let ty::RawPtr(_) = base_ty.kind {
                            if !self.tcx.features().const_raw_ptr_deref {
                                self.record_error(ops::RawPtrDeref);
                                emit_feature_err(
                                    &self.tcx.sess.parse_sess, sym::const_raw_ptr_deref,
                                    self.span, GateIssue::Language,
                                    &format!(
                                        "dereferencing raw pointers in {}s is unstable",
                                        self.mode,
                                    ),
                                );
                            }
                        }
                    }
                }
            }

            ProjectionElem::ConstantIndex {..} |
            ProjectionElem::Subslice {..} |
            ProjectionElem::Field(..) |
            ProjectionElem::Index(_) => {
                let base_ty = Place::ty_from(place_base, proj_base, self.body, self.tcx).ty;
                if let Some(def) = base_ty.ty_adt_def() {
                    if def.is_union() {
                        match self.mode {
                            Mode::ConstFn => {
                                if !self.tcx.features().const_fn_union
                                    && !self.suppress_errors
                                {
                                    self.record_error(ops::UnionAccess);
                                    emit_feature_err(
                                        &self.tcx.sess.parse_sess, sym::const_fn_union,
                                        self.span, GateIssue::Language,
                                        "unions in const fn are unstable",
                                    );
                                }
                            },

                            | Mode::NonConstFn
                            | Mode::Static
                            | Mode::StaticMut
                            | Mode::Const
                            => {},
                        }
                    }
                }
            }

            ProjectionElem::Downcast(..) => {
                self.not_const(ops::Downcast)
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        debug!("visit_operand: operand={:?} location={:?}", operand, location);
        self.super_operand(operand, location);

        match *operand {
            Operand::Move(ref place) => {
                // Mark the consumed locals to indicate later drops are noops.
                if let Some(local) = place.as_local() {
                    self.cx.per_local[NeedsDrop].remove(local);
                }
            }
            Operand::Copy(_) |
            Operand::Constant(_) => {}
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        debug!("visit_rvalue: rvalue={:?} location={:?}", rvalue, location);

        // Check nested operands and places.
        if let Rvalue::Ref(_, kind, ref place) = *rvalue {
            // Special-case reborrows.
            let mut reborrow_place = None;
            if let &[ref proj_base @ .., elem] = place.projection.as_ref() {
                if elem == ProjectionElem::Deref {
                    let base_ty = Place::ty_from(&place.base, proj_base, self.body, self.tcx).ty;
                    if let ty::Ref(..) = base_ty.kind {
                        reborrow_place = Some(proj_base);
                    }
                }
            }

            if let Some(proj) = reborrow_place {
                let ctx = match kind {
                    BorrowKind::Shared => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::SharedBorrow,
                    ),
                    BorrowKind::Shallow => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::ShallowBorrow,
                    ),
                    BorrowKind::Unique => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::UniqueBorrow,
                    ),
                    BorrowKind::Mut { .. } => PlaceContext::MutatingUse(
                        MutatingUseContext::Borrow,
                    ),
                };
                self.visit_place_base(&place.base, ctx, location);
                self.visit_projection(&place.base, proj, ctx, location);
            } else {
                self.super_rvalue(rvalue, location);
            }
        } else {
            self.super_rvalue(rvalue, location);
        }

        match *rvalue {
            Rvalue::Use(_) |
            Rvalue::Repeat(..) |
            Rvalue::UnaryOp(UnOp::Neg, _) |
            Rvalue::UnaryOp(UnOp::Not, _) |
            Rvalue::NullaryOp(NullOp::SizeOf, _) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::Cast(CastKind::Pointer(_), ..) |
            Rvalue::Discriminant(..) |
            Rvalue::Len(_) |
            Rvalue::Ref(..) |
            Rvalue::Aggregate(..) => {}

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) if self.mode != Mode::NonConstFn => {
                        unleash_miri!(self);
                        if !self.tcx.features().const_raw_ptr_to_usize_cast
                            && !self.suppress_errors
                        {
                            // in const fn and constants require the feature gate
                            // FIXME: make it unsafe inside const fn and constants
                            self.record_error(ops::RawPtrToIntCast);
                            emit_feature_err(
                                &self.tcx.sess.parse_sess, sym::const_raw_ptr_to_usize_cast,
                                self.span, GateIssue::Language,
                                &format!(
                                    "casting pointers to integers in {}s is unstable",
                                    self.mode,
                                ),
                            );
                        }
                    }
                    _ => {}
                }
            }

            Rvalue::BinaryOp(op, ref lhs, _) => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.body, self.tcx).kind {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    unleash_miri!(self);
                    if self.mode.requires_const_checking() &&
                        !self.tcx.features().const_compare_raw_pointers &&
                        !self.suppress_errors
                    {
                        self.record_error(ops::RawPtrComparison);
                        // require the feature gate inside constants and const fn
                        // FIXME: make it unsafe to use these operations
                        emit_feature_err(
                            &self.tcx.sess.parse_sess,
                            sym::const_compare_raw_pointers,
                            self.span,
                            GateIssue::Language,
                            &format!("comparing raw pointers inside {}", self.mode),
                        );
                    }
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => {
                unleash_miri!(self);
                if self.mode.requires_const_checking() && !self.suppress_errors {
                    self.record_error(ops::HeapAllocation);
                    let mut err = struct_span_err!(self.tcx.sess, self.span, E0010,
                                                   "allocations are not allowed in {}s", self.mode);
                    err.span_label(self.span, format!("allocation not allowed in {}s", self.mode));
                    if self.tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(
                            "The value of statics and constants must be known at compile time, \
                             and they live for the entire lifetime of a program. Creating a boxed \
                             value allocates memory on the heap at runtime, and therefore cannot \
                             be done at compile time."
                        );
                    }
                    err.emit();
                }
            }
        }
    }

    fn visit_terminator_kind(&mut self,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        debug!("visit_terminator_kind: kind={:?} location={:?}", kind, location);
        if let TerminatorKind::Call { ref func, ref args, ref destination, .. } = *kind {
            if let Some((ref dest, _)) = *destination {
                self.assign(dest, ValueSource::Call {
                    callee: func,
                    args,
                    return_ty: dest.ty(self.body, self.tcx).ty,
                }, location);
            }

            let fn_ty = func.ty(self.body, self.tcx);
            let mut callee_def_id = None;
            let mut is_shuffle = false;
            match fn_ty.kind {
                ty::FnDef(def_id, _) => {
                    callee_def_id = Some(def_id);
                    match self.tcx.fn_sig(def_id).abi() {
                        Abi::RustIntrinsic |
                        Abi::PlatformIntrinsic => {
                            assert!(!self.tcx.is_const_fn(def_id));
                            match &*self.tcx.item_name(def_id).as_str() {
                                // special intrinsic that can be called diretly without an intrinsic
                                // feature gate needs a language feature gate
                                "transmute" => {
                                    if self.mode.requires_const_checking()
                                        && !self.suppress_errors
                                    {
                                        // const eval transmute calls only with the feature gate
                                        if !self.tcx.features().const_transmute {
                                            self.record_error(ops::Transmute);
                                            emit_feature_err(
                                                &self.tcx.sess.parse_sess, sym::const_transmute,
                                                self.span, GateIssue::Language,
                                                &format!("The use of std::mem::transmute() \
                                                is gated in {}s", self.mode));
                                        }
                                    }
                                }

                                name if name.starts_with("simd_shuffle") => {
                                    is_shuffle = true;
                                }

                                // no need to check feature gates, intrinsics are only callable
                                // from the libstd or with forever unstable feature gates
                                _ => {}
                            }
                        }
                        _ => {
                            // In normal functions no calls are feature-gated.
                            if self.mode.requires_const_checking() {
                                let unleash_miri = self
                                    .tcx
                                    .sess
                                    .opts
                                    .debugging_opts
                                    .unleash_the_miri_inside_of_you;
                                if self.tcx.is_const_fn(def_id)
                                    || unleash_miri
                                    || self.suppress_errors
                                {
                                    // stable const fns or unstable const fns
                                    // with their feature gate active
                                    // FIXME(eddyb) move stability checks from `is_const_fn` here.
                                } else if self.is_const_panic_fn(def_id) {
                                    // Check the const_panic feature gate.
                                    // FIXME: cannot allow this inside `allow_internal_unstable`
                                    // because that would make `panic!` insta stable in constants,
                                    // since the macro is marked with the attribute.
                                    if !self.tcx.features().const_panic {
                                        // Don't allow panics in constants without the feature gate.
                                        self.record_error(ops::Panic);
                                        emit_feature_err(
                                            &self.tcx.sess.parse_sess,
                                            sym::const_panic,
                                            self.span,
                                            GateIssue::Language,
                                            &format!("panicking in {}s is unstable", self.mode),
                                        );
                                    }
                                } else if let Some(feature)
                                              = self.tcx.is_unstable_const_fn(def_id) {
                                    // Check `#[unstable]` const fns or `#[rustc_const_unstable]`
                                    // functions without the feature gate active in this crate in
                                    // order to report a better error message than the one below.
                                    if !self.span.allows_unstable(feature) {
                                        self.record_error(ops::FnCallUnstable(def_id, feature));
                                        let mut err = self.tcx.sess.struct_span_err(self.span,
                                            &format!("`{}` is not yet stable as a const fn",
                                                    self.tcx.def_path_str(def_id)));
                                        if nightly_options::is_nightly_build() {
                                            help!(&mut err,
                                                  "add `#![feature({})]` to the \
                                                   crate attributes to enable",
                                                  feature);
                                        }
                                        err.emit();
                                    }
                                } else {
                                    self.record_error(ops::FnCallNonConst(def_id));
                                    let mut err = struct_span_err!(
                                        self.tcx.sess,
                                        self.span,
                                        E0015,
                                        "calls in {}s are limited to constant functions, \
                                         tuple structs and tuple variants",
                                        self.mode,
                                    );
                                    err.emit();
                                }
                            }
                        }
                    }
                }
                ty::FnPtr(_) => {
                    unleash_miri!(self);
                    if self.mode.requires_const_checking() && !self.suppress_errors {
                        self.record_error(ops::FnCallIndirect);
                        let mut err = self.tcx.sess.struct_span_err(
                            self.span,
                            "function pointers are not allowed in const fn"
                        );
                        err.emit();
                    }
                }
                _ => {
                    self.not_const(ops::FnCallOther);
                }
            }

            // No need to do anything in constants and statics, as everything is "constant" anyway
            // so promotion would be useless.
            if self.mode != Mode::Static && self.mode != Mode::Const {
                let constant_args = callee_def_id.and_then(|id| {
                    args_required_const(self.tcx, id)
                }).unwrap_or_default();
                for (i, arg) in args.iter().enumerate() {
                    if !(is_shuffle && i == 2 || constant_args.contains(&i)) {
                        continue;
                    }

                    let candidate = Candidate::Argument { bb: location.block, index: i };
                    // Since the argument is required to be constant,
                    // we care about constness, not promotability.
                    // If we checked for promotability, we'd miss out on
                    // the results of function calls (which are never promoted
                    // in runtime code).
                    // This is not a problem, because the argument explicitly
                    // requests constness, in contrast to regular promotion
                    // which happens even without the user requesting it.
                    //
                    // `promote_consts` is responsible for emitting the error if
                    // the argument is not promotable.
                    if !IsNotPromotable::in_operand(self, arg) {
                        debug!("visit_terminator_kind: candidate={:?}", candidate);
                        self.promotion_candidates.push(candidate);
                    }
                }
            }

            // Check callee and argument operands.
            self.visit_operand(func, location);
            for arg in args {
                self.visit_operand(arg, location);
            }
        } else if let TerminatorKind::Drop {
            location: ref place, ..
        } | TerminatorKind::DropAndReplace {
            location: ref place, ..
        } = *kind {
            match *kind {
                TerminatorKind::DropAndReplace { .. } => {}
                _ => self.super_terminator_kind(kind, location),
            }

            // Deny *any* live drops anywhere other than functions.
            if self.mode.requires_const_checking() && !self.suppress_errors {
                unleash_miri!(self);
                // HACK(eddyb): emulate a bit of dataflow analysis,
                // conservatively, that drop elaboration will do.
                let needs_drop = if let Some(local) = place.as_local() {
                    if NeedsDrop::in_local(self, local) {
                        Some(self.body.local_decls[local].source_info.span)
                    } else {
                        None
                    }
                } else {
                    Some(self.span)
                };

                if let Some(span) = needs_drop {
                    // Double-check the type being dropped, to minimize false positives.
                    let ty = place.ty(self.body, self.tcx).ty;
                    if ty.needs_drop(self.tcx, self.param_env) {
                        self.record_error_spanned(ops::LiveDrop, span);
                        struct_span_err!(self.tcx.sess, span, E0493,
                                         "destructors cannot be evaluated at compile-time")
                            .span_label(span, format!("{}s cannot evaluate destructors",
                                                      self.mode))
                            .emit();
                    }
                }
            }

            match *kind {
                TerminatorKind::DropAndReplace { ref value, .. } => {
                    self.assign(place, ValueSource::DropAndReplace(value), location);
                    self.visit_operand(value, location);
                }
                _ => {}
            }
        } else {
            // Qualify any operands inside other terminators.
            self.super_terminator_kind(kind, location);
        }
    }

    fn visit_assign(&mut self,
                    dest: &Place<'tcx>,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        debug!("visit_assign: dest={:?} rvalue={:?} location={:?}", dest, rvalue, location);
        self.assign(dest, ValueSource::Rvalue(rvalue), location);

        self.visit_rvalue(rvalue, location);
    }

    fn visit_source_info(&mut self, source_info: &SourceInfo) {
        debug!("visit_source_info: source_info={:?}", source_info);
        self.span = source_info.span;
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        debug!("visit_statement: statement={:?} location={:?}", statement, location);
        match statement.kind {
            StatementKind::Assign(..) => {
                self.super_statement(statement, location);
            }
            StatementKind::FakeRead(FakeReadCause::ForMatchedPlace, _) => {
                self.not_const(ops::IfOrMatch);
            }
            // FIXME(eddyb) should these really do nothing?
            StatementKind::FakeRead(..) |
            StatementKind::SetDiscriminant { .. } |
            StatementKind::StorageLive(_) |
            StatementKind::StorageDead(_) |
            StatementKind::InlineAsm {..} |
            StatementKind::Retag { .. } |
            StatementKind::AscribeUserType(..) |
            StatementKind::Nop => {}
        }
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        mir_const_qualif,
        ..*providers
    };
}

// FIXME(eddyb) this is only left around for the validation logic
// in `promote_consts`, see the comment in `validate_operand`.
pub(super) const QUALIF_ERROR_BIT: u8 = 1 << IsNotPromotable::IDX;

fn mir_const_qualif(tcx: TyCtxt<'_>, def_id: DefId) -> (u8, &BitSet<Local>) {
    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_validated()`, which steals
    // from `mir_const(), forces this query to execute before
    // performing the steal.
    let body = &tcx.mir_const(def_id).borrow();

    if body.return_ty().references_error() {
        tcx.sess.delay_span_bug(body.span, "mir_const_qualif: MIR had errors");
        return (QUALIF_ERROR_BIT, tcx.arena.alloc(BitSet::new_empty(0)));
    }

    Checker::new(tcx, def_id, body, Mode::Const).check_const()
}

pub struct QualifyAndPromoteConstants<'tcx> {
    pub promoted: Cell<IndexVec<Promoted, Body<'tcx>>>,
}

impl<'tcx> Default for QualifyAndPromoteConstants<'tcx> {
    fn default() -> Self {
        QualifyAndPromoteConstants {
            promoted: Cell::new(IndexVec::new()),
        }
    }
}

impl<'tcx> MirPass<'tcx> for QualifyAndPromoteConstants<'tcx> {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        // There's not really any point in promoting errorful MIR.
        if body.return_ty().references_error() {
            tcx.sess.delay_span_bug(body.span, "QualifyAndPromoteConstants: MIR had errors");
            return;
        }

        if src.promoted.is_some() {
            return;
        }

        let def_id = src.def_id();
        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

        let mode = determine_mode(tcx, hir_id, def_id);

        debug!("run_pass: mode={:?}", mode);
        if let Mode::NonConstFn | Mode::ConstFn = mode {
            // This is ugly because Checker holds onto mir,
            // which can't be mutated until its scope ends.
            let (temps, candidates) = {
                let mut checker = Checker::new(tcx, def_id, body, mode);
                if let Mode::ConstFn = mode {
                    let use_min_const_fn_checks =
                        !tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you &&
                        tcx.is_min_const_fn(def_id);
                    if use_min_const_fn_checks {
                        // Enforce `min_const_fn` for stable `const fn`s.
                        use super::qualify_min_const_fn::is_min_const_fn;
                        if let Err((span, err)) = is_min_const_fn(tcx, def_id, body) {
                            error_min_const_fn_violation(tcx, span, err);
                            return;
                        }

                        // `check_const` should not produce any errors, but better safe than sorry
                        // FIXME(#53819)
                        // NOTE(eddyb) `check_const` is actually needed for promotion inside
                        // `min_const_fn` functions.
                    }

                    // Enforce a constant-like CFG for `const fn`.
                    checker.check_const();
                } else {
                    while let Some((bb, data)) = checker.rpo.next() {
                        checker.visit_basic_block_data(bb, data);
                    }
                }

                let promotion_candidates = checker.valid_promotion_candidates();
                (checker.temp_promotion_state, promotion_candidates)
            };

            // Do the actual promotion, now that we know what's viable.
            self.promoted.set(
                promote_consts::promote_candidates(def_id, body, tcx, temps, candidates)
            );
        } else {
            check_short_circuiting_in_const_local(tcx, body, mode);

            let promoted_temps = match mode {
                Mode::Const => tcx.mir_const_qualif(def_id).1,
                _ => Checker::new(tcx, def_id, body, mode).check_const().1,
            };
            remove_drop_and_storage_dead_on_promoted_locals(body, promoted_temps);
        }

        if mode == Mode::Static && !tcx.has_attr(def_id, sym::thread_local) {
            // `static`s (not `static mut`s) which are not `#[thread_local]` must be `Sync`.
            check_static_is_sync(tcx, body, hir_id);
        }
    }
}

fn determine_mode(tcx: TyCtxt<'_>, hir_id: HirId, def_id: DefId) -> Mode {
    match tcx.hir().body_owner_kind(hir_id) {
        hir::BodyOwnerKind::Closure => Mode::NonConstFn,
        hir::BodyOwnerKind::Fn if tcx.is_const_fn(def_id) => Mode::ConstFn,
        hir::BodyOwnerKind::Fn => Mode::NonConstFn,
        hir::BodyOwnerKind::Const => Mode::Const,
        hir::BodyOwnerKind::Static(hir::MutImmutable) => Mode::Static,
        hir::BodyOwnerKind::Static(hir::MutMutable) => Mode::StaticMut,
    }
}

fn error_min_const_fn_violation(tcx: TyCtxt<'_>, span: Span, msg: Cow<'_, str>) {
    struct_span_err!(tcx.sess, span, E0723, "{}", msg)
        .note("for more information, see issue https://github.com/rust-lang/rust/issues/57563")
        .help("add `#![feature(const_fn)]` to the crate attributes to enable")
        .emit();
}

fn check_short_circuiting_in_const_local(tcx: TyCtxt<'_>, body: &mut Body<'tcx>, mode: Mode) {
    if body.control_flow_destroyed.is_empty() {
        return;
    }

    let mut locals = body.vars_iter();
    if let Some(local) = locals.next() {
        let span = body.local_decls[local].source_info.span;
        let mut error = tcx.sess.struct_span_err(
            span,
            &format!(
                "new features like let bindings are not permitted in {}s \
                which also use short circuiting operators",
                mode,
            ),
        );
        for (span, kind) in body.control_flow_destroyed.iter() {
            error.span_note(
                *span,
                &format!("use of {} here does not actually short circuit due to \
                the const evaluator presently not being able to do control flow. \
                See https://github.com/rust-lang/rust/issues/49146 for more \
                information.", kind),
            );
        }
        for local in locals {
            let span = body.local_decls[local].source_info.span;
            error.span_note(span, "more locals defined here");
        }
        error.emit();
    }
}

/// In `const` and `static` everything without `StorageDead`
/// is `'static`, we don't have to create promoted MIR fragments,
/// just remove `Drop` and `StorageDead` on "promoted" locals.
fn remove_drop_and_storage_dead_on_promoted_locals(
    body: &mut Body<'tcx>,
    promoted_temps: &BitSet<Local>,
) {
    debug!("run_pass: promoted_temps={:?}", promoted_temps);

    for block in body.basic_blocks_mut() {
        block.statements.retain(|statement| {
            match statement.kind {
                StatementKind::StorageDead(index) => !promoted_temps.contains(index),
                _ => true
            }
        });
        let terminator = block.terminator_mut();
        match &terminator.kind {
            TerminatorKind::Drop {
                location,
                target,
                ..
            } => {
                if let Some(index) = location.as_local() {
                    if promoted_temps.contains(index) {
                        terminator.kind = TerminatorKind::Goto { target: *target };
                    }
                }
            }
            _ => {}
        }
    }
}

fn check_static_is_sync(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>, hir_id: HirId) {
    let ty = body.return_ty();
    tcx.infer_ctxt().enter(|infcx| {
        let cause = traits::ObligationCause::new(body.span, hir_id, traits::SharedStatic);
        let mut fulfillment_cx = traits::FulfillmentContext::new();
        let sync_def_id = tcx.require_lang_item(lang_items::SyncTraitLangItem, Some(body.span));
        fulfillment_cx.register_bound(&infcx, ty::ParamEnv::empty(), ty, sync_def_id, cause);
        if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(&err, None, false);
        }
    });
}

fn args_required_const(tcx: TyCtxt<'_>, def_id: DefId) -> Option<FxHashSet<usize>> {
    let attrs = tcx.get_attrs(def_id);
    let attr = attrs.iter().find(|a| a.check_name(sym::rustc_args_required_const))?;
    let mut ret = FxHashSet::default();
    for meta in attr.meta_item_list()? {
        match meta.literal()?.kind {
            LitKind::Int(a, _) => { ret.insert(a as usize); }
            _ => return None,
        }
    }
    Some(ret)
}

fn validator_mismatch(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    mut old_errors: Vec<(Span, String)>,
    mut new_errors: Vec<(Span, String)>,
) {
    error!("old validator: {:?}", old_errors);
    error!("new validator: {:?}", new_errors);

    // ICE on nightly if the validators do not emit exactly the same errors.
    // Users can supress this panic with an unstable compiler flag (hopefully after
    // filing an issue).
    let opts = &tcx.sess.opts;
    let strict_validation_enabled = opts.unstable_features.is_nightly_build()
        && !opts.debugging_opts.suppress_const_validation_back_compat_ice;

    if !strict_validation_enabled {
        return;
    }

    // If this difference would cause a regression from the old to the new or vice versa, trigger
    // the ICE.
    if old_errors.is_empty() || new_errors.is_empty() {
        span_bug!(body.span, "{}", VALIDATOR_MISMATCH_ERR);
    }

    // HACK: Borrows that would allow mutation are forbidden in const contexts, but they cause the
    // new validator to be more conservative about when a dropped local has been moved out of.
    //
    // Supress the mismatch ICE in cases where the validators disagree only on the number of
    // `LiveDrop` errors and both observe the same sequence of `MutBorrow`s.

    let is_live_drop = |(_, s): &mut (_, String)| s.starts_with("LiveDrop");
    let is_mut_borrow = |(_, s): &&(_, String)| s.starts_with("MutBorrow");

    let old_live_drops: Vec<_> = old_errors.drain_filter(is_live_drop).collect();
    let new_live_drops: Vec<_> = new_errors.drain_filter(is_live_drop).collect();

    let only_live_drops_differ = old_live_drops != new_live_drops && old_errors == new_errors;

    let old_mut_borrows = old_errors.iter().filter(is_mut_borrow);
    let new_mut_borrows = new_errors.iter().filter(is_mut_borrow);

    let at_least_one_mut_borrow = old_mut_borrows.clone().next().is_some();

    if only_live_drops_differ && at_least_one_mut_borrow && old_mut_borrows.eq(new_mut_borrows) {
        return;
    }

    span_bug!(body.span, "{}", VALIDATOR_MISMATCH_ERR);
}

const VALIDATOR_MISMATCH_ERR: &str =
    r"Disagreement between legacy and dataflow-based const validators.
    After filing an issue, use `-Zsuppress-const-validation-back-compat-ice` to compile your code.";
