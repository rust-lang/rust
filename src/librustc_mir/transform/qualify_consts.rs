//! A pass that qualifies constness of temporaries in constants,
//! static initializers and functions and also drives promotion.
//!
//! The Qualif flags below can be used to also provide better
//! diagnostics as to why a constant rvalue wasn't promoted.

use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_target::spec::abi::Abi;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::traits::{self, TraitEngine};
use rustc::ty::{self, TyCtxt, Ty, TypeFoldable};
use rustc::ty::cast::CastTy;
use rustc::ty::query::Providers;
use rustc::mir::*;
use rustc::mir::traversal::ReversePostorder;
use rustc::mir::visit::{PlaceContext, Visitor, MutatingUseContext, NonMutatingUseContext};
use rustc::middle::lang_items;
use rustc::session::config::nightly_options;
use syntax::ast::LitKind;
use syntax::feature_gate::{emit_feature_err, GateIssue};
use syntax_pos::{Span, DUMMY_SP};

use std::fmt;
use std::ops::{Deref, Index, IndexMut};
use std::usize;

use crate::transform::{MirPass, MirSource};
use super::promote_consts::{self, Candidate, TempState};

/// What kind of item we are in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Const,
    Static,
    StaticMut,
    ConstFn,
    Fn
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Mode::Const => write!(f, "constant"),
            Mode::Static | Mode::StaticMut => write!(f, "static"),
            Mode::ConstFn => write!(f, "constant function"),
            Mode::Fn => write!(f, "function")
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
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mode: Mode,
    mir: &'a Mir<'tcx>,

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
    Call {
        callee: &'a Operand<'tcx>,
        args: &'a [Operand<'tcx>],
        return_ty: Ty<'tcx>,
    },
}

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
        proj: &PlaceProjection<'tcx>,
    ) -> bool {
        let base_qualif = Self::in_place(cx, &proj.base);
        let qualif = base_qualif && Self::mask_for_ty(
            cx,
            proj.base.ty(cx.mir, cx.tcx)
                .projection_ty(cx.tcx, &proj.elem)
                .to_ty(cx.tcx),
        );
        match proj.elem {
            ProjectionElem::Deref |
            ProjectionElem::Subslice { .. } |
            ProjectionElem::Field(..) |
            ProjectionElem::ConstantIndex { .. } |
            ProjectionElem::Downcast(..) => qualif,

            ProjectionElem::Index(local) => qualif || Self::in_local(cx, local),
        }
    }

    fn in_projection(cx: &ConstCx<'_, 'tcx>, proj: &PlaceProjection<'tcx>) -> bool {
        Self::in_projection_structurally(cx, proj)
    }

    fn in_place(cx: &ConstCx<'_, 'tcx>, place: &Place<'tcx>) -> bool {
        match *place {
            Place::Local(local) => Self::in_local(cx, local),
            Place::Promoted(_) => bug!("qualifying already promoted MIR"),
            Place::Static(ref static_) => Self::in_static(cx, static_),
            Place::Projection(ref proj) => Self::in_projection(cx, proj),
        }
    }

    fn in_operand(cx: &ConstCx<'_, 'tcx>, operand: &Operand<'tcx>) -> bool {
        match *operand {
            Operand::Copy(ref place) |
            Operand::Move(ref place) => Self::in_place(cx, place),

            Operand::Constant(ref constant) => {
                if let ty::LazyConst::Unevaluated(def_id, _) = constant.literal {
                    // Don't peek inside trait associated constants.
                    if cx.tcx.trait_of_item(*def_id).is_some() {
                        Self::in_any_value_of_ty(cx, constant.ty).unwrap_or(false)
                    } else {
                        let (bits, _) = cx.tcx.at(constant.span).mir_const_qualif(*def_id);

                        let qualif = PerQualif::decode_from_bits(bits).0[Self::IDX];

                        // Just in case the type is more specific than
                        // the definition, e.g., impl associated const
                        // with type parameters, take it into account.
                        qualif && Self::mask_for_ty(cx, constant.ty)
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
            Rvalue::Len(ref place) => Self::in_place(cx, place),

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
                if let Place::Projection(ref proj) = *place {
                    if let ProjectionElem::Deref = proj.elem {
                        let base_ty = proj.base.ty(cx.mir, cx.tcx).to_ty(cx.tcx);
                        if let ty::Ref(..) = base_ty.sty {
                            return Self::in_place(cx, &proj.base);
                        }
                    }
                }

                Self::in_place(cx, place)
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
            ValueSource::Call { callee, args, return_ty } => {
                Self::in_call(cx, callee, args, return_ty)
            }
        }
    }
}

// Constant containing interior mutability (UnsafeCell).
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
                let ty = place.ty(cx.mir, cx.tcx).to_ty(cx.tcx);

                if let BorrowKind::Mut { .. } = kind {
                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences. However, only &mut []
                    // is allowed right now, and only in functions.
                    if cx.mode == Mode::StaticMut {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        match ty.sty {
                            ty::Array(..) | ty::Slice(_) => {}
                            _ => return true,
                        }
                    } else if let ty::Array(_, len) = ty.sty {
                        // FIXME(eddyb) the `cx.mode == Mode::Fn` condition
                        // seems unnecessary, given that this is merely a ZST.
                        if !(len.unwrap_usize(cx.tcx) == 0 && cx.mode == Mode::Fn) {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }

            Rvalue::Aggregate(ref kind, _) => {
                if let AggregateKind::Adt(def, ..) = **kind {
                    if Some(def.did) == cx.tcx.lang_items().unsafe_cell_type() {
                        let ty = rvalue.ty(cx.mir, cx.tcx);
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

// Constant containing an ADT that implements Drop.
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

// Not constant at all - non-`const fn` calls, asm!,
// pointer comparisons, ptr-to-int casts, etc.
struct IsNotConst;

impl Qualif for IsNotConst {
    const IDX: usize = 2;

    fn in_static(cx: &ConstCx<'_, 'tcx>, static_: &Static<'tcx>) -> bool {
        // Only allow statics (not consts) to refer to other statics.
        let allowed = cx.mode == Mode::Static || cx.mode == Mode::StaticMut;

        !allowed ||
            cx.tcx.get_attrs(static_.def_id).iter().any(|attr| attr.check_name("thread_local"))
    }

    fn in_projection(cx: &ConstCx<'_, 'tcx>, proj: &PlaceProjection<'tcx>) -> bool {
        match proj.elem {
            ProjectionElem::Deref |
            ProjectionElem::Downcast(..) => return true,

            ProjectionElem::ConstantIndex {..} |
            ProjectionElem::Subslice {..} |
            ProjectionElem::Index(_) => {}

            ProjectionElem::Field(..) => {
                if cx.mode == Mode::Fn {
                    let base_ty = proj.base.ty(cx.mir, cx.tcx).to_ty(cx.tcx);
                    if let Some(def) = base_ty.ty_adt_def() {
                        if def.is_union() {
                            return true;
                        }
                    }
                }
            }
        }

        Self::in_projection_structurally(cx, proj)
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> bool {
        match *rvalue {
            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) if cx.mode == Mode::Fn => {
                let operand_ty = operand.ty(cx.mir, cx.tcx);
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

            Rvalue::BinaryOp(op, ref lhs, _) if cx.mode == Mode::Fn => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(cx.mir, cx.tcx).sty {
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
        let fn_ty = callee.ty(cx.mir, cx.tcx);
        match fn_ty.sty {
            ty::FnDef(def_id, _) => {
                match cx.tcx.fn_sig(def_id).abi() {
                    Abi::RustIntrinsic |
                    Abi::PlatformIntrinsic => {
                        assert!(!cx.tcx.is_const_fn(def_id));
                        match &cx.tcx.item_name(def_id).as_str()[..] {
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
                            | "overflowing_add"
                            | "overflowing_sub"
                            | "overflowing_mul"
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

// Refers to temporaries which cannot be promoted as
// promote_consts decided they weren't simple enough.
struct IsNotPromotable;

impl Qualif for IsNotPromotable {
    const IDX: usize = 3;

    fn in_call(
        cx: &ConstCx<'_, 'tcx>,
        callee: &Operand<'tcx>,
        _args: &[Operand<'tcx>],
        _return_ty: Ty<'tcx>,
    ) -> bool {
        if cx.mode == Mode::Fn {
            if let ty::FnDef(def_id, _) = callee.ty(cx.mir, cx.tcx).sty {
                // Never promote runtime `const fn` calls of
                // functions without `#[rustc_promotable]`.
                if !cx.tcx.is_promotable_const_fn(def_id) {
                    return true;
                }
            }
        }

        // FIXME(eddyb) do we need "not promotable" in anything
        // other than `Mode::Fn` by any chance?

        false
    }
}

// Ensure the `IDX` values are sequential (`0..QUALIF_COUNT`).
macro_rules! static_assert_seq_qualifs {
    ($i:expr => $first:ident $(, $rest:ident)*) => {
        static_assert!(SEQ_QUALIFS: {
            static_assert_seq_qualifs!($i + 1 => $($rest),*);

            $first::IDX == $i
        });
    };
    ($i:expr =>) => {
        static_assert!(SEQ_QUALIFS: QUALIF_COUNT == $i);
    };
}
static_assert_seq_qualifs!(0 => HasMutInterior, NeedsDrop, IsNotConst, IsNotPromotable);

impl ConstCx<'_, 'tcx> {
    fn qualifs_in_any_value_of_ty(&self, ty: Ty<'tcx>) -> PerQualif<bool> {
        let mut qualifs = PerQualif::default();
        qualifs[HasMutInterior] = HasMutInterior::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs[NeedsDrop] = NeedsDrop::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs[IsNotConst] = IsNotConst::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs[IsNotPromotable] = IsNotPromotable::in_any_value_of_ty(self, ty).unwrap_or(false);
        qualifs
    }

    fn qualifs_in_local(&self, local: Local) -> PerQualif<bool> {
        let mut qualifs = PerQualif::default();
        qualifs[HasMutInterior] = HasMutInterior::in_local(self, local);
        qualifs[NeedsDrop] = NeedsDrop::in_local(self, local);
        qualifs[IsNotConst] = IsNotConst::in_local(self, local);
        qualifs[IsNotPromotable] = IsNotPromotable::in_local(self, local);
        qualifs
    }

    fn qualifs_in_value(&self, source: ValueSource<'_, 'tcx>) -> PerQualif<bool> {
        let mut qualifs = PerQualif::default();
        qualifs[HasMutInterior] = HasMutInterior::in_value(self, source);
        qualifs[NeedsDrop] = NeedsDrop::in_value(self, source);
        qualifs[IsNotConst] = IsNotConst::in_value(self, source);
        qualifs[IsNotPromotable] = IsNotPromotable::in_value(self, source);
        qualifs
    }
}

struct Checker<'a, 'tcx> {
    cx: ConstCx<'a, 'tcx>,

    span: Span,
    def_id: DefId,
    rpo: ReversePostorder<'a, 'tcx>,

    temp_promotion_state: IndexVec<Local, TempState>,
    promotion_candidates: Vec<Candidate>,
}

macro_rules! unleash_miri {
    ($this:expr) => {{
        if $this.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
            $this.tcx.sess.span_warn($this.span, "skipping const checks");
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
    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
           def_id: DefId,
           mir: &'a Mir<'tcx>,
           mode: Mode)
           -> Self {
        assert!(def_id.is_local());
        let mut rpo = traversal::reverse_postorder(mir);
        let temps = promote_consts::collect_temps(mir, &mut rpo);
        rpo.reset();

        let param_env = tcx.param_env(def_id);

        let mut cx = ConstCx {
            tcx,
            param_env,
            mode,
            mir,
            per_local: PerQualif::new(BitSet::new_empty(mir.local_decls.len())),
        };

        for (local, decl) in mir.local_decls.iter_enumerated() {
            match mir.local_kind(local) {
                LocalKind::Arg => {
                    let qualifs = cx.qualifs_in_any_value_of_ty(decl.ty);
                    for (per_local, qualif) in &mut cx.per_local.as_mut().zip(qualifs).0 {
                        if *qualif {
                            per_local.insert(local);
                        }
                    }
                    cx.per_local[IsNotPromotable].insert(local);
                }

                LocalKind::Var if mode == Mode::Fn => {
                    cx.per_local[IsNotConst].insert(local);
                }

                LocalKind::Temp if !temps[local].is_promotable() => {
                    cx.per_local[IsNotPromotable].insert(local);
                }

                _ => {}
            }
        }

        Checker {
            cx,
            span: mir.span,
            def_id,
            rpo,
            temp_promotion_state: temps,
            promotion_candidates: vec![]
        }
    }

    // FIXME(eddyb) we could split the errors into meaningful
    // categories, but enabling full miri would make that
    // slightly pointless (even with feature-gating).
    fn not_const(&mut self) {
        unleash_miri!(self);
        if self.mode != Mode::Fn {
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

    /// Assigns an rvalue/call qualification to the given destination.
    fn assign(&mut self, dest: &Place<'tcx>, source: ValueSource<'_, 'tcx>, location: Location) {
        trace!("assign: {:?} <- {:?}", dest, source);

        let mut qualifs = self.qualifs_in_value(source);

        if let ValueSource::Rvalue(&Rvalue::Ref(_, kind, ref place)) = source {
            // Getting `true` from `HasMutInterior::in_rvalue` means
            // the borrowed place is disallowed from being borrowed,
            // due to either a mutable borrow (with some exceptions),
            // or an shared borrow of a value with interior mutability.
            // Then `HasMutInterior` is replaced with `IsNotConst`,
            // to avoid duplicate errors (e.g. from reborrowing).
            if qualifs[HasMutInterior] {
                qualifs[HasMutInterior] = false;
                qualifs[IsNotConst] = true;

                if self.mode != Mode::Fn {
                    if let BorrowKind::Mut { .. } = kind {
                        let mut err = struct_span_err!(self.tcx.sess,  self.span, E0017,
                                                       "references in {}s may only refer \
                                                        to immutable values", self.mode);
                        err.span_label(self.span, format!("{}s require immutable values",
                                                            self.mode));
                        if self.tcx.sess.teach(&err.get_code().unwrap()) {
                            err.note("References in statics and constants may only refer to \
                                      immutable values.\n\n\
                                      Statics are shared everywhere, and if they refer to \
                                      mutable data one might violate memory safety since \
                                      holding multiple mutable references to shared data is \
                                      not allowed.\n\n\
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
            } else {
                // We might have a candidate for promotion.
                let candidate = Candidate::Ref(location);
                // We can only promote interior borrows of promotable temps.
                let mut place = place;
                while let Place::Projection(ref proj) = *place {
                    if proj.elem == ProjectionElem::Deref {
                        break;
                    }
                    place = &proj.base;
                }
                debug!("qualify_consts: promotion candidate: place={:?}", place);
                if let Place::Local(local) = *place {
                    if self.mir.local_kind(local) == LocalKind::Temp {
                        debug!("qualify_consts: promotion candidate: local={:?}", local);
                        // The borrowed place doesn't have `HasMutInterior`
                        // (from `in_rvalue`), so we can safely ignore
                        // `HasMutInterior` from the local's qualifications.
                        // This allows borrowing fields which don't have
                        // `HasMutInterior`, from a type that does, e.g.:
                        // `let _: &'static _ = &(Cell::new(1), 2).1;`
                        let mut local_qualifs = self.qualifs_in_local(local);
                        local_qualifs[HasMutInterior] = false;
                        if !local_qualifs.0.iter().any(|&qualif| qualif) {
                            debug!("qualify_consts: promotion candidate: {:?}", candidate);
                            self.promotion_candidates.push(candidate);
                        }
                    }
                }
            }
        }

        let mut dest = dest;
        let index = loop {
            match dest {
                // We treat all locals equal in constants
                Place::Local(index) => break *index,
                // projections are transparent for assignments
                // we qualify the entire destination at once, even if just a field would have
                // stricter qualification
                Place::Projection(proj) => {
                    // Catch more errors in the destination. `visit_place` also checks various
                    // projection rules like union field access and raw pointer deref
                    self.visit_place(
                        dest,
                        PlaceContext::MutatingUse(MutatingUseContext::Store),
                        location
                    );
                    dest = &proj.base;
                },
                Place::Promoted(..) => bug!("promoteds don't exist yet during promotion"),
                Place::Static(..) => {
                    // Catch more errors in the destination. `visit_place` also checks that we
                    // do not try to access statics from constants or try to mutate statics
                    self.visit_place(
                        dest,
                        PlaceContext::MutatingUse(MutatingUseContext::Store),
                        location
                    );
                    return;
                }
            }
        };

        let kind = self.mir.local_kind(index);
        debug!("store to {:?} {:?}", kind, index);

        // Only handle promotable temps in non-const functions.
        if self.mode == Mode::Fn {
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
    fn check_const(&mut self) -> (u8, Lrc<BitSet<Local>>) {
        debug!("const-checking {} {:?}", self.mode, self.def_id);

        let mir = self.mir;

        let mut seen_blocks = BitSet::new_empty(mir.basic_blocks().len());
        let mut bb = START_BLOCK;
        loop {
            seen_blocks.insert(bb.index());

            self.visit_basic_block_data(bb, &mir[bb]);

            let target = match mir[bb].terminator().kind {
                TerminatorKind::Goto { target } |
                TerminatorKind::Drop { target, .. } |
                TerminatorKind::Assert { target, .. } |
                TerminatorKind::Call { destination: Some((_, target)), .. } => {
                    Some(target)
                }

                // Non-terminating calls cannot produce any value.
                TerminatorKind::Call { destination: None, .. } => {
                    break;
                }

                TerminatorKind::SwitchInt {..} |
                TerminatorKind::DropAndReplace { .. } |
                TerminatorKind::Resume |
                TerminatorKind::Abort |
                TerminatorKind::GeneratorDrop |
                TerminatorKind::Yield { .. } |
                TerminatorKind::Unreachable |
                TerminatorKind::FalseEdges { .. } |
                TerminatorKind::FalseUnwind { .. } => None,

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
                    self.not_const();
                    break;
                }
            }
        }


        // Collect all the temps we need to promote.
        let mut promoted_temps = BitSet::new_empty(self.temp_promotion_state.len());

        debug!("qualify_const: promotion_candidates={:?}", self.promotion_candidates);
        for candidate in &self.promotion_candidates {
            match *candidate {
                Candidate::Ref(Location { block: bb, statement_index: stmt_idx }) => {
                    match self.mir[bb].statements[stmt_idx].kind {
                        StatementKind::Assign(_, box Rvalue::Ref(_, _, Place::Local(index))) => {
                            promoted_temps.insert(index);
                        }
                        _ => {}
                    }
                }
                Candidate::Argument { .. } => {}
            }
        }

        let promoted_temps = Lrc::new(promoted_temps);

        let mut qualifs = self.qualifs_in_local(RETURN_PLACE);

        // Account for errors in consts by using the
        // conservative type qualification instead.
        if qualifs[IsNotConst] {
            qualifs = self.qualifs_in_any_value_of_ty(mir.return_ty());
        }

        (qualifs.encode_to_bits(), promoted_temps)
    }
}

/// Checks MIR for const-correctness, using `ConstCx`
/// for value qualifications, and accumulates writes of
/// rvalue/call results to locals, in `local_qualif`.
/// For functions (constant or not), it also records
/// candidates for promotion in `promotion_candidates`.
impl<'a, 'tcx> Visitor<'tcx> for Checker<'a, 'tcx> {
    fn visit_place(&mut self,
                    place: &Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        debug!("visit_place: place={:?} context={:?} location={:?}", place, context, location);
        self.super_place(place, context, location);
        match *place {
            Place::Local(_) |
            Place::Promoted(_) => {}
            Place::Static(ref global) => {
                if self.tcx
                       .get_attrs(global.def_id)
                       .iter()
                       .any(|attr| attr.check_name("thread_local")) {
                    if self.mode != Mode::Fn {
                        span_err!(self.tcx.sess, self.span, E0625,
                                  "thread-local statics cannot be \
                                   accessed at compile-time");
                    }
                    return;
                }

                // Only allow statics (not consts) to refer to other statics.
                if self.mode == Mode::Static || self.mode == Mode::StaticMut {
                    if self.mode == Mode::Static && context.is_mutating_use() {
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

                if self.mode != Mode::Fn {
                    let mut err = struct_span_err!(self.tcx.sess, self.span, E0013,
                                                   "{}s cannot refer to statics, use \
                                                    a constant instead", self.mode);
                    if self.tcx.sess.teach(&err.get_code().unwrap()) {
                        err.note(
                            "Static and const variables can refer to other const variables. But a \
                             const variable cannot refer to a static variable."
                        );
                        err.help(
                            "To fix this, the value can be extracted as a const and then used."
                        );
                    }
                    err.emit()
                }
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        if context.is_mutating_use() {
                            // `not_const` errors out in const contexts
                            self.not_const()
                        }
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        match self.mode {
                            Mode::Fn => {},
                            _ => {
                                if let ty::RawPtr(_) = base_ty.sty {
                                    if !self.tcx.features().const_raw_ptr_deref {
                                        emit_feature_err(
                                            &self.tcx.sess.parse_sess, "const_raw_ptr_deref",
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
                        let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                        if let Some(def) = base_ty.ty_adt_def() {
                            if def.is_union() {
                                match self.mode {
                                    Mode::ConstFn => {
                                        if !self.tcx.features().const_fn_union {
                                            emit_feature_err(
                                                &self.tcx.sess.parse_sess, "const_fn_union",
                                                self.span, GateIssue::Language,
                                                "unions in const fn are unstable",
                                            );
                                        }
                                    },

                                    | Mode::Fn
                                    | Mode::Static
                                    | Mode::StaticMut
                                    | Mode::Const
                                    => {},
                                }
                            }
                        }
                    }

                    ProjectionElem::Downcast(..) => {
                        self.not_const()
                    }
                }
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        debug!("visit_operand: operand={:?} location={:?}", operand, location);
        self.super_operand(operand, location);

        match *operand {
            Operand::Move(ref place) => {
                // Mark the consumed locals to indicate later drops are noops.
                if let Place::Local(local) = *place {
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
        if let Rvalue::Ref(region, kind, ref place) = *rvalue {
            // Special-case reborrows.
            let mut is_reborrow = false;
            if let Place::Projection(ref proj) = *place {
                if let ProjectionElem::Deref = proj.elem {
                    let base_ty = proj.base.ty(self.mir, self.tcx).to_ty(self.tcx);
                    if let ty::Ref(..) = base_ty.sty {
                        is_reborrow = true;
                    }
                }
            }

            if is_reborrow {
                let ctx = match kind {
                    BorrowKind::Shared => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::SharedBorrow(region),
                    ),
                    BorrowKind::Shallow => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::ShallowBorrow(region),
                    ),
                    BorrowKind::Unique => PlaceContext::NonMutatingUse(
                        NonMutatingUseContext::UniqueBorrow(region),
                    ),
                    BorrowKind::Mut { .. } => PlaceContext::MutatingUse(
                        MutatingUseContext::Borrow(region),
                    ),
                };
                self.super_place(place, ctx, location);
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
            Rvalue::Cast(CastKind::ReifyFnPointer, ..) |
            Rvalue::Cast(CastKind::UnsafeFnPointer, ..) |
            Rvalue::Cast(CastKind::ClosureFnPointer, ..) |
            Rvalue::Cast(CastKind::Unsize, ..) |
            Rvalue::Discriminant(..) |
            Rvalue::Len(_) |
            Rvalue::Ref(..) |
            Rvalue::Aggregate(..) => {}

            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.mir, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) if self.mode != Mode::Fn => {
                        unleash_miri!(self);
                        if !self.tcx.features().const_raw_ptr_to_usize_cast {
                            // in const fn and constants require the feature gate
                            // FIXME: make it unsafe inside const fn and constants
                            emit_feature_err(
                                &self.tcx.sess.parse_sess, "const_raw_ptr_to_usize_cast",
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
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.mir, self.tcx).sty {
                    assert!(op == BinOp::Eq || op == BinOp::Ne ||
                            op == BinOp::Le || op == BinOp::Lt ||
                            op == BinOp::Ge || op == BinOp::Gt ||
                            op == BinOp::Offset);

                    unleash_miri!(self);
                    if self.mode != Mode::Fn && !self.tcx.features().const_compare_raw_pointers {
                        // require the feature gate inside constants and const fn
                        // FIXME: make it unsafe to use these operations
                        emit_feature_err(
                            &self.tcx.sess.parse_sess,
                            "const_compare_raw_pointers",
                            self.span,
                            GateIssue::Language,
                            &format!("comparing raw pointers inside {}", self.mode),
                        );
                    }
                }
            }

            Rvalue::NullaryOp(NullOp::Box, _) => {
                unleash_miri!(self);
                if self.mode != Mode::Fn {
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
                             bb: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        debug!("visit_terminator_kind: bb={:?} kind={:?} location={:?}", bb, kind, location);
        if let TerminatorKind::Call { ref func, ref args, ref destination, .. } = *kind {
            if let Some((ref dest, _)) = *destination {
                self.assign(dest, ValueSource::Call {
                    callee: func,
                    args,
                    return_ty: dest.ty(self.mir, self.tcx).to_ty(self.tcx),
                }, location);
            }

            let fn_ty = func.ty(self.mir, self.tcx);
            let mut callee_def_id = None;
            let mut is_shuffle = false;
            match fn_ty.sty {
                ty::FnDef(def_id, _) => {
                    callee_def_id = Some(def_id);
                    match self.tcx.fn_sig(def_id).abi() {
                        Abi::RustIntrinsic |
                        Abi::PlatformIntrinsic => {
                            assert!(!self.tcx.is_const_fn(def_id));
                            match &self.tcx.item_name(def_id).as_str()[..] {
                                // special intrinsic that can be called diretly without an intrinsic
                                // feature gate needs a language feature gate
                                "transmute" => {
                                    // never promote transmute calls
                                    if self.mode != Mode::Fn {
                                        // const eval transmute calls only with the feature gate
                                        if !self.tcx.features().const_transmute {
                                            emit_feature_err(
                                                &self.tcx.sess.parse_sess, "const_transmute",
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
                            if self.mode != Mode::Fn {
                                let unleash_miri = self
                                    .tcx
                                    .sess
                                    .opts
                                    .debugging_opts
                                    .unleash_the_miri_inside_of_you;
                                if self.tcx.is_const_fn(def_id) || unleash_miri {
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
                                        emit_feature_err(
                                            &self.tcx.sess.parse_sess,
                                            "const_panic",
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
                                    if !self.span.allows_unstable(&feature.as_str()) {
                                        let mut err = self.tcx.sess.struct_span_err(self.span,
                                            &format!("`{}` is not yet stable as a const fn",
                                                    self.tcx.item_path_str(def_id)));
                                        if nightly_options::is_nightly_build() {
                                            help!(&mut err,
                                                  "add `#![feature({})]` to the \
                                                   crate attributes to enable",
                                                  feature);
                                        }
                                        err.emit();
                                    }
                                } else {
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
                    if self.mode != Mode::Fn {
                        let mut err = self.tcx.sess.struct_span_err(
                            self.span,
                            &format!("function pointers are not allowed in const fn"));
                        err.emit();
                    }
                }
                _ => {
                    self.not_const();
                }
            }

            if self.mode == Mode::Fn {
                let constant_args = callee_def_id.and_then(|id| {
                    args_required_const(self.tcx, id)
                }).unwrap_or_default();
                for (i, arg) in args.iter().enumerate() {
                    if !(is_shuffle && i == 2 || constant_args.contains(&i)) {
                        continue;
                    }

                    let candidate = Candidate::Argument { bb, index: i };
                    // Since the argument is required to be constant,
                    // we care about constness, not promotability.
                    // If we checked for promotability, we'd miss out on
                    // the results of function calls (which are never promoted
                    // in runtime code).
                    // This is not a problem, because the argument explicitly
                    // requests constness, in contrast to regular promotion
                    // which happens even without the user requesting it.
                    // We can error out with a hard error if the argument is not
                    // constant here.
                    if !IsNotConst::in_operand(self, arg) {
                        debug!("visit_terminator_kind: candidate={:?}", candidate);
                        self.promotion_candidates.push(candidate);
                    } else {
                        if is_shuffle {
                            span_err!(self.tcx.sess, self.span, E0526,
                                      "shuffle indices are not constant");
                        } else {
                            self.tcx.sess.span_err(self.span,
                                &format!("argument {} is required to be a constant",
                                         i + 1));
                        }
                    }
                }
            }

            // Check callee and argument operands.
            self.visit_operand(func, location);
            for arg in args {
                self.visit_operand(arg, location);
            }
        } else if let TerminatorKind::Drop { location: ref place, .. } = *kind {
            self.super_terminator_kind(bb, kind, location);

            // Deny *any* live drops anywhere other than functions.
            if self.mode != Mode::Fn {
                unleash_miri!(self);
                // HACK(eddyb): emulate a bit of dataflow analysis,
                // conservatively, that drop elaboration will do.
                let needs_drop = if let Place::Local(local) = *place {
                    if NeedsDrop::in_local(self, local) {
                        Some(self.mir.local_decls[local].source_info.span)
                    } else {
                        None
                    }
                } else {
                    Some(self.span)
                };

                if let Some(span) = needs_drop {
                    // Double-check the type being dropped, to minimize false positives.
                    let ty = place.ty(self.mir, self.tcx).to_ty(self.tcx);
                    if ty.needs_drop(self.tcx, self.param_env) {
                        struct_span_err!(self.tcx.sess, span, E0493,
                                         "destructors cannot be evaluated at compile-time")
                            .span_label(span, format!("{}s cannot evaluate destructors",
                                                      self.mode))
                            .emit();
                    }
                }
            }
        } else {
            // Qualify any operands inside other terminators.
            self.super_terminator_kind(bb, kind, location);
        }
    }

    fn visit_assign(&mut self,
                    _: BasicBlock,
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

    fn visit_statement(&mut self, bb: BasicBlock, statement: &Statement<'tcx>, location: Location) {
        debug!("visit_statement: bb={:?} statement={:?} location={:?}", bb, statement, location);
        match statement.kind {
            StatementKind::Assign(..) => {
                self.super_statement(bb, statement, location);
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

    fn visit_terminator(&mut self,
                        bb: BasicBlock,
                        terminator: &Terminator<'tcx>,
                        location: Location) {
        debug!("visit_terminator: bb={:?} terminator={:?} location={:?}", bb, terminator, location);
        self.super_terminator(bb, terminator, location);
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        mir_const_qualif,
        ..*providers
    };
}

fn mir_const_qualif<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              def_id: DefId)
                              -> (u8, Lrc<BitSet<Local>>) {
    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_validated()`, which steals
    // from `mir_const(), forces this query to execute before
    // performing the steal.
    let mir = &tcx.mir_const(def_id).borrow();

    if mir.return_ty().references_error() {
        tcx.sess.delay_span_bug(mir.span, "mir_const_qualif: Mir had errors");
        return (1 << IsNotConst::IDX, Lrc::new(BitSet::new_empty(0)));
    }

    Checker::new(tcx, def_id, mir, Mode::Const).check_const()
}

pub struct QualifyAndPromoteConstants;

impl MirPass for QualifyAndPromoteConstants {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          src: MirSource<'tcx>,
                          mir: &mut Mir<'tcx>) {
        // There's not really any point in promoting errorful MIR.
        if mir.return_ty().references_error() {
            tcx.sess.delay_span_bug(mir.span, "QualifyAndPromoteConstants: Mir had errors");
            return;
        }

        if src.promoted.is_some() {
            return;
        }

        let def_id = src.def_id();
        let id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let mut const_promoted_temps = None;
        let mode = match tcx.hir().body_owner_kind_by_hir_id(id) {
            hir::BodyOwnerKind::Closure => Mode::Fn,
            hir::BodyOwnerKind::Fn => {
                if tcx.is_const_fn(def_id) {
                    Mode::ConstFn
                } else {
                    Mode::Fn
                }
            }
            hir::BodyOwnerKind::Const => {
                const_promoted_temps = Some(tcx.mir_const_qualif(def_id).1);
                Mode::Const
            }
            hir::BodyOwnerKind::Static(hir::MutImmutable) => Mode::Static,
            hir::BodyOwnerKind::Static(hir::MutMutable) => Mode::StaticMut,
        };

        debug!("run_pass: mode={:?}", mode);
        if mode == Mode::Fn || mode == Mode::ConstFn {
            // This is ugly because Checker holds onto mir,
            // which can't be mutated until its scope ends.
            let (temps, candidates) = {
                let mut checker = Checker::new(tcx, def_id, mir, mode);
                if mode == Mode::ConstFn {
                    if tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you {
                        checker.check_const();
                    } else if tcx.is_min_const_fn(def_id) {
                        // enforce `min_const_fn` for stable const fns
                        use super::qualify_min_const_fn::is_min_const_fn;
                        if let Err((span, err)) = is_min_const_fn(tcx, def_id, mir) {
                            let mut diag = struct_span_err!(
                                tcx.sess,
                                span,
                                E0723,
                                "{} (see issue #57563)",
                                err,
                            );
                            diag.help(
                                "add #![feature(const_fn)] to the crate attributes to enable",
                            );
                            diag.emit();
                        } else {
                            // this should not produce any errors, but better safe than sorry
                            // FIXME(#53819)
                            checker.check_const();
                        }
                    } else {
                        // Enforce a constant-like CFG for `const fn`.
                        checker.check_const();
                    }
                } else {
                    while let Some((bb, data)) = checker.rpo.next() {
                        checker.visit_basic_block_data(bb, data);
                    }
                }

                (checker.temp_promotion_state, checker.promotion_candidates)
            };

            // Do the actual promotion, now that we know what's viable.
            promote_consts::promote_candidates(mir, tcx, temps, candidates);
        } else {
            if !mir.control_flow_destroyed.is_empty() {
                let mut locals = mir.vars_iter();
                if let Some(local) = locals.next() {
                    let span = mir.local_decls[local].source_info.span;
                    let mut error = tcx.sess.struct_span_err(
                        span,
                        &format!(
                            "new features like let bindings are not permitted in {}s \
                            which also use short circuiting operators",
                            mode,
                        ),
                    );
                    for (span, kind) in mir.control_flow_destroyed.iter() {
                        error.span_note(
                            *span,
                            &format!("use of {} here does not actually short circuit due to \
                            the const evaluator presently not being able to do control flow. \
                            See https://github.com/rust-lang/rust/issues/49146 for more \
                            information.", kind),
                        );
                    }
                    for local in locals {
                        let span = mir.local_decls[local].source_info.span;
                        error.span_note(
                            span,
                            "more locals defined here",
                        );
                    }
                    error.emit();
                }
            }
            let promoted_temps = if mode == Mode::Const {
                // Already computed by `mir_const_qualif`.
                const_promoted_temps.unwrap()
            } else {
                Checker::new(tcx, def_id, mir, mode).check_const().1
            };

            // In `const` and `static` everything without `StorageDead`
            // is `'static`, we don't have to create promoted MIR fragments,
            // just remove `Drop` and `StorageDead` on "promoted" locals.
            debug!("run_pass: promoted_temps={:?}", promoted_temps);
            for block in mir.basic_blocks_mut() {
                block.statements.retain(|statement| {
                    match statement.kind {
                        StatementKind::StorageDead(index) => {
                            !promoted_temps.contains(index)
                        }
                        _ => true
                    }
                });
                let terminator = block.terminator_mut();
                match terminator.kind {
                    TerminatorKind::Drop { location: Place::Local(index), target, .. } => {
                        if promoted_temps.contains(index) {
                            terminator.kind = TerminatorKind::Goto {
                                target,
                            };
                        }
                    }
                    _ => {}
                }
            }
        }

        // Statics must be Sync.
        if mode == Mode::Static {
            // `#[thread_local]` statics don't have to be `Sync`.
            for attr in &tcx.get_attrs(def_id)[..] {
                if attr.check_name("thread_local") {
                    return;
                }
            }
            let ty = mir.return_ty();
            tcx.infer_ctxt().enter(|infcx| {
                let param_env = ty::ParamEnv::empty();
                let cause = traits::ObligationCause::new(mir.span, id, traits::SharedStatic);
                let mut fulfillment_cx = traits::FulfillmentContext::new();
                fulfillment_cx.register_bound(&infcx,
                                              param_env,
                                              ty,
                                              tcx.require_lang_item(lang_items::SyncTraitLangItem),
                                              cause);
                if let Err(err) = fulfillment_cx.select_all_or_error(&infcx) {
                    infcx.report_fulfillment_errors(&err, None, false);
                }
            });
        }
    }
}

fn args_required_const(tcx: TyCtxt<'_, '_, '_>, def_id: DefId) -> Option<FxHashSet<usize>> {
    let attrs = tcx.get_attrs(def_id);
    let attr = attrs.iter().find(|a| a.check_name("rustc_args_required_const"))?;
    let mut ret = FxHashSet::default();
    for meta in attr.meta_item_list()? {
        match meta.literal()?.node {
            LitKind::Int(a, _) => { ret.insert(a as usize); }
            _ => return None,
        }
    }
    Some(ret)
}
