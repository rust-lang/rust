//! A folding traversal mechanism for complex data structures that contain type
//! information.
//!
//! This is a modifying traversal. It consumes the data structure, producing a
//! (possibly) modified version of it. Both fallible and infallible versions are
//! available. The name is potentially confusing, because this traversal is more
//! like `Iterator::map` than `Iterator::fold`.
//!
//! This traversal has limited flexibility. Only a small number of "types of
//! interest" within the complex data structures can receive custom
//! modification. These are the ones containing the most important type-related
//! information, such as `Ty`, `Predicate`, `Region`, and `Const`.
//!
//! There are three groups of traits involved in each traversal.
//! - `TypeFoldable`. This is implemented once for many types, including:
//!   - Types of interest, for which the the methods delegate to the
//!     folder.
//!   - All other types, including generic containers like `Vec` and `Option`.
//!     It defines a "skeleton" of how they should be folded.
//! - `TypeSuperFoldable`. This is implemented only for each type of interest,
//!   and defines the folding "skeleton" for these types.
//! - `TypeFolder`/`FallibleTypeFolder. One of these is implemented for each
//!   folder. This defines how types of interest are folded.
//!
//! This means each fold is a mixture of (a) generic folding operations, and (b)
//! custom fold operations that are specific to the folder.
//! - The `TypeFoldable` impls handle most of the traversal, and call into
//!   `TypeFolder`/`FallibleTypeFolder` when they encounter a type of interest.
//! - A `TypeFolder`/`FallibleTypeFolder` may call into another `TypeFoldable`
//!   impl, because some of the types of interest are recursive and can contain
//!   other types of interest.
//! - A `TypeFolder`/`FallibleTypeFolder` may also call into a `TypeSuperFoldable`
//!   impl, because each folder might provide custom handling only for some types
//!   of interest, or only for some variants of each type of interest, and then
//!   use default traversal for the remaining cases.
//!
//! For example, if you have `struct S(Ty, U)` where `S: TypeFoldable` and `U:
//! TypeFoldable`, and an instance `s = S(ty, u)`, it would be folded like so:
//! ```text
//! s.fold_with(folder) calls
//! - ty.fold_with(folder) calls
//!   - folder.fold_ty(ty) may call
//!     - ty.super_fold_with(folder)
//! - u.fold_with(folder)
//! ```
use crate::mir;
use crate::ty::{self, Binder, BoundTy, Ty, TyCtxt, TypeVisitable};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;

use std::collections::BTreeMap;

/// This trait is implemented for every type that can be folded,
/// providing the skeleton of the traversal.
///
/// To implement this conveniently, use the derive macro located in
/// `rustc_macros`.
pub trait TypeFoldable<'tcx>: TypeVisitable<'tcx> {
    /// The entry point for folding. To fold a value `t` with a folder `f`
    /// call: `t.try_fold_with(f)`.
    ///
    /// For most types, this just traverses the value, calling `try_fold_with`
    /// on each field/element.
    ///
    /// For types of interest (such as `Ty`), the implementation of method
    /// calls a folder method specifically for that type (such as
    /// `F::try_fold_ty`). This is where control transfers from `TypeFoldable`
    /// to `TypeFolder`.
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error>;

    /// A convenient alternative to `try_fold_with` for use with infallible
    /// folders. Do not override this method, to ensure coherence with
    /// `try_fold_with`.
    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.try_fold_with(folder).into_ok()
    }
}

// This trait is implemented for types of interest.
pub trait TypeSuperFoldable<'tcx>: TypeFoldable<'tcx> {
    /// Provides a default fold for a type of interest. This should only be
    /// called within `TypeFolder` methods, when a non-custom traversal is
    /// desired for the value of the type of interest passed to that method.
    /// For example, in `MyFolder::try_fold_ty(ty)`, it is valid to call
    /// `ty.try_super_fold_with(self)`, but any other folding should be done
    /// with `xyz.try_fold_with(self)`.
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error>;

    /// A convenient alternative to `try_super_fold_with` for use with
    /// infallible folders. Do not override this method, to ensure coherence
    /// with `try_super_fold_with`.
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.try_super_fold_with(folder).into_ok()
    }
}

/// This trait is implemented for every infallible folding traversal. There is
/// a fold method defined for every type of interest. Each such method has a
/// default that does an "identity" fold. Implementations of these methods
/// often fall back to a `super_fold_with` method if the primary argument
/// doesn't satisfy a particular condition.
///
/// A blanket implementation of [`FallibleTypeFolder`] will defer to
/// the infallible methods of this trait to ensure that the two APIs
/// are coherent.
pub trait TypeFolder<'tcx>: FallibleTypeFolder<'tcx, Error = !> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    fn fold_binder<T>(&mut self, t: Binder<'tcx, T>) -> Binder<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        t.super_fold_with(self)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        t.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        r.super_fold_with(self)
    }

    fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        c.super_fold_with(self)
    }

    fn fold_unevaluated(&mut self, uv: ty::Unevaluated<'tcx>) -> ty::Unevaluated<'tcx> {
        uv.super_fold_with(self)
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        p.super_fold_with(self)
    }

    fn fold_mir_const(&mut self, c: mir::ConstantKind<'tcx>) -> mir::ConstantKind<'tcx> {
        bug!("most type folders should not be folding MIR datastructures: {:?}", c)
    }
}

/// This trait is implemented for every folding traversal. There is a fold
/// method defined for every type of interest. Each such method has a default
/// that does an "identity" fold.
///
/// A blanket implementation of this trait (that defers to the relevant
/// method of [`TypeFolder`]) is provided for all infallible folders in
/// order to ensure the two APIs are coherent.
pub trait FallibleTypeFolder<'tcx>: Sized {
    type Error;

    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    fn try_fold_binder<T>(&mut self, t: Binder<'tcx, T>) -> Result<Binder<'tcx, T>, Self::Error>
    where
        T: TypeFoldable<'tcx>,
    {
        t.try_super_fold_with(self)
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        t.try_super_fold_with(self)
    }

    fn try_fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, Self::Error> {
        r.try_super_fold_with(self)
    }

    fn try_fold_const(&mut self, c: ty::Const<'tcx>) -> Result<ty::Const<'tcx>, Self::Error> {
        c.try_super_fold_with(self)
    }

    fn try_fold_unevaluated(
        &mut self,
        c: ty::Unevaluated<'tcx>,
    ) -> Result<ty::Unevaluated<'tcx>, Self::Error> {
        c.try_super_fold_with(self)
    }

    fn try_fold_predicate(
        &mut self,
        p: ty::Predicate<'tcx>,
    ) -> Result<ty::Predicate<'tcx>, Self::Error> {
        p.try_super_fold_with(self)
    }

    fn try_fold_mir_const(
        &mut self,
        c: mir::ConstantKind<'tcx>,
    ) -> Result<mir::ConstantKind<'tcx>, Self::Error> {
        bug!("most type folders should not be folding MIR datastructures: {:?}", c)
    }
}

// This blanket implementation of the fallible trait for infallible folders
// delegates to infallible methods to ensure coherence.
impl<'tcx, F> FallibleTypeFolder<'tcx> for F
where
    F: TypeFolder<'tcx>,
{
    type Error = !;

    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        TypeFolder::tcx(self)
    }

    fn try_fold_binder<T>(&mut self, t: Binder<'tcx, T>) -> Result<Binder<'tcx, T>, !>
    where
        T: TypeFoldable<'tcx>,
    {
        Ok(self.fold_binder(t))
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        Ok(self.fold_ty(t))
    }

    fn try_fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        Ok(self.fold_region(r))
    }

    fn try_fold_const(&mut self, c: ty::Const<'tcx>) -> Result<ty::Const<'tcx>, !> {
        Ok(self.fold_const(c))
    }

    fn try_fold_unevaluated(
        &mut self,
        c: ty::Unevaluated<'tcx>,
    ) -> Result<ty::Unevaluated<'tcx>, !> {
        Ok(self.fold_unevaluated(c))
    }

    fn try_fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> Result<ty::Predicate<'tcx>, !> {
        Ok(self.fold_predicate(p))
    }

    fn try_fold_mir_const(
        &mut self,
        c: mir::ConstantKind<'tcx>,
    ) -> Result<mir::ConstantKind<'tcx>, !> {
        Ok(self.fold_mir_const(c))
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(ty::Const<'tcx>) -> ty::Const<'tcx>,
{
    pub tcx: TyCtxt<'tcx>,
    pub ty_op: F,
    pub lt_op: G,
    pub ct_op: H,
}

impl<'tcx, F, G, H> TypeFolder<'tcx> for BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(ty::Const<'tcx>) -> ty::Const<'tcx>,
{
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t = ty.super_fold_with(self);
        (self.ty_op)(t)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        let r = r.super_fold_with(self);
        (self.lt_op)(r)
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let ct = ct.super_fold_with(self);
        (self.ct_op)(ct)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

impl<'tcx> TyCtxt<'tcx> {
    /// Folds the escaping and free regions in `value` using `f`, and
    /// sets `skipped_regions` to true if any late-bound region was found
    /// and skipped.
    pub fn fold_regions<T>(
        self,
        value: T,
        mut f: impl FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        value.fold_with(&mut RegionFolder::new(self, &mut f))
    }
}

/// Folds over the substructure of a type, visiting its component
/// types and all regions that occur *free* within it.
///
/// That is, `Ty` can contain function or method types that bind
/// regions at the call site (`ReLateBound`), and occurrences of
/// regions (aka "lifetimes") that are bound within a type are not
/// visited by this folder; only regions that occur free will be
/// visited by `fld_r`.

pub struct RegionFolder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Stores the index of a binder *just outside* the stuff we have
    /// visited.  So this begins as INNERMOST; when we pass through a
    /// binder, it is incremented (via `shift_in`).
    current_index: ty::DebruijnIndex,

    /// Callback invokes for each free region. The `DebruijnIndex`
    /// points to the binder *just outside* the ones we have passed
    /// through.
    fold_region_fn:
        &'a mut (dyn FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx> + 'a),
}

impl<'a, 'tcx> RegionFolder<'a, 'tcx> {
    #[inline]
    pub fn new(
        tcx: TyCtxt<'tcx>,
        fold_region_fn: &'a mut dyn FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> RegionFolder<'a, 'tcx> {
        RegionFolder { tcx, current_index: ty::INNERMOST, fold_region_fn }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionFolder<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    #[instrument(skip(self), level = "debug", ret)]
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, _) if debruijn < self.current_index => {
                debug!(?self.current_index, "skipped bound region");
                r
            }
            _ => {
                debug!(?self.current_index, "folding free region");
                (self.fold_region_fn)(r, self.current_index)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Bound vars replacer

pub trait BoundVarReplacerDelegate<'tcx> {
    fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx>;
    fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx>;
    fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx>;
}

pub struct FnMutDelegate<R, T, C> {
    pub regions: R,
    pub types: T,
    pub consts: C,
}
impl<'tcx, R, T, C> BoundVarReplacerDelegate<'tcx> for FnMutDelegate<R, T, C>
where
    R: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
    T: FnMut(ty::BoundTy) -> Ty<'tcx>,
    C: FnMut(ty::BoundVar, Ty<'tcx>) -> ty::Const<'tcx>,
{
    fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx> {
        (self.regions)(br)
    }
    fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx> {
        (self.types)(bt)
    }
    fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx> {
        (self.consts)(bv, ty)
    }
}

/// Replaces the escaping bound vars (late bound regions or bound types) in a type.
struct BoundVarReplacer<'tcx, D> {
    tcx: TyCtxt<'tcx>,

    /// As with `RegionFolder`, represents the index of a binder *just outside*
    /// the ones we have visited.
    current_index: ty::DebruijnIndex,

    delegate: D,
}

impl<'tcx, D: BoundVarReplacerDelegate<'tcx>> BoundVarReplacer<'tcx, D> {
    fn new(tcx: TyCtxt<'tcx>, delegate: D) -> Self {
        BoundVarReplacer { tcx, current_index: ty::INNERMOST, delegate }
    }
}

impl<'tcx, D> TypeFolder<'tcx> for BoundVarReplacer<'tcx, D>
where
    D: BoundVarReplacerDelegate<'tcx>,
{
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn == self.current_index => {
                let ty = self.delegate.replace_ty(bound_ty);
                ty::fold::shift_vars(self.tcx, ty, self.current_index.as_u32())
            }
            _ if t.has_vars_bound_at_or_above(self.current_index) => t.super_fold_with(self),
            _ => t,
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, br) if debruijn == self.current_index => {
                let region = self.delegate.replace_region(br);
                if let ty::ReLateBound(debruijn1, br) = *region {
                    // If the callback returns a late-bound region,
                    // that region should always use the INNERMOST
                    // debruijn index. Then we adjust it to the
                    // correct depth.
                    assert_eq!(debruijn1, ty::INNERMOST);
                    self.tcx.reuse_or_mk_region(region, ty::ReLateBound(debruijn, br))
                } else {
                    region
                }
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Bound(debruijn, bound_const) if debruijn == self.current_index => {
                let ct = self.delegate.replace_const(bound_const, ct.ty());
                ty::fold::shift_vars(self.tcx, ct, self.current_index.as_u32())
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Replaces all regions bound by the given `Binder` with the
    /// results returned by the closure; the closure is expected to
    /// return a free region (relative to this binder), and hence the
    /// binder is removed in the return type. The closure is invoked
    /// once for each unique `BoundRegionKind`; multiple references to the
    /// same `BoundRegionKind` will reuse the previous result. A map is
    /// returned at the end with each bound region and the free region
    /// that replaced it.
    ///
    /// # Panics
    ///
    /// This method only replaces late bound regions. Any types or
    /// constants bound by `value` will cause an ICE.
    pub fn replace_late_bound_regions<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut fld_r: F,
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        T: TypeFoldable<'tcx>,
    {
        let mut region_map = BTreeMap::new();
        let real_fld_r = |br: ty::BoundRegion| *region_map.entry(br).or_insert_with(|| fld_r(br));
        let value = self.replace_late_bound_regions_uncached(value, real_fld_r);
        (value, region_map)
    }

    pub fn replace_late_bound_regions_uncached<T, F>(
        self,
        value: Binder<'tcx, T>,
        replace_regions: F,
    ) -> T
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        T: TypeFoldable<'tcx>,
    {
        let value = value.skip_binder();
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let delegate = FnMutDelegate {
                regions: replace_regions,
                types: |b| bug!("unexpected bound ty in binder: {b:?}"),
                consts: |b, ty| bug!("unexpected bound ct in binder: {b:?} {ty}"),
            };
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    /// Replaces all escaping bound vars. The `fld_r` closure replaces escaping
    /// bound regions; the `fld_t` closure replaces escaping bound types and the `fld_c`
    /// closure replaces escaping bound consts.
    pub fn replace_escaping_bound_vars_uncached<T: TypeFoldable<'tcx>>(
        self,
        value: T,
        delegate: impl BoundVarReplacerDelegate<'tcx>,
    ) -> T {
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    /// Replaces all types or regions bound by the given `Binder`. The `fld_r`
    /// closure replaces bound regions, the `fld_t` closure replaces bound
    /// types, and `fld_c` replaces bound constants.
    pub fn replace_bound_vars_uncached<T: TypeFoldable<'tcx>>(
        self,
        value: Binder<'tcx, T>,
        delegate: impl BoundVarReplacerDelegate<'tcx>,
    ) -> T {
        self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate)
    }

    /// Replaces any late-bound regions bound in `value` with
    /// free variants attached to `all_outlive_scope`.
    pub fn liberate_late_bound_regions<T>(
        self,
        all_outlive_scope: DefId,
        value: ty::Binder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.replace_late_bound_regions_uncached(value, |br| {
            self.mk_region(ty::ReFree(ty::FreeRegion {
                scope: all_outlive_scope,
                bound_region: br.kind,
            }))
        })
    }

    pub fn shift_bound_var_indices<T>(self, bound_vars: usize, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let shift_bv = |bv: ty::BoundVar| ty::BoundVar::from_usize(bv.as_usize() + bound_vars);
        self.replace_escaping_bound_vars_uncached(
            value,
            FnMutDelegate {
                regions: |r: ty::BoundRegion| {
                    self.mk_region(ty::ReLateBound(
                        ty::INNERMOST,
                        ty::BoundRegion { var: shift_bv(r.var), kind: r.kind },
                    ))
                },
                types: |t: ty::BoundTy| {
                    self.mk_ty(ty::Bound(
                        ty::INNERMOST,
                        ty::BoundTy { var: shift_bv(t.var), kind: t.kind },
                    ))
                },
                consts: |c, ty: Ty<'tcx>| {
                    self.mk_const(ty::ConstS {
                        kind: ty::ConstKind::Bound(ty::INNERMOST, shift_bv(c)),
                        ty,
                    })
                },
            },
        )
    }

    /// Replaces any late-bound regions bound in `value` with `'erased`. Useful in codegen but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(self, value: Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.replace_late_bound_regions(value, |_| self.lifetimes.re_erased).0
    }

    /// Rewrite any late-bound regions so that they are anonymous. Region numbers are
    /// assigned starting at 0 and increasing monotonically in the order traversed
    /// by the fold operation.
    ///
    /// The chief purpose of this function is to canonicalize regions so that two
    /// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
    /// structurally identical. For example, `for<'a, 'b> fn(&'a isize, &'b isize)` and
    /// `for<'a, 'b> fn(&'b isize, &'a isize)` will become identical after anonymization.
    pub fn anonymize_late_bound_regions<T>(self, sig: Binder<'tcx, T>) -> Binder<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        let mut counter = 0;
        let inner = self
            .replace_late_bound_regions(sig, |_| {
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_u32(counter),
                    kind: ty::BrAnon(counter),
                };
                let r = self.mk_region(ty::ReLateBound(ty::INNERMOST, br));
                counter += 1;
                r
            })
            .0;
        let bound_vars = self.mk_bound_variable_kinds(
            (0..counter).map(|i| ty::BoundVariableKind::Region(ty::BrAnon(i))),
        );
        Binder::bind_with_vars(inner, bound_vars)
    }

    /// Anonymize all bound variables in `value`, this is mostly used to improve caching.
    pub fn anonymize_bound_vars<T>(self, value: Binder<'tcx, T>) -> Binder<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        struct Anonymize<'a, 'tcx> {
            tcx: TyCtxt<'tcx>,
            map: &'a mut FxIndexMap<ty::BoundVar, ty::BoundVariableKind>,
        }
        impl<'tcx> BoundVarReplacerDelegate<'tcx> for Anonymize<'_, 'tcx> {
            fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx> {
                let entry = self.map.entry(br.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Region(ty::BrAnon(index as u32)))
                    .expect_region();
                let br = ty::BoundRegion { var, kind };
                self.tcx.mk_region(ty::ReLateBound(ty::INNERMOST, br))
            }
            fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx> {
                let entry = self.map.entry(bt.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon))
                    .expect_ty();
                self.tcx.mk_ty(ty::Bound(ty::INNERMOST, BoundTy { var, kind }))
            }
            fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx> {
                let entry = self.map.entry(bv);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let () = entry.or_insert_with(|| ty::BoundVariableKind::Const).expect_const();
                self.tcx.mk_const(ty::ConstS { ty, kind: ty::ConstKind::Bound(ty::INNERMOST, var) })
            }
        }

        let mut map = Default::default();
        let delegate = Anonymize { tcx: self, map: &mut map };
        let inner = self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate);
        let bound_vars = self.mk_bound_variable_kinds(map.into_values());
        Binder::bind_with_vars(inner, bound_vars)
    }
}

///////////////////////////////////////////////////////////////////////////
// Shifter
//
// Shifts the De Bruijn indices on all escaping bound vars by a
// fixed amount. Useful in substitution or when otherwise introducing
// a binding level that is not intended to capture the existing bound
// vars. See comment on `shift_vars_through_binders` method in
// `subst.rs` for more details.

struct Shifter<'tcx> {
    tcx: TyCtxt<'tcx>,
    current_index: ty::DebruijnIndex,
    amount: u32,
}

impl<'tcx> Shifter<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, amount: u32) -> Self {
        Shifter { tcx, current_index: ty::INNERMOST, amount }
    }
}

impl<'tcx> TypeFolder<'tcx> for Shifter<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, br) => {
                if self.amount == 0 || debruijn < self.current_index {
                    r
                } else {
                    let debruijn = debruijn.shifted_in(self.amount);
                    let shifted = ty::ReLateBound(debruijn, br);
                    self.tcx.mk_region(shifted)
                }
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Bound(debruijn, bound_ty) => {
                if self.amount == 0 || debruijn < self.current_index {
                    ty
                } else {
                    let debruijn = debruijn.shifted_in(self.amount);
                    self.tcx.mk_ty(ty::Bound(debruijn, bound_ty))
                }
            }

            _ => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Bound(debruijn, bound_ct) = ct.kind() {
            if self.amount == 0 || debruijn < self.current_index {
                ct
            } else {
                let debruijn = debruijn.shifted_in(self.amount);
                self.tcx.mk_const(ty::ConstS {
                    kind: ty::ConstKind::Bound(debruijn, bound_ct),
                    ty: ct.ty(),
                })
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}

pub fn shift_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    region: ty::Region<'tcx>,
    amount: u32,
) -> ty::Region<'tcx> {
    match *region {
        ty::ReLateBound(debruijn, br) if amount > 0 => {
            tcx.mk_region(ty::ReLateBound(debruijn.shifted_in(amount), br))
        }
        _ => region,
    }
}

pub fn shift_vars<'tcx, T>(tcx: TyCtxt<'tcx>, value: T, amount: u32) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!("shift_vars(value={:?}, amount={})", value, amount);

    value.fold_with(&mut Shifter::new(tcx, amount))
}
