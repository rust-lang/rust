use core::panic;

use rustc_type_ir::data_structures::IndexMap;
use rustc_type_ir::inherent::*;
use rustc_type_ir::{
    self as ty, InferCtxtLike, Interner, PlaceholderConst, PlaceholderRegion, PlaceholderType,
    TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

pub struct BoundVarReplacer<'a, Infcx, I = <Infcx as InferCtxtLike>::Interner>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    infcx: &'a Infcx,
    // These three maps track the bound variable that were replaced by placeholders. It might be
    // nice to remove these since we already have the `kind` in the placeholder; we really just need
    // the `var` (but we *could* bring that into scope if we were to track them as we pass them).
    mapped_regions: IndexMap<ty::PlaceholderRegion<I>, ty::BoundRegion<I>>,
    mapped_types: IndexMap<ty::PlaceholderType<I>, ty::BoundTy<I>>,
    mapped_consts: IndexMap<ty::PlaceholderConst<I>, ty::BoundConst<I>>,
    // The current depth relative to *this* folding, *not* the entire normalization. In other words,
    // the depth of binders we've passed here.
    current_index: ty::DebruijnIndex,
    // The `UniverseIndex` of the binding levels above us. These are optional, since we are lazy:
    // we don't actually create a universe until we see a bound var we have to replace.
    universe_indices: &'a mut Vec<Option<ty::UniverseIndex>>,
}

impl<'a, Infcx, I> BoundVarReplacer<'a, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    /// Returns a type with all bound vars replaced by placeholders,
    /// together with mappings from the new placeholders back to the original variable.
    ///
    /// Panics if there are any bound vars that use a binding level above `universe_indices.len()`.
    pub fn replace_bound_vars<T: TypeFoldable<I>>(
        infcx: &'a Infcx,
        universe_indices: &'a mut Vec<Option<ty::UniverseIndex>>,
        value: T,
    ) -> (
        T,
        IndexMap<ty::PlaceholderRegion<I>, ty::BoundRegion<I>>,
        IndexMap<ty::PlaceholderType<I>, ty::BoundTy<I>>,
        IndexMap<ty::PlaceholderConst<I>, ty::BoundConst<I>>,
    ) {
        let mut replacer = BoundVarReplacer {
            infcx,
            mapped_regions: Default::default(),
            mapped_types: Default::default(),
            mapped_consts: Default::default(),
            current_index: ty::INNERMOST,
            universe_indices,
        };

        let value = value.fold_with(&mut replacer);

        (value, replacer.mapped_regions, replacer.mapped_types, replacer.mapped_consts)
    }

    fn universe_for(&mut self, debruijn: ty::DebruijnIndex) -> ty::UniverseIndex {
        let infcx = self.infcx;
        let index =
            self.universe_indices.len() + self.current_index.as_usize() - debruijn.as_usize() - 1;
        let universe = self.universe_indices[index].unwrap_or_else(|| {
            for i in self.universe_indices.iter_mut().take(index + 1) {
                *i = i.or_else(|| Some(infcx.create_next_universe()))
            }
            self.universe_indices[index].unwrap()
        });
        universe
    }
}

impl<Infcx, I> TypeFolder<I> for BoundVarReplacer<'_, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn cx(&self) -> I {
        self.infcx.cx()
    }

    fn fold_binder<T: TypeFoldable<I>>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        match r.kind() {
            ty::ReBound(ty::BoundVarIndexKind::Bound(debruijn), _)
                if debruijn.as_usize()
                    >= self.current_index.as_usize() + self.universe_indices.len() =>
            {
                panic!(
                    "Bound vars {r:#?} outside of `self.universe_indices`: {:#?}",
                    self.universe_indices
                );
            }
            ty::ReBound(ty::BoundVarIndexKind::Bound(debruijn), br)
                if debruijn >= self.current_index =>
            {
                let universe = self.universe_for(debruijn);
                let p = PlaceholderRegion::new(universe, br);
                self.mapped_regions.insert(p, br);
                Region::new_placeholder(self.cx(), p)
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        match t.kind() {
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                panic!(
                    "Bound vars {t:#?} outside of `self.universe_indices`: {:#?}",
                    self.universe_indices
                );
            }
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn >= self.current_index =>
            {
                let universe = self.universe_for(debruijn);
                let p = PlaceholderType::new(universe, bound_ty);
                self.mapped_types.insert(p, bound_ty);
                Ty::new_placeholder(self.cx(), p)
            }
            _ if t.has_vars_bound_at_or_above(self.current_index) => t.super_fold_with(self),
            _ => t,
        }
    }

    fn fold_const(&mut self, ct: I::Const) -> I::Const {
        match ct.kind() {
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                panic!(
                    "Bound vars {ct:#?} outside of `self.universe_indices`: {:#?}",
                    self.universe_indices
                );
            }
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_const)
                if debruijn >= self.current_index =>
            {
                let universe = self.universe_for(debruijn);
                let p = PlaceholderConst::new(universe, bound_const);
                self.mapped_consts.insert(p, bound_const);
                Const::new_placeholder(self.cx(), p)
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

/// The inverse of [`BoundVarReplacer`]: replaces placeholders with the bound vars from which
/// they came.
pub struct PlaceholderReplacer<'a, I: Interner> {
    cx: I,
    mapped_regions: IndexMap<ty::PlaceholderRegion<I>, ty::BoundRegion<I>>,
    mapped_types: IndexMap<ty::PlaceholderType<I>, ty::BoundTy<I>>,
    mapped_consts: IndexMap<ty::PlaceholderConst<I>, ty::BoundConst<I>>,
    universe_indices: &'a [Option<ty::UniverseIndex>],
    current_index: ty::DebruijnIndex,
}

impl<'a, I: Interner> PlaceholderReplacer<'a, I> {
    pub fn replace_placeholders<T: TypeFoldable<I>>(
        cx: I,
        mapped_regions: IndexMap<ty::PlaceholderRegion<I>, ty::BoundRegion<I>>,
        mapped_types: IndexMap<ty::PlaceholderType<I>, ty::BoundTy<I>>,
        mapped_consts: IndexMap<ty::PlaceholderConst<I>, ty::BoundConst<I>>,
        universe_indices: &'a [Option<ty::UniverseIndex>],
        value: T,
    ) -> T {
        let mut replacer = PlaceholderReplacer {
            cx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            universe_indices,
            current_index: ty::INNERMOST,
        };
        value.fold_with(&mut replacer)
    }

    fn debruijn_for_universe(&self, universe: ty::UniverseIndex) -> ty::DebruijnIndex {
        let index = self
            .universe_indices
            .iter()
            .position(|u| matches!(u, Some(u_idx) if *u_idx == universe))
            .unwrap_or_else(|| panic!("unexpected placeholder universe {universe:?}"));

        ty::DebruijnIndex::from_usize(
            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
        )
    }
}

impl<I: Interner> TypeFolder<I> for PlaceholderReplacer<'_, I> {
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_binder<T: TypeFoldable<I>>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T> {
        if !t.has_placeholders() && !t.has_infer() {
            return t;
        }

        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        if let ty::RePlaceholder(p) = r.kind() {
            if let Some(replace_var) = self.mapped_regions.get(&p) {
                let db = self.debruijn_for_universe(p.universe());
                return Region::new_bound(self.cx(), db, *replace_var);
            }
        }

        r
    }

    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        if let ty::Placeholder(p) = ty.kind() {
            match self.mapped_types.get(&p) {
                Some(replace_var) => {
                    let db = self.debruijn_for_universe(p.universe());
                    Ty::new_bound(self.cx(), db, *replace_var)
                }
                None => ty,
            }
        } else if ty.has_placeholders() || ty.has_infer() {
            ty.super_fold_with(self)
        } else {
            ty
        }
    }

    fn fold_const(&mut self, ct: I::Const) -> I::Const {
        if let ty::ConstKind::Placeholder(p) = ct.kind() {
            match self.mapped_consts.get(&p) {
                Some(replace_var) => {
                    let db = self.debruijn_for_universe(p.universe());
                    Const::new_bound(self.cx(), db, *replace_var)
                }
                None => ct,
            }
        } else if ct.has_placeholders() || ct.has_infer() {
            ct.super_fold_with(self)
        } else {
            ct
        }
    }
}
