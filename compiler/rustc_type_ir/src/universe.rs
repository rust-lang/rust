use tracing::{debug, instrument};

use crate::data_structures::HashSet;
use crate::inherent::*;
use crate::visit::TypeVisitableExt;
use crate::{
    ConstKind, InferCtxtLike, InferTy, Interner, Region, RegionKind, TyKind, TypeFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitor, UniverseIndex,
};

/// The largest universe a variable or placeholder was from in `t`
pub fn max_universe<Infcx: InferCtxtLike<Interner = I>, I: Interner, T: TypeFoldable<I>>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    max_universe_inner::<_, _, _, true, true, true>(infcx, t)
}

/// The largest universe a variable was from in `t`
pub fn max_universe_of_infer_vars<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    max_universe_inner::<_, _, _, false, false, true>(infcx, t)
}

/// The largest universe a placeholder was from in `t`
pub fn max_universe_of_placeholders<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    max_universe_inner::<_, _, _, true, true, false>(infcx, t)
}

/// The largest universe a type or const placeholder was from in `t`
pub fn max_universe_of_non_region_placeholders<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    max_universe_inner::<_, _, _, true, false, false>(infcx, t)
}

fn max_universe_inner<
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
    const VISIT_NON_REGION_PLACEHOLDER: bool,
    const VISIT_REGION_PLACEHOLDER: bool,
    const VISIT_INFER: bool,
>(
    infcx: &Infcx,
    t: T,
) -> UniverseIndex {
    if !MaxUniverse::<
        Infcx,
        I,
        VISIT_NON_REGION_PLACEHOLDER,
        VISIT_REGION_PLACEHOLDER,
        VISIT_INFER,
    >::needs_visit(&t)
    {
        return UniverseIndex::ROOT;
    }

    let mut visitor = MaxUniverse::<
        _,
        _,
        VISIT_NON_REGION_PLACEHOLDER,
        VISIT_REGION_PLACEHOLDER,
        VISIT_INFER,
    >::new(infcx);
    // FIXME: make this a debug_assert and let callers resolve vars. Then the input only needs to
    // be `TypeVisitable`.
    let t = infcx.resolve_vars_if_possible(t);
    t.visit_with(&mut visitor);
    visitor.max_universe()
}

struct MaxUniverse<
    'a,
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    const VISIT_NON_REGION_PLACEHOLDER: bool,
    const VISIT_REGION_PLACEHOLDER: bool,
    const VISIT_INFER: bool,
> {
    max_universe: UniverseIndex,
    infcx: &'a Infcx,
    cache: HashSet<I::Ty>,
}

impl<
    'a,
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    const VISIT_NON_REGION_PLACEHOLDER: bool,
    const VISIT_REGION_PLACEHOLDER: bool,
    const VISIT_INFER: bool,
> MaxUniverse<'a, Infcx, I, VISIT_NON_REGION_PLACEHOLDER, VISIT_REGION_PLACEHOLDER, VISIT_INFER>
{
    fn new(infcx: &'a Infcx) -> Self {
        MaxUniverse { infcx, max_universe: UniverseIndex::ROOT, cache: Default::default() }
    }

    fn max_universe(self) -> UniverseIndex {
        self.max_universe
    }

    #[instrument(ret, level = "debug")]
    fn needs_visit<T: TypeVisitable<I>>(t: &T) -> bool {
        ((VISIT_NON_REGION_PLACEHOLDER || VISIT_REGION_PLACEHOLDER) && t.has_placeholders())
            || (VISIT_INFER && t.has_infer())
    }
}

impl<
    'a,
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    const VISIT_NON_REGION_PLACEHOLDER: bool,
    const VISIT_REGION_PLACEHOLDER: bool,
    const VISIT_INFER: bool,
> TypeVisitor<I>
    for MaxUniverse<
        'a,
        Infcx,
        I,
        VISIT_NON_REGION_PLACEHOLDER,
        VISIT_REGION_PLACEHOLDER,
        VISIT_INFER,
    >
{
    type Result = ();

    fn visit_ty(&mut self, t: I::Ty) {
        if !Self::needs_visit(&t) {
            return;
        }

        if self.cache.contains(&t) {
            return;
        }

        match t.kind() {
            TyKind::Placeholder(p) if VISIT_NON_REGION_PLACEHOLDER => {
                self.max_universe = self.max_universe.max(p.universe)
            }
            TyKind::Infer(InferTy::TyVar(inf)) if VISIT_INFER => {
                let u = self.infcx.universe_of_ty(inf).unwrap();
                debug!("var {inf:?} in universe {u:?}");
                self.max_universe = self.max_universe.max(u);
            }
            _ => t.super_visit_with(self),
        }

        assert!(self.cache.insert(t), "we shouldn't visit {t:?} twice");
    }

    fn visit_const(&mut self, c: I::Const) {
        if !Self::needs_visit(&c) {
            return;
        }

        match c.kind() {
            ConstKind::Placeholder(p) if VISIT_NON_REGION_PLACEHOLDER => {
                self.max_universe = self.max_universe.max(p.universe)
            }
            ConstKind::Infer(rustc_type_ir::InferConst::Var(inf)) if VISIT_INFER => {
                let u = self.infcx.universe_of_ct(inf).unwrap();
                debug!("var {inf:?} in universe {u:?}");
                self.max_universe = self.max_universe.max(u);
            }
            _ => c.super_visit_with(self),
        }
    }

    fn visit_region(&mut self, r: Region<I>) {
        match r.kind() {
            RegionKind::RePlaceholder(p) if VISIT_REGION_PLACEHOLDER => {
                self.max_universe = self.max_universe.max(p.universe)
            }
            RegionKind::ReVar(var) if VISIT_INFER => {
                match self.infcx.opportunistic_resolve_lt_var(var).kind() {
                    RegionKind::RePlaceholder(p) if VISIT_REGION_PLACEHOLDER => {
                        self.max_universe = self.max_universe.max(p.universe)
                    }
                    RegionKind::ReVar(var) if VISIT_INFER => {
                        let u = self.infcx.universe_of_lt(var).unwrap();
                        debug!("var {var:?} in universe {u:?}");
                        self.max_universe = self.max_universe.max(u);
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
}
