use rustc_infer::infer::canonical::Canonical;
use rustc_middle::ty::{self, Ty, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::traits::query::type_op;

use std::fmt;
use std::rc::Rc;

use crate::borrow_check::region_infer::values::RegionElement;
use crate::borrow_check::MirBorrowckCtxt;

#[derive(Clone)]
crate struct UniverseInfo<'tcx>(UniverseInfoInner<'tcx>);

/// What operation a universe was created for.
#[derive(Clone)]
enum UniverseInfoInner<'tcx> {
    /// Relating two types which have binders.
    RelateTys { expected: Ty<'tcx>, found: Ty<'tcx> },
    /// Created from performing a `TypeOp`.
    TypeOp(Rc<dyn TypeOpInfo<'tcx> + 'tcx>),
    /// Any other reason.
    Other,
}

impl UniverseInfo<'tcx> {
    crate fn other() -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::Other)
    }

    crate fn relate(expected: Ty<'tcx>, found: Ty<'tcx>) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::RelateTys { expected, found })
    }

    crate fn _report_error(
        &self,
        _mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        _placeholder: ty::PlaceholderRegion,
        _error_element: RegionElement,
        _span: Span,
    ) {
        todo!();
    }
}

crate trait ToUniverseInfo<'tcx> {
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx>;
}

impl<'tcx> ToUniverseInfo<'tcx>
    for Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::prove_predicate::ProvePredicate<'tcx>>>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::TypeOp(Rc::new(PredicateQuery {
            _canonical_query: self,
            _base_universe: base_universe,
        })))
    }
}

impl<'tcx, T: Copy + fmt::Display + TypeFoldable<'tcx> + 'tcx> ToUniverseInfo<'tcx>
    for Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Normalize<T>>>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::TypeOp(Rc::new(NormalizeQuery {
            _canonical_query: self,
            _base_universe: base_universe,
        })))
    }
}

impl<'tcx> ToUniverseInfo<'tcx>
    for Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::AscribeUserType<'tcx>>>
{
    fn to_universe_info(self, _base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        // Ascribe user type isn't usually called on types that have different
        // bound regions.
        UniverseInfo::other()
    }
}

impl<'tcx, F, G> ToUniverseInfo<'tcx> for Canonical<'tcx, type_op::custom::CustomTypeOp<F, G>> {
    fn to_universe_info(self, _base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        // We can't rerun custom type ops.
        UniverseInfo::other()
    }
}

#[allow(unused_lifetimes)]
trait TypeOpInfo<'tcx> {
    // TODO: Methods for rerunning type op and reporting an error
}

struct PredicateQuery<'tcx> {
    _canonical_query:
        Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::prove_predicate::ProvePredicate<'tcx>>>,
    _base_universe: ty::UniverseIndex,
}

impl TypeOpInfo<'tcx> for PredicateQuery<'tcx> {}

struct NormalizeQuery<'tcx, T> {
    _canonical_query: Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Normalize<T>>>,
    _base_universe: ty::UniverseIndex,
}

impl<T> TypeOpInfo<'tcx> for NormalizeQuery<'tcx, T> where
    T: Copy + fmt::Display + TypeFoldable<'tcx> + 'tcx
{
}
