//! Logic required to produce a monomorphic stable body.
//!
//! We first retrieve and monomorphize the rustc body representation, i.e., we generate a
//! monomorphic body using internal representation.
//! After that, we convert the internal representation into a stable one.
use crate::rustc_smir::{Stable, Tables};
use rustc_middle::mir;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::ty::{self, Ty, TyCtxt};

/// Builds a monomorphic body for a given instance.
pub struct BodyBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
}

impl<'tcx> BodyBuilder<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, instance: ty::Instance<'tcx>) -> Self {
        BodyBuilder { tcx, instance }
    }

    pub fn build(mut self, tables: &mut Tables<'tcx>) -> stable_mir::mir::Body {
        let mut body = self.tcx.instance_mir(self.instance.def).clone();
        let generics = self.tcx.generics_of(self.instance.def_id());
        if generics.requires_monomorphization(self.tcx) {
            self.visit_body(&mut body);
        }
        body.stable(tables)
    }

    fn monomorphize<T>(&self, value: T) -> T
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>>,
    {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::ParamEnv::reveal_all(),
            ty::EarlyBinder::bind(value),
        )
    }
}

impl<'tcx> MutVisitor<'tcx> for BodyBuilder<'tcx> {
    fn visit_ty_const(&mut self, ct: &mut ty::Const<'tcx>, _location: mir::Location) {
        *ct = self.monomorphize(*ct);
    }

    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: mir::visit::TyContext) {
        *ty = self.monomorphize(*ty);
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}
