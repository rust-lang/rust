//! Logic required to produce a monomorphic stable body.
//!
//! We first retrieve and monomorphize the rustc body representation, i.e., we generate a
//! monomorphic body using internal representation.
//! After that, we convert the internal representation into a stable one.

use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::ty::{self, TyCtxt};

use crate::rustc_smir::{Stable, Tables};
use crate::stable_mir;

/// Builds a monomorphic body for a given instance.
pub(crate) struct BodyBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
}

impl<'tcx> BodyBuilder<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, instance: ty::Instance<'tcx>) -> Self {
        let instance = match instance.def {
            // To get the fallback body of an intrinsic, we need to convert it to an item.
            ty::InstanceKind::Intrinsic(def_id) => ty::Instance::new(def_id, instance.args),
            _ => instance,
        };
        BodyBuilder { tcx, instance }
    }

    /// Build a stable monomorphic body for a given instance based on the MIR body.
    ///
    /// All constants are also evaluated.
    pub(crate) fn build(mut self, tables: &mut Tables<'tcx>) -> stable_mir::mir::Body {
        let body = tables.tcx.instance_mir(self.instance.def).clone();
        let mono_body = if !self.instance.args.is_empty()
            // Without the `generic_const_exprs` feature gate, anon consts in signatures do not
            // get generic parameters. Which is wrong, but also not a problem without
            // generic_const_exprs
            || self.tcx.def_kind(self.instance.def_id()) != DefKind::AnonConst
        {
            let mut mono_body = self.instance.instantiate_mir_and_normalize_erasing_regions(
                tables.tcx,
                ty::TypingEnv::fully_monomorphized(),
                ty::EarlyBinder::bind(body),
            );
            self.visit_body(&mut mono_body);
            mono_body
        } else {
            // Already monomorphic.
            body
        };
        mono_body.stable(tables)
    }
}

impl<'tcx> MutVisitor<'tcx> for BodyBuilder<'tcx> {
    fn visit_const_operand(
        &mut self,
        constant: &mut mir::ConstOperand<'tcx>,
        location: mir::Location,
    ) {
        let const_ = constant.const_;
        let val = match const_.eval(self.tcx, ty::TypingEnv::fully_monomorphized(), constant.span) {
            Ok(v) => v,
            Err(mir::interpret::ErrorHandled::Reported(..)) => return,
            Err(mir::interpret::ErrorHandled::TooGeneric(..)) => {
                unreachable!("Failed to evaluate instance constant: {:?}", const_)
            }
        };
        let ty = constant.ty();
        constant.const_ = mir::Const::Val(val, ty);
        self.super_const_operand(constant, location);
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}
