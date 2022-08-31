//! Polymorphization Analysis
//! =========================
//!
//! This module implements an analysis of functions, methods and closures to determine which
//! generic parameters are unused (and eventually, in what ways generic parameters are used - only
//! for their size, offset of a field, etc.).

use rustc_hir::{def::DefKind, def_id::DefId, ConstContext};
use rustc_index::bit_set::FiniteBitSet;
use rustc_middle::mir::{
    visit::{TyContext, Visitor},
    Local, LocalDecl, Location,
};
use rustc_middle::ty::{
    self,
    query::Providers,
    subst::SubstsRef,
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
    Const, Ty, TyCtxt,
};
use rustc_span::symbol::sym;
use std::convert::TryInto;
use std::ops::ControlFlow;

use crate::errors::UnusedGenericParams;

/// Provide implementations of queries relating to polymorphization analysis.
pub fn provide(providers: &mut Providers) {
    providers.unused_generic_params = unused_generic_params;
}

/// Determine which generic parameters are used by the instance.
///
/// Returns a bitset where bits representing unused parameters are set (`is_empty` indicates all
/// parameters are used).
fn unused_generic_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceDef<'tcx>,
) -> FiniteBitSet<u32> {
    if !tcx.sess.opts.unstable_opts.polymorphize {
        // If polymorphization disabled, then all parameters are used.
        return FiniteBitSet::new_empty();
    }

    let def_id = instance.def_id();
    // Exit early if this instance should not be polymorphized.
    if !should_polymorphize(tcx, def_id, instance) {
        return FiniteBitSet::new_empty();
    }

    let generics = tcx.generics_of(def_id);
    debug!(?generics);

    // Exit early when there are no parameters to be unused.
    if generics.count() == 0 {
        return FiniteBitSet::new_empty();
    }

    // Create a bitset with N rightmost ones for each parameter.
    let generics_count: u32 =
        generics.count().try_into().expect("more generic parameters than can fit into a `u32`");
    let mut unused_parameters = FiniteBitSet::<u32>::new_empty();
    unused_parameters.set_range(0..generics_count);
    debug!(?unused_parameters, "(start)");

    mark_used_by_default_parameters(tcx, def_id, generics, &mut unused_parameters);
    debug!(?unused_parameters, "(after default)");

    // Visit MIR and accumulate used generic parameters.
    let body = match tcx.hir().body_const_context(def_id.expect_local()) {
        // Const functions are actually called and should thus be considered for polymorphization
        // via their runtime MIR.
        Some(ConstContext::ConstFn) | None => tcx.optimized_mir(def_id),
        Some(_) => tcx.mir_for_ctfe(def_id),
    };
    let mut vis = MarkUsedGenericParams { tcx, def_id, unused_parameters: &mut unused_parameters };
    vis.visit_body(body);
    debug!(?unused_parameters, "(end)");

    // Emit errors for debugging and testing if enabled.
    if !unused_parameters.is_empty() {
        emit_unused_generic_params_error(tcx, def_id, generics, &unused_parameters);
    }

    unused_parameters
}

/// Returns `true` if the instance should be polymorphized.
fn should_polymorphize<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    instance: ty::InstanceDef<'tcx>,
) -> bool {
    // If an instance's MIR body is not polymorphic then the modified substitutions that are
    // derived from polymorphization's result won't make any difference.
    if !instance.has_polymorphic_mir_body() {
        return false;
    }

    // Don't polymorphize intrinsics or virtual calls - calling `instance_mir` will panic.
    if matches!(instance, ty::InstanceDef::Intrinsic(..) | ty::InstanceDef::Virtual(..)) {
        return false;
    }

    // Polymorphization results are stored in cross-crate metadata only when there are unused
    // parameters, so assume that non-local items must have only used parameters (else this query
    // would not be invoked, and the cross-crate metadata used instead).
    if !def_id.is_local() {
        return false;
    }

    // Foreign items have no bodies to analyze.
    if tcx.is_foreign_item(def_id) {
        return false;
    }

    // Make sure there is MIR available.
    match tcx.hir().body_const_context(def_id.expect_local()) {
        Some(ConstContext::ConstFn) | None if !tcx.is_mir_available(def_id) => {
            debug!("no mir available");
            return false;
        }
        Some(_) if !tcx.is_ctfe_mir_available(def_id) => {
            debug!("no ctfe mir available");
            return false;
        }
        _ => true,
    }
}

/// Some parameters are considered used-by-default, such as non-generic parameters and the dummy
/// generic parameters from closures, this function marks them as used. `leaf_is_closure` should
/// be `true` if the item that `unused_generic_params` was invoked on is a closure.
#[instrument(level = "debug", skip(tcx, def_id, generics, unused_parameters))]
fn mark_used_by_default_parameters<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    generics: &'tcx ty::Generics,
    unused_parameters: &mut FiniteBitSet<u32>,
) {
    match tcx.def_kind(def_id) {
        DefKind::Closure | DefKind::Generator => {
            for param in &generics.params {
                debug!(?param, "(closure/gen)");
                unused_parameters.clear(param.index);
            }
        }
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::Fn
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static(_)
        | DefKind::Ctor(_, _)
        | DefKind::AssocFn
        | DefKind::AssocConst
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::ImplTraitPlaceholder
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Impl => {
            for param in &generics.params {
                debug!(?param, "(other)");
                if let ty::GenericParamDefKind::Lifetime = param.kind {
                    unused_parameters.clear(param.index);
                }
            }
        }
    }

    if let Some(parent) = generics.parent {
        mark_used_by_default_parameters(tcx, parent, tcx.generics_of(parent), unused_parameters);
    }
}

/// Emit errors for the function annotated by `#[rustc_polymorphize_error]`, labelling each generic
/// parameter which was unused.
#[instrument(level = "debug", skip(tcx, generics))]
fn emit_unused_generic_params_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    generics: &'tcx ty::Generics,
    unused_parameters: &FiniteBitSet<u32>,
) {
    let base_def_id = tcx.typeck_root_def_id(def_id);
    if !tcx.has_attr(base_def_id, sym::rustc_polymorphize_error) {
        return;
    }

    let fn_span = match tcx.opt_item_ident(def_id) {
        Some(ident) => ident.span,
        _ => tcx.def_span(def_id),
    };

    let mut param_spans = Vec::new();
    let mut param_names = Vec::new();
    let mut next_generics = Some(generics);
    while let Some(generics) = next_generics {
        for param in &generics.params {
            if unused_parameters.contains(param.index).unwrap_or(false) {
                debug!(?param);
                let def_span = tcx.def_span(param.def_id);
                param_spans.push(def_span);
                param_names.push(param.name.to_string());
            }
        }

        next_generics = generics.parent.map(|did| tcx.generics_of(did));
    }

    tcx.sess.emit_err(UnusedGenericParams { span: fn_span, param_spans, param_names });
}

/// Visitor used to aggregate generic parameter uses.
struct MarkUsedGenericParams<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    unused_parameters: &'a mut FiniteBitSet<u32>,
}

impl<'a, 'tcx> MarkUsedGenericParams<'a, 'tcx> {
    /// Invoke `unused_generic_params` on a body contained within the current item (e.g.
    /// a closure, generator or constant).
    #[instrument(level = "debug", skip(self, def_id, substs))]
    fn visit_child_body(&mut self, def_id: DefId, substs: SubstsRef<'tcx>) {
        let instance = ty::InstanceDef::Item(ty::WithOptConstParam::unknown(def_id));
        let unused = self.tcx.unused_generic_params(instance);
        debug!(?self.unused_parameters, ?unused);
        for (i, arg) in substs.iter().enumerate() {
            let i = i.try_into().unwrap();
            if !unused.contains(i).unwrap_or(false) {
                arg.visit_with(self);
            }
        }
        debug!(?self.unused_parameters);
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MarkUsedGenericParams<'a, 'tcx> {
    #[instrument(level = "debug", skip(self, local))]
    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        if local == Local::from_usize(1) {
            let def_kind = self.tcx.def_kind(self.def_id);
            if matches!(def_kind, DefKind::Closure | DefKind::Generator) {
                // Skip visiting the closure/generator that is currently being processed. This only
                // happens because the first argument to the closure is a reference to itself and
                // that will call `visit_substs`, resulting in each generic parameter captured being
                // considered used by default.
                debug!("skipping closure substs");
                return;
            }
        }

        self.super_local_decl(local, local_decl);
    }

    fn visit_const(&mut self, c: Const<'tcx>, _: Location) {
        c.visit_with(self);
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>, _: TyContext) {
        ty.visit_with(self);
    }
}

impl<'a, 'tcx> TypeVisitor<'tcx> for MarkUsedGenericParams<'a, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn visit_const(&mut self, c: Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        if !c.has_param_types_or_consts() {
            return ControlFlow::CONTINUE;
        }

        match c.kind() {
            ty::ConstKind::Param(param) => {
                debug!(?param);
                self.unused_parameters.clear(param.index);
                ControlFlow::CONTINUE
            }
            ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs: _, promoted: Some(p)})
                // Avoid considering `T` unused when constants are of the form:
                //   `<Self as Foo<T>>::foo::promoted[p]`
                if self.def_id == def.did && !self.tcx.generics_of(def.did).has_self =>
            {
                // If there is a promoted, don't look at the substs - since it will always contain
                // the generic parameters, instead, traverse the promoted MIR.
                let promoted = self.tcx.promoted_mir(def.did);
                self.visit_body(&promoted[p]);
                ControlFlow::CONTINUE
            }
            ty::ConstKind::Unevaluated(uv)
                if matches!(self.tcx.def_kind(uv.def.did), DefKind::AnonConst | DefKind::InlineConst) =>
            {
                self.visit_child_body(uv.def.did, uv.substs);
                ControlFlow::CONTINUE
            }
            _ => c.super_visit_with(self),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if !ty.has_param_types_or_consts() {
            return ControlFlow::CONTINUE;
        }

        match *ty.kind() {
            ty::Closure(def_id, substs) | ty::Generator(def_id, substs, ..) => {
                debug!(?def_id);
                // Avoid cycle errors with generators.
                if def_id == self.def_id {
                    return ControlFlow::CONTINUE;
                }

                // Consider any generic parameters used by any closures/generators as used in the
                // parent.
                self.visit_child_body(def_id, substs);
                ControlFlow::CONTINUE
            }
            ty::Param(param) => {
                debug!(?param);
                self.unused_parameters.clear(param.index);
                ControlFlow::CONTINUE
            }
            _ => ty.super_visit_with(self),
        }
    }
}

/// Visitor used to check if a generic parameter is used.
struct HasUsedGenericParams<'a> {
    unused_parameters: &'a FiniteBitSet<u32>,
}

impl<'a, 'tcx> TypeVisitor<'tcx> for HasUsedGenericParams<'a> {
    type BreakTy = ();

    #[instrument(level = "debug", skip(self))]
    fn visit_const(&mut self, c: Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        if !c.has_param_types_or_consts() {
            return ControlFlow::CONTINUE;
        }

        match c.kind() {
            ty::ConstKind::Param(param) => {
                if self.unused_parameters.contains(param.index).unwrap_or(false) {
                    ControlFlow::CONTINUE
                } else {
                    ControlFlow::BREAK
                }
            }
            _ => c.super_visit_with(self),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if !ty.has_param_types_or_consts() {
            return ControlFlow::CONTINUE;
        }

        match ty.kind() {
            ty::Param(param) => {
                if self.unused_parameters.contains(param.index).unwrap_or(false) {
                    ControlFlow::CONTINUE
                } else {
                    ControlFlow::BREAK
                }
            }
            _ => ty.super_visit_with(self),
        }
    }
}
