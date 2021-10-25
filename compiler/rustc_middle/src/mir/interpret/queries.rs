use super::{ErrorHandled, EvalToConstValueResult, GlobalId};

use crate::mir;
use crate::ty::subst::InternalSubsts;
use crate::ty::{self, TyCtxt};
use rustc_hir::def_id::DefId;
use rustc_span::Span;

impl<'tcx> TyCtxt<'tcx> {
    /// Evaluates a constant without providing any substitutions. This is useful to evaluate consts
    /// that can't take any generic arguments like statics, const items or enum discriminants. If a
    /// generic parameter is used within the constant `ErrorHandled::ToGeneric` will be returned.
    pub fn const_eval_poly(self, def_id: DefId) -> EvalToConstValueResult<'tcx> {
        // In some situations def_id will have substitutions within scope, but they aren't allowed
        // to be used. So we can't use `Instance::mono`, instead we feed unresolved substitutions
        // into `const_eval` which will return `ErrorHandled::ToGeneric` if any of them are
        // encountered.
        let substs = InternalSubsts::identity_for_item(self, def_id);
        let instance = ty::Instance::new(def_id, substs);
        let cid = GlobalId { instance, promoted: None };
        let param_env = self.param_env(def_id).with_reveal_all_normalized(self);
        self.const_eval_global_id(param_env, cid, None)
    }

    /// Resolves and evaluates a constant.
    ///
    /// The constant can be located on a trait like `<A as B>::C`, in which case the given
    /// substitutions and environment are used to resolve the constant. Alternatively if the
    /// constant has generic parameters in scope the substitutions are used to evaluate the value of
    /// the constant. For example in `fn foo<T>() { let _ = [0; bar::<T>()]; }` the repeat count
    /// constant `bar::<T>()` requires a substitution for `T`, if the substitution for `T` is still
    /// too generic for the constant to be evaluated then `Err(ErrorHandled::TooGeneric)` is
    /// returned.
    #[instrument(level = "debug", skip(self))]
    pub fn const_eval_resolve(
        self,
        param_env: ty::ParamEnv<'tcx>,
        ct: ty::Unevaluated<'tcx>,
        span: Option<Span>,
    ) -> EvalToConstValueResult<'tcx> {
        match ty::Instance::resolve_opt_const_arg(self, param_env, ct.def, ct.substs(self)) {
            Ok(Some(instance)) => {
                let cid = GlobalId { instance, promoted: ct.promoted };
                self.const_eval_global_id(param_env, cid, span)
            }
            Ok(None) => Err(ErrorHandled::TooGeneric),
            Err(error_reported) => Err(ErrorHandled::Reported(error_reported)),
        }
    }

    pub fn const_eval_instance(
        self,
        param_env: ty::ParamEnv<'tcx>,
        instance: ty::Instance<'tcx>,
        span: Option<Span>,
    ) -> EvalToConstValueResult<'tcx> {
        self.const_eval_global_id(param_env, GlobalId { instance, promoted: None }, span)
    }

    /// Evaluate a constant.
    pub fn const_eval_global_id(
        self,
        param_env: ty::ParamEnv<'tcx>,
        cid: GlobalId<'tcx>,
        span: Option<Span>,
    ) -> EvalToConstValueResult<'tcx> {
        let param_env = param_env.with_const();
        // Const-eval shouldn't depend on lifetimes at all, so we can erase them, which should
        // improve caching of queries.
        let inputs = self.erase_regions(param_env.and(cid));
        if let Some(span) = span {
            self.at(span).eval_to_const_value_raw(inputs)
        } else {
            self.eval_to_const_value_raw(inputs)
        }
    }

    /// Evaluate a static's initializer, returning the allocation of the initializer's memory.
    pub fn eval_static_initializer(
        self,
        def_id: DefId,
    ) -> Result<&'tcx mir::Allocation, ErrorHandled> {
        trace!("eval_static_initializer: Need to compute {:?}", def_id);
        assert!(self.is_static(def_id));
        let instance = ty::Instance::mono(self, def_id);
        let gid = GlobalId { instance, promoted: None };
        self.eval_to_allocation(gid, ty::ParamEnv::reveal_all())
    }

    /// Evaluate anything constant-like, returning the allocation of the final memory.
    fn eval_to_allocation(
        self,
        gid: GlobalId<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Result<&'tcx mir::Allocation, ErrorHandled> {
        let param_env = param_env.with_const();
        trace!("eval_to_allocation: Need to compute {:?}", gid);
        let raw_const = self.eval_to_allocation_raw(param_env.and(gid))?;
        Ok(self.global_alloc(raw_const.alloc_id).unwrap_memory())
    }
}
