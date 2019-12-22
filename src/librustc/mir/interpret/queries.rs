use super::{ConstEvalResult, ErrorHandled, GlobalId};

use crate::mir;
use crate::hir::def_id::DefId;
use crate::ty::{self, TyCtxt};
use crate::ty::subst::{InternalSubsts, SubstsRef};
use syntax_pos::Span;


impl<'tcx> TyCtxt<'tcx> {

    /// Evaluates a constant without providing any substitutions. This is useful to evaluate consts
    /// that can't take any generic arguments like statics, const items or enum discriminants. If a
    /// generic parameter is used within the constant `ErrorHandled::ToGeneric` will be returned.
    pub fn const_eval_poly(self, def_id: DefId) -> ConstEvalResult<'tcx> {
        // In some situations def_id will have substitutions within scope, but they aren't allowed
        // to be used. So we can't use `Instance::mono`, instead we feed unresolved substitutions
        // into `const_eval` which will return `ErrorHandled::ToGeneric` if any og them are
        // encountered.
        let substs = InternalSubsts::identity_for_item(self, def_id);
        let instance = ty::Instance::new(def_id, substs);
        let cid = GlobalId {
            instance,
            promoted: None,
        };
        let param_env = self.param_env(def_id);
        self.const_eval_validated(param_env.and(cid))
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
    pub fn const_eval_resolve(
        self,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        span: Option<Span>
    ) -> ConstEvalResult<'tcx> {
        let instance = ty::Instance::resolve(
            self,
            param_env,
            def_id,
            substs,
        );
        if let Some(instance) = instance {
            self.const_eval_instance(param_env, instance, span)
        } else {
            Err(ErrorHandled::TooGeneric)
        }
    }

    pub fn const_eval_instance(
        self,
        param_env: ty::ParamEnv<'tcx>,
        instance: ty::Instance<'tcx>,
        span: Option<Span>
    ) -> ConstEvalResult<'tcx> {
        let cid = GlobalId {
            instance,
            promoted: None,
        };
        if let Some(span) = span {
            self.at(span).const_eval_validated(param_env.and(cid))
        } else {
            self.const_eval_validated(param_env.and(cid))
        }
    }

    /// Evaluate a promoted constant.
    pub fn const_eval_promoted(
        self,
        instance: ty::Instance<'tcx>,
        promoted: mir::Promoted
    ) -> ConstEvalResult<'tcx> {
        let cid = GlobalId {
            instance,
            promoted: Some(promoted),
        };
        let param_env = ty::ParamEnv::reveal_all();
        self.const_eval_validated(param_env.and(cid))
    }
}
