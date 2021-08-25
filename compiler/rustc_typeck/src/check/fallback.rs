use crate::check::FnCtxt;
use rustc_infer::infer::type_variable::Diverging;
use rustc_middle::ty::{self, Ty};

impl<'tcx> FnCtxt<'_, 'tcx> {
    /// Performs type inference fallback, returning true if any fallback
    /// occurs.
    pub(super) fn type_inference_fallback(&self) -> bool {
        // All type checking constraints were added, try to fallback unsolved variables.
        self.select_obligations_where_possible(false, |_| {});
        let mut fallback_has_occurred = false;

        // We do fallback in two passes, to try to generate
        // better error messages.
        // The first time, we do *not* replace opaque types.
        for ty in &self.unsolved_variables() {
            debug!("unsolved_variable = {:?}", ty);
            fallback_has_occurred |= self.fallback_if_possible(ty);
        }

        // We now see if we can make progress. This might
        // cause us to unify inference variables for opaque types,
        // since we may have unified some other type variables
        // during the first phase of fallback.
        // This means that we only replace inference variables with their underlying
        // opaque types as a last resort.
        //
        // In code like this:
        //
        // ```rust
        // type MyType = impl Copy;
        // fn produce() -> MyType { true }
        // fn bad_produce() -> MyType { panic!() }
        // ```
        //
        // we want to unify the opaque inference variable in `bad_produce`
        // with the diverging fallback for `panic!` (e.g. `()` or `!`).
        // This will produce a nice error message about conflicting concrete
        // types for `MyType`.
        //
        // If we had tried to fallback the opaque inference variable to `MyType`,
        // we will generate a confusing type-check error that does not explicitly
        // refer to opaque types.
        self.select_obligations_where_possible(fallback_has_occurred, |_| {});

        // We now run fallback again, but this time we allow it to replace
        // unconstrained opaque type variables, in addition to performing
        // other kinds of fallback.
        for ty in &self.unsolved_variables() {
            fallback_has_occurred |= self.fallback_opaque_type_vars(ty);
        }

        // See if we can make any more progress.
        self.select_obligations_where_possible(fallback_has_occurred, |_| {});

        fallback_has_occurred
    }

    // Tries to apply a fallback to `ty` if it is an unsolved variable.
    //
    // - Unconstrained ints are replaced with `i32`.
    //
    // - Unconstrained floats are replaced with with `f64`.
    //
    // - Non-numerics get replaced with `!` when `#![feature(never_type_fallback)]`
    //   is enabled. Otherwise, they are replaced with `()`.
    //
    // Fallback becomes very dubious if we have encountered type-checking errors.
    // In that case, fallback to Error.
    // The return value indicates whether fallback has occurred.
    fn fallback_if_possible(&self, ty: Ty<'tcx>) -> bool {
        // Careful: we do NOT shallow-resolve `ty`. We know that `ty`
        // is an unsolved variable, and we determine its fallback based
        // solely on how it was created, not what other type variables
        // it may have been unified with since then.
        //
        // The reason this matters is that other attempts at fallback may
        // (in principle) conflict with this fallback, and we wish to generate
        // a type error in that case. (However, this actually isn't true right now,
        // because we're only using the builtin fallback rules. This would be
        // true if we were using user-supplied fallbacks. But it's still useful
        // to write the code to detect bugs.)
        //
        // (Note though that if we have a general type variable `?T` that is then unified
        // with an integer type variable `?I` that ultimately never gets
        // resolved to a special integral type, `?T` is not considered unsolved,
        // but `?I` is. The same is true for float variables.)
        let fallback = match ty.kind() {
            _ if self.is_tainted_by_errors() => self.tcx.ty_error(),
            ty::Infer(ty::IntVar(_)) => self.tcx.types.i32,
            ty::Infer(ty::FloatVar(_)) => self.tcx.types.f64,
            _ => match self.type_var_diverges(ty) {
                Diverging::Diverges => self.tcx.mk_diverging_default(),
                Diverging::NotDiverging => return false,
            },
        };
        debug!("fallback_if_possible(ty={:?}): defaulting to `{:?}`", ty, fallback);

        let span = self
            .infcx
            .type_var_origin(ty)
            .map(|origin| origin.span)
            .unwrap_or(rustc_span::DUMMY_SP);
        self.demand_eqtype(span, ty, fallback);
        true
    }

    /// Second round of fallback: Unconstrained type variables
    /// created from the instantiation of an opaque
    /// type fall back to the opaque type itself. This is a
    /// somewhat incomplete attempt to manage "identity passthrough"
    /// for `impl Trait` types.
    ///
    /// For example, in this code:
    ///
    ///```
    /// type MyType = impl Copy;
    /// fn defining_use() -> MyType { true }
    /// fn other_use() -> MyType { defining_use() }
    /// ```
    ///
    /// `defining_use` will constrain the instantiated inference
    /// variable to `bool`, while `other_use` will constrain
    /// the instantiated inference variable to `MyType`.
    ///
    /// When we process opaque types during writeback, we
    /// will handle cases like `other_use`, and not count
    /// them as defining usages
    ///
    /// However, we also need to handle cases like this:
    ///
    /// ```rust
    /// pub type Foo = impl Copy;
    /// fn produce() -> Option<Foo> {
    ///     None
    ///  }
    ///  ```
    ///
    /// In the above snippet, the inference variable created by
    /// instantiating `Option<Foo>` will be completely unconstrained.
    /// We treat this as a non-defining use by making the inference
    /// variable fall back to the opaque type itself.
    fn fallback_opaque_type_vars(&self, ty: Ty<'tcx>) -> bool {
        let span = self
            .infcx
            .type_var_origin(ty)
            .map(|origin| origin.span)
            .unwrap_or(rustc_span::DUMMY_SP);
        let oty = self.inner.borrow().opaque_types_vars.get(ty).map(|v| *v);
        if let Some(opaque_ty) = oty {
            debug!(
                "fallback_opaque_type_vars(ty={:?}): falling back to opaque type {:?}",
                ty, opaque_ty
            );
            self.demand_eqtype(span, ty, opaque_ty);
            true
        } else {
            return false;
        }
    }
}
