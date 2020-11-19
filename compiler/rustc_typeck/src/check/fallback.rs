use crate::check::FallbackMode;
use crate::check::FnCtxt;

impl<'tcx> FnCtxt<'_, 'tcx> {
    pub(super) fn type_inference_fallback(&self) {
        // All type checking constraints were added, try to fallback unsolved variables.
        self.select_obligations_where_possible(false, |_| {});
        let mut fallback_has_occurred = false;

        // We do fallback in two passes, to try to generate
        // better error messages.
        // The first time, we do *not* replace opaque types.
        for ty in &self.unsolved_variables() {
            debug!("unsolved_variable = {:?}", ty);
            fallback_has_occurred |= self.fallback_if_possible(ty, FallbackMode::NoOpaque);
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
            fallback_has_occurred |= self.fallback_if_possible(ty, FallbackMode::All);
        }

        // See if we can make any more progress.
        self.select_obligations_where_possible(fallback_has_occurred, |_| {});
    }
}
