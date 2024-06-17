/// Lang items used by the new trait solver. This can be mapped to whatever internal
/// representation of `LangItem`s used in the underlying compiler implementation.
pub enum TraitSolverLangItem {
    Future,
    FutureOutput,
    AsyncFnKindHelper,
    AsyncFnKindUpvars,
}
