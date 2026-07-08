//@ compile-flags: -Znext-solver
//@ check-pass

fn mk_vec() -> Vec<Vec<&'static str>> {
    loop {}
}

fn parse_feature(feature: &str) -> impl Iterator<Item = &str> {
    std::iter::once(feature)
}

fn main() {
    // In `codegen_select_candidate`, we try to find an impl for `FlatMap::into_iter`
    // We try to prove `FlatMap<...>: IntoIterator`.
    // The blanket impl requires `FlatMap<...>: Iterator`.
    // The `Iterator` impl of `FlatMap` has nested goals:
    // `Projection(Fn::Output<parse_feature, (?assoc,)>, iter::Once`.
    // We normalize this goal and set rerun condition to `AnyOpaqueHasInferAsHidden`
    // with reason `SelfTyInfer` because we instantiated impls with infers when
    // assembling candidates for intermediate goals.
    // Then we evaluate the normalized goal:
    // `Projection(Fn::Output<parse_feature, (&str,)>, iter::Once`.
    // The alias term is normalized to rigid alias `impl Iterator<Item = &str>` as
    // we're in `TypingMode::ErasedNotCoherence`.
    // But the expected term is revealed `iter::Once` thus relating failed.
    // The goal fails with rerun condition `OpaqueInStorage(parse_feature::opaque)`.
    // This goal should be rerun in `TypingMode::Codegen` mode, but
    // `AnyOpaqueHasInferAsHidden + OpaqueInStorage = OpaqueInStorageOrAnyOpaqueHasInferAsHidden`
    // which didn't trigger rerun in `TypingMode::Codegen` mode previously.
    mk_vec()
        .into_iter()
        .flatten()
        .flat_map(parse_feature).into_iter();
}
