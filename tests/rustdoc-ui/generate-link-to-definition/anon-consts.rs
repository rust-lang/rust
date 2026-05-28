// Ensure that we don't crash on anonymous constants.
// FIXME: Ideally, we would actually support linkifying type-dependent paths in
//        anon consts but for now that's disabled until we figure out what the
//        proper solution is implementation-wise.
// issue: <https://github.com/rust-lang/rust/issues/156418>
//@ check-pass

fn scope() {
    struct Hold<const N: usize>;
    let _ = X::<{ 0usize.saturating_add(1) }>;
}
