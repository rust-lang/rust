//@ known-bug: #156288
#[warn(rust_2021_incompatible_closure_captures)]
const _: () = |b| move || b;
