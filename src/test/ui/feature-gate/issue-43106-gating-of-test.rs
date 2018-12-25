// error-pattern: `main` function not found

// At time of authorship, crate-level #[test] attribute with no
// `--test` signals unconditional error complaining of missing main
// function (despite having one), similar to #[bench].
//
// (The non-crate level cases are in
// issue-43106-gating-of-builtin-attrs.rs.)

#![test                    = "4200"]

fn main() { }
