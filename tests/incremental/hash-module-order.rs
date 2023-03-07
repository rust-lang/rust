// revisions: rpass1 rpass2
// compile-flags: -Z incremental-ignore-spans -Z query-dep-graph

// Tests that module hashing depends on the order of the items
// (since the order is exposed through `Mod.item_ids`).
// Changing the order of items (while keeping `Span`s the same)
// should still result in `hir_owner` being invalidated.
// Note that it's possible to keep the spans unchanged using
// a proc-macro (e.g. producing the module via `quote!`)
// but we use `-Z incremental-ignore-spans` for simplicity

#![feature(rustc_attrs)]

#[cfg(rpass1)]
#[rustc_clean(cfg="rpass1",except="hir_owner")]
mod foo {
    struct First;
    struct Second;
}

#[cfg(rpass2)]
#[rustc_clean(cfg="rpass2",except="hir_owner")]
mod foo {
    struct Second;
    struct First;
}

fn main() {}
