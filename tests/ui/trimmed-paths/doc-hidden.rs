//@ edition: 2024
//@ aux-crate: helper=doc_hidden_helper.rs

// Test that `#[doc(hidden)]` items in other crates do not disqualify another
// item with the same name from path trimming in diagnostics.

// Declare several modules and types whose short names match those in the aux crate.
//
// Of these, only `ActuallyPub` and `ActuallyPubInPubMod` should be disqualified
// from path trimming, because the other names only collide with `#[doc(hidden)]`
// names.
mod local {
    pub(crate) struct ActuallyPub {}
    pub(crate) struct DocHidden {}

    pub(crate) mod pub_mod {
        pub(crate) struct ActuallyPubInPubMod {}
        pub(crate) struct DocHiddenInPubMod {}
    }

    pub(crate) mod hidden_mod {
        pub(crate) struct ActuallyPubInHiddenMod {}
        pub(crate) struct DocHiddenInHiddenMod {}
    }
}

fn main() {
    uses_local();
    uses_helper();
}

fn uses_local() {
    use local::{ActuallyPub, DocHidden};
    use local::pub_mod::{ActuallyPubInPubMod, DocHiddenInPubMod};
    use local::hidden_mod::{ActuallyPubInHiddenMod, DocHiddenInHiddenMod};

    let _: (
        //~^ NOTE expected due to this
        ActuallyPub,
        DocHidden,
        ActuallyPubInPubMod,
        DocHiddenInPubMod,
        ActuallyPubInHiddenMod,
        DocHiddenInHiddenMod,
    ) = 3u32;
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `(ActuallyPub, ..., ..., ..., ..., ...)`, found `u32`
    //~| NOTE expected tuple `(local::ActuallyPub, DocHidden, local::pub_mod::ActuallyPubInPubMod, DocHiddenInPubMod, ActuallyPubInHiddenMod, DocHiddenInHiddenMod)`
}

fn uses_helper() {
    use helper::{ActuallyPub, DocHidden};
    use helper::pub_mod::{ActuallyPubInPubMod, DocHiddenInPubMod};
    use helper::hidden_mod::{ActuallyPubInHiddenMod, DocHiddenInHiddenMod};

    let _: (
        //~^ NOTE expected due to this
        ActuallyPub,
        DocHidden,
        ActuallyPubInPubMod,
        DocHiddenInPubMod,
        ActuallyPubInHiddenMod,
        DocHiddenInHiddenMod,
    ) = 3u32;
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `(ActuallyPub, ..., ..., ..., ..., ...)`, found `u32`
    //~| NOTE expected tuple `(doc_hidden_helper::ActuallyPub, doc_hidden_helper::DocHidden, doc_hidden_helper::pub_mod::ActuallyPubInPubMod, doc_hidden_helper::pub_mod::DocHiddenInPubMod, doc_hidden_helper::hidden_mod::ActuallyPubInHiddenMod, doc_hidden_helper::hidden_mod::DocHiddenInHiddenMod)`
}
