//@ revisions: old current
//@[old] edition: 2021
//@[current] edition: 2024
//@ aux-crate: reexport_source=reexport-source.rs
//@ aux-crate: reexport_preserving=reexport-preserving.rs
//@ aux-crate: reexport_old=reexport-old.rs
//@ aux-crate: reexport_current=reexport-current.rs
//@ check-pass

fn main() {
    // Crates with `edition_redirect` preserve redirects even when the re-export
    // itself is written in an older edition.
    #[cfg(old)]
    let _: reexport_preserving::Item = reexport_source::old();
    #[cfg(current)]
    let _: reexport_preserving::Item = reexport_source::current();

    // Other crates consume a redirect at the first `use`, fixing the re-export
    // to the item selected by that import's edition.
    let _: reexport_old::Item = reexport_source::old();
    let _: reexport_current::Item = reexport_source::current();

    // Redirecting a module changes path traversal at the first ordinary `use`,
    // but does not make the module's children independently redirected.
    let _: reexport_preserving::Child = reexport_source::current_child();
    let _: reexport_old::Child = reexport_source::old_child();
    let _: reexport_current::Child = reexport_source::current_child();
}
