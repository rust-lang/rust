//@ revisions: old current
//@[old] edition: 2021
//@[current] edition: 2024
//@ aux-crate: reexport_source=reexport-source.rs
//@ aux-crate: reexport_preserving=reexport-preserving.rs
//@ aux-crate: reexport_old=reexport-old.rs
//@ aux-crate: reexport_current=reexport-current.rs
//@ check-pass

fn main() {
    // A redirect is consumed by the first crate that imports it. The resulting
    // re-export is therefore fixed to that crate's edition for all downstream
    // users.
    let _: reexport_preserving::Item = reexport_source::old();
    let _: reexport_preserving::Child = reexport_source::old_child();

    let _: reexport_old::Item = reexport_source::old();
    let _: reexport_current::Item = reexport_source::current();

    // Redirecting a module changes path traversal at the first ordinary `use`,
    // but does not make the module's children independently redirected.
    let _: reexport_old::Child = reexport_source::old_child();
    let _: reexport_current::Child = reexport_source::current_child();
}
