#![crate_type = "lib"]

#![deny(unknown_lints)]
#![deny(renamed_and_removed_lints)]
//~^ NOTE lint level is defined

// both allowed, since the compiler doesn't yet know what rustdoc lints are valid
#![deny(rustdoc::x)]
#![deny(rustdoc::intra_doc_link_resolution_failure)]

#![deny(intra_doc_link_resolution_failure)]
//~^ ERROR removed: use `rustdoc::broken_intra_doc_links`
#![deny(non_autolinks)]
// FIXME: the old names for rustdoc lints should warn by default once `rustdoc::` makes it to the
// stable channel.
