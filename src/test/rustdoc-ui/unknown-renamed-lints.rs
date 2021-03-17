#![deny(unknown_lints)]
//~^ NOTE lint level is defined
#![deny(renamed_and_removed_lints)]
//~^ NOTE lint level is defined
#![deny(x)]
//~^ ERROR unknown lint
#![deny(rustdoc::x)]
//~^ ERROR unknown lint: `rustdoc::x`
#![deny(intra_doc_link_resolution_failure)]
//~^ ERROR renamed to `rustdoc::broken_intra_doc_links`

#![deny(non_autolinks)]
// FIXME: the old names for rustdoc lints should warn by default once `rustdoc::` makes it to the
// stable channel.

#![deny(rustdoc)]
//~^ ERROR removed: use `rustdoc::all` instead

// Explicitly don't try to handle this case, it was never valid
#![deny(rustdoc::intra_doc_link_resolution_failure)]
//~^ ERROR unknown lint
