#![deny(unknown_lints)]
//~^ NOTE lint level is defined
#![deny(renamed_and_removed_lints)]
//~^ NOTE lint level is defined
#![deny(x)]
//~^ ERROR unknown lint
#![deny(rustdoc::x)]
//~^ ERROR unknown lint: `rustdoc::x`
#![deny(intra_doc_link_resolution_failure)]
//~^ ERROR has been renamed

// This would ideally say 'renamed to rustdoc::non_autolinks', but this is close enough.
#![deny(non_autolinks)]
//~^ ERROR has been removed: use `rustdoc::non_autolinks` instead [renamed_and_removed_lints]

// This doesn't give you the right code directly, but at least points you on the
// right path.
#![deny(rustdoc::intra_doc_link_resolution_failure)]
//~^ ERROR unknown lint
