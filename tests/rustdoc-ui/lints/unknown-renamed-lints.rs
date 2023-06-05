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
//~^ ERROR renamed to `rustdoc::bare_urls`
#![deny(rustdoc::non_autolinks)]
//~^ ERROR renamed to `rustdoc::bare_urls`

#![deny(private_doc_tests)]
//~^ ERROR renamed to `rustdoc::private_doc_tests`

#![deny(rustdoc)]
//~^ ERROR removed: use `rustdoc::all` instead

// Explicitly don't try to handle this case, it was never valid
#![deny(rustdoc::intra_doc_link_resolution_failure)]
//~^ ERROR unknown lint
