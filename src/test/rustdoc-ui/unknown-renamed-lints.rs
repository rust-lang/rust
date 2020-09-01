#![deny(unknown_lints)]
//~^ NOTE lint level is defined
#![deny(renamed_and_removed_lints)]
//~^ NOTE lint level is defined
#![deny(x)]
//~^ ERROR unknown lint
#![deny(intra_doc_link_resolution_failure)]
//~^ ERROR lint `intra_doc_link_resolution_failure` has been renamed
