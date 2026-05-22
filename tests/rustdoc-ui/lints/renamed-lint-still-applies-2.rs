// compile-args: --crate-type lib

// This file does not emit the rename warnings
// due to compilation aborting before we emit delayed lints

#![deny(broken_intra_doc_links)]
//! [x]
//~^ ERROR unresolved link

#![deny(rustdoc::non_autolinks)]
//! http://example.com
//~^ ERROR not a hyperlink
