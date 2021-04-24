// compile-args: --crate-type lib
#![deny(broken_intra_doc_links)]
// FIXME: the old names for rustdoc lints should warn by default once `rustdoc::` makes it to the
// stable channel.
//! [x]
//~^ ERROR unresolved link

#![deny(rustdoc::non_autolinks)]
//~^ WARNING renamed to `rustdoc::bare_urls`
//! http://example.com
//~^ ERROR not a hyperlink
