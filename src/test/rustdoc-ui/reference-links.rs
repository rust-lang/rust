// Test that errors point to the reference, not to the title text.
#![deny(broken_intra_doc_links)]
//! Links to [a] [link][a]
//!
//! [a]: std::process::Comman
//~^ ERROR unresolved
//~| ERROR unresolved
