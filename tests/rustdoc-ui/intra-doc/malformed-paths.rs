// This test ensures that (syntactically) malformed paths will not crash rustdoc.
#![deny(rustdoc::broken_intra_doc_links)]

// This is a regression test for <https://github.com/rust-lang/rust/issues/140026>.
//! [`Type::`]
//~^ ERROR

// This is a regression test for <https://github.com/rust-lang/rust/issues/147981>.
//! [`struct@Type@field`]
//~^ ERROR

//! [Type&content]
//~^ ERROR
//! [`Type::field%extra`]
//~^ ERROR

pub struct Type { pub field: () }
