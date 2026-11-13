// regression test for https://github.com/rust-lang/rust/issues/141866
//@ check-pass
#![deny(rustdoc::broken_intra_doc_links)]

//! > [!NOTE]
//! > This should not cause any warnings
