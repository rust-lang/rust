//@ check-pass
//@ compile-flags: --document-private-items

// This ensures that no ICE is triggered when rustdoc is run on this code.
// https://github.com/rust-lang/rust/issues/95633

mod stdlib {
    pub (crate) use std::i8;
}
