// This test ensures that the rustdoc `unused_footnote` is working as expected.

#![deny(rustdoc::unused_footnote_definition)]

//! Footnote referenced. [^2]
//!
//! [^1]: footnote defined
//! [^2]: footnote defined
//~^^ unused_footnote_definition
