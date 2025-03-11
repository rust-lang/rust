#![deny(rustdoc::broken_footnote)]

//! Footnote referenced [^1]. And [^2]. And [^bla].
//!
//! [^1]: footnote defined
//~^^^ ERROR: no footnote definition matching this footnote
//~| ERROR: no footnote definition matching this footnote
