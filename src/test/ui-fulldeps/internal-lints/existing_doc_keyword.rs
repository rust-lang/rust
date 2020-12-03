// compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![feature(doc_keyword)]

#![crate_type = "lib"]

#[doc(keyword = "tadam")] //~ ERROR
mod tadam {}
