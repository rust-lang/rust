#![crate_type = "proc-macro"]
#![feature(proc_macro_tracked_path)]

extern crate proc_macro;
use proc_macro::*;

use std::str;

#[proc_macro]
pub fn access_tracked_paths(_: TokenStream) -> TokenStream {
    tracked_path::path("emojis.txt");

    // currently only valid utf-8 paths are supported
    if false {
        let invalid = [1_u8, 2,123, 254, 0, 0, 1, 1];
        let invalid: &str = unsafe {
            str::from_utf8_unchecked(&invalid[..])
        };
        tracked_path::path(invalid);
    }

    TokenStream::new()
}
