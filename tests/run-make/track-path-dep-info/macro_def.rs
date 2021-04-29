#![feature(proc_macro_tracked_env,proc_macro_tracked_path)]
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

use std::str;

#[proc_macro]
pub fn access_tracked_paths(_: TokenStream) -> TokenStream {
    assert!(tracked::path("emojis.txt").is_ok());

    // currently only valid utf-8 paths are supported
    let invalid = [1_u8, 2,123, 254, 0, 0, 1, 1];
    let invalid: &str = unsafe {
        str::from_utf8_unchecked(&invalid[..])
    };
    assert!(tracked::path(invalid).is_err());

    TokenStream::new()
}
