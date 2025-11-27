#![feature(proc_macro_track_path)]
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn access_tracked_paths(_: TokenStream) -> TokenStream {
    tracked::path("emojis.txt");
    TokenStream::new()
}
