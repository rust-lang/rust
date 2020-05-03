#![feature(non_ascii_idents)]
#![deny(confusable_idents)]
#![allow(uncommon_codepoints, non_upper_case_globals)]

const ｓ: usize = 42; //~ ERROR identifier pair considered confusable

fn main() {
    let s = "rust";
}
