// https://github.com/rust-lang/rust/issues/148070
#![no_main]
use stat; //~ ERROR unresolved import `stat`
use str; //~ ERROR unresolved import `str`
use sync; //~ ERROR unresolved import `sync`
