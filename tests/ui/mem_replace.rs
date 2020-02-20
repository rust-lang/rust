// run-rustfix
#![allow(unused_imports)]
#![warn(
    clippy::all,
    clippy::style,
    clippy::mem_replace_option_with_none,
    clippy::mem_replace_with_default
)]

use std::mem;

fn replace_option_with_none() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
}

fn replace_with_default() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());
    let s = &mut String::from("foo");
    let _ = std::mem::replace(s, String::default());
    let _ = std::mem::replace(s, Default::default());
}

fn main() {
    replace_option_with_none();
    replace_with_default();
}
