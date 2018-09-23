#![feature(tool_lints)]
#![warn(clippy::all, clippy::style, clippy::mem_replace_option_with_none)]

use std::mem;

fn main() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
}
