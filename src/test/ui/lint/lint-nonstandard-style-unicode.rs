// build-pass (FIXME(62277): could be check-pass?)

#![allow(dead_code)]

#![forbid(non_camel_case_types)]
#![forbid(non_upper_case_globals)]
#![feature(non_ascii_idents)]

// Some scripts (e.g., hiragana) don't have a concept of
// upper/lowercase

struct ヒ;

static ラ: usize = 0;

pub fn main() {}
