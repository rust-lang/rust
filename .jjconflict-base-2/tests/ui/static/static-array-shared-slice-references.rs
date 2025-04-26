//! regression test for <https://github.com/rust-lang/rust/issues/21891>
//@ build-pass
#![allow(dead_code)]

static FOO: [usize; 3] = [1, 2, 3];

static SLICE_1: &'static [usize] = &FOO;
static SLICE_2: &'static [usize] = &FOO;

fn main() {}
