//@ check-pass
#![allow(dead_code)]

struct Str([u8]);

#[derive(Clone)]
struct CharSplits<'a, Sep> {
    string: &'a Str,
    sep: Sep,
    allow_trailing_empty: bool,
    only_ascii: bool,
    finished: bool,
}

fn clone(s: &Str) -> &Str {
    Clone::clone(&s)
}

fn main() {}
