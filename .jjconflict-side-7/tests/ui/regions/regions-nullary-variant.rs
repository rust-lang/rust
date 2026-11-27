//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


enum roption<'a> {
    a, b(&'a usize)
}

fn mk(cond: bool, ptr: &usize) -> roption<'_> {
    if cond {roption::a} else {roption::b(ptr)}
}

pub fn main() {}
