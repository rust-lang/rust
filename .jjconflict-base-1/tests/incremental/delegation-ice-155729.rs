//@ revisions: bpass

#![feature(fn_delegation)]

pub mod to_reuse {
    pub fn bar() {}
}

mod a {
    use to_reuse;
    reuse to_reuse::bar;
}

fn main() {}
