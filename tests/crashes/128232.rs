//@ known-bug: #128232

#![feature(generic_const_exprs, unsized_const_params)]

fn function() {}

struct Wrapper<const F: fn()>;

impl Wrapper<{ bar() }> {
    fn call() {}
}

fn main() {
    Wrapper::<function>::call;
}
