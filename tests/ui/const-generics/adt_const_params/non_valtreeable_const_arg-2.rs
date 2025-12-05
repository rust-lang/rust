#![feature(generic_const_exprs, unsized_const_params)]
#![allow(incomplete_features)]

// Regression test for 128232

fn function() {}

struct Wrapper<const F: fn()>;
//~^ ERROR: using function pointers as const generic parameters is forbidden

impl Wrapper<{ bar() }> {
    //~^ ERROR: cannot find function `bar` in this scope
    fn call() {}
}

fn main() {
    Wrapper::<function>::call;
    //~^ ERROR: the function or associated item `call` exists for struct `Wrapper<function>`,
}
