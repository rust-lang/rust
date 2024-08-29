#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

struct DataWrapper<'static> {
    //~^ ERROR invalid lifetime parameter name: `'static`
    data: &'a [u8; Self::SIZE],
    //~^ ERROR use of undeclared lifetime name `'a`
}

impl DataWrapper<'a> {
    //~^ ERROR undeclared lifetime
    const SIZE: usize = 14;
}

fn main(){}
