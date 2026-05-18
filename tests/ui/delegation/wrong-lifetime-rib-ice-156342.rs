#![feature(fn_delegation)]
#![feature(type_info)]

use std::mem::type_info::Trait;

impl Trait {
//~^ ERROR: cannot define inherent `impl` for a type outside of the crate where the type is defined
    reuse None::<&()>;
    //~^ ERROR: expected function, found unit variant `None`
}

fn foo<T>() {}

reuse foo::<&&&&&&&&&&()> as foo1;
reuse foo::<&std::borrow::Cow<'_, &()>> as foo2;

fn main() {}
