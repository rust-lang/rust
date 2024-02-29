//@ revisions: quiet verbose
//@ [verbose]compile-flags: -Zverbose-internals

#![allow(unused_parens)]

trait Foo {
    type Assoc;
}

fn called()
where
    for<'b> fn(&'b ()): Foo,
{
}

fn caller()
where
    (for<'a> fn(&'a ())): Foo,
{
    called()
    //[quiet]~^ ERROR trait `Foo` is not implemented for `fn(&())`
    //[verbose]~^^ ERROR the trait `Foo` is not implemented for `fn(&ReErased ())`
}

fn main() {}
