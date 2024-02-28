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
    //[quiet]~^ ERROR trait `for<'b> Foo` is not implemented for `fn(&'b ())`
    //[verbose]~^^ ERROR the trait `for<Region
}

fn main() {}
