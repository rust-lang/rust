//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test a where clause that uses a non-normalized projection type.


trait Int
{
    type T;

    fn dummy(&self) { }
}

trait NonZero
{
    fn non_zero(self) -> bool;
}

fn foo<I:Int<T=J>,J>(t: I) -> bool
    where <I as Int>::T : NonZero
    //    ^~~~~~~~~~~~~ canonical form is just J
{
    bar::<J>()
}

fn bar<NZ:NonZero>() -> bool { true }

fn main ()
{
}
