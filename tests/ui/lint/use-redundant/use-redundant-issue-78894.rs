//@ check-pass
//@ edition:2018

#![warn(unused_imports)]

mod foo {
    macro_rules! foo1 {
        () => ();
    }

    pub(crate) use foo1;
}

fn main ()
{
    bar!();

    macro_rules! bar {
        () => ();
    }

    use bar;

    mod m {
        bar1!();

        macro_rules! bar1 {
            () => ();
        }

        use bar1;
    }

    {
        foo::foo1!();
    }

    {
        use foo::foo1;
        foo1!();
    }

    {
        use foo::foo1; //~ WARNING unused import: `foo::foo1`
        foo::foo1!();
    }

}
