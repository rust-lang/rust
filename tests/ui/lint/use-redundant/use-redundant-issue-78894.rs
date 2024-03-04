//@ check-pass
//@ edition:2018
#![warn(unused_imports)]

fn main ()
{
    bar!();

    macro_rules! bar {
        () => ();
    }

    use bar;
}
