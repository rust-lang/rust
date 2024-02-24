#![warn(clippy::multiple_bound_locations)]

fn ty<F: std::fmt::Debug>(a: F)
//~^ ERROR: bound is defined in more than one place
where
    F: Sized,
{
}

fn lifetime<'a, 'b: 'a, 'c>(a: &'b str, b: &'a str, c: &'c str)
//~^ ERROR: bound is defined in more than one place
where
    'b: 'c,
{
}

fn ty_pred<F: Sized>()
//~^ ERROR: bound is defined in more than one place
where
    for<'a> F: Send + 'a,
{
}

struct B;

impl B {
    fn ty<F: std::fmt::Debug>(a: F)
    //~^ ERROR: bound is defined in more than one place
    where
        F: Sized,
    {
    }

    fn lifetime<'a, 'b: 'a, 'c>(a: &'b str, b: &'a str, c: &'c str)
    //~^ ERROR: bound is defined in more than one place
    where
        'b: 'c,
    {
    }

    fn ty_pred<F: Sized>()
    //~^ ERROR: bound is defined in more than one place
    where
        for<'a> F: Send + 'a,
    {
    }
}

struct C<F>(F);

impl<F> C<F> {
    fn foo(_f: F) -> Self
    where
        F: std::fmt::Display,
    {
        todo!()
    }
}

fn main() {}
