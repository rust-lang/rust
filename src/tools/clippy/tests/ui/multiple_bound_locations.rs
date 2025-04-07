#![warn(clippy::multiple_bound_locations)]

fn ty<F: std::fmt::Debug>(a: F)
//~^ multiple_bound_locations
where
    F: Sized,
{
}

fn lifetime<'a, 'b: 'a, 'c>(a: &'b str, b: &'a str, c: &'c str)
//~^ multiple_bound_locations
where
    'b: 'c,
{
}

fn ty_pred<F: Sized>()
//~^ multiple_bound_locations
where
    for<'a> F: Send + 'a,
{
}

struct B;

impl B {
    fn ty<F: std::fmt::Debug>(a: F)
    //~^ multiple_bound_locations
    where
        F: Sized,
    {
    }

    fn lifetime<'a, 'b: 'a, 'c>(a: &'b str, b: &'a str, c: &'c str)
    //~^ multiple_bound_locations
    where
        'b: 'c,
    {
    }

    fn ty_pred<F: Sized>()
    //~^ multiple_bound_locations
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
