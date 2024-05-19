use std::cell::Cell;

fn check<'a, 'b>(x: Cell<&'a ()>, y: Cell<&'b ()>)
where
    'a: 'b,
{
}

fn test<'a, 'b>(x: Cell<&'a ()>, y: Cell<&'b ()>) {
    let f = check;
    //~^ ERROR lifetime may not live long enough
    f(x, y);
}

fn main() {}
