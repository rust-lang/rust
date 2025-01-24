//@ edition: 2021

macro_rules! w {
    ($($tt:tt)*) => {};
}

w!('foo#lifetime);
//~^ ERROR prefix `'foo` is unknown

fn main() {}
