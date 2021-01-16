// build-fail
// normalize-stderr-test: ".nll/" -> "/"
#![feature(test)]

fn main() {
    rec(Empty, std::hint::black_box(false));
}

struct Empty;

impl Iterator for Empty {
    type Item = ();
    fn next<'a>(&'a mut self) -> core::option::Option<()> {
        None
    }
}

fn identity<T>(x: T) -> T {
    x
}

fn rec<T>(mut it: T, b: bool)
where
    T: Iterator,
{
    if b {
        T::count(it);
    } else {
        rec(identity(&mut it), b)
        //~^ ERROR reached the recursion limit while instantiating
    }
}
