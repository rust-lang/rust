// build-fail

fn main() {
    rec(Empty);
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

fn rec<T>(mut it: T)
//~^ ERROR reached the recursion limit while instantiating
where
    T: Iterator,
{
    if () == () {
        T::count(it);
    } else {
        rec(identity(&mut it))
    }
}
