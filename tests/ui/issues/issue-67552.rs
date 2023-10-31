// build-fail
// compile-flags: -Copt-level=0
// normalize-stderr-test: ".nll/" -> "/"
// ignore-compare-mode-next-solver (hangs)

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
where
    T: Iterator,
{
    if () == () {
        T::count(it);
    } else {
        rec(identity(&mut it))
        //~^ ERROR reached the recursion limit while instantiating
    }
}
