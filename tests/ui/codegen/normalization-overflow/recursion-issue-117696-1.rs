//@ build-fail
fn main() {
    let mut it = (Empty);
    rec(&mut it);
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
        //~^ ERROR: reached the recursion limit while instantiating
    } else {
        rec(identity(&mut it))
    }
}
