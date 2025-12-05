// Check that we correctly prevent users from making trait objects
// from traits with associated consts.

trait Bar {
    const X: usize;
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
    //~^ ERROR E0038
    t
}

fn main() {
}
