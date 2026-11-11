// Check that we correctly prevent users from making trait objects
// from traits where `Self : Sized`.

trait Bar
    where Self : Sized
{
    fn bar<T>(&self, t: T);
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
    //~^ ERROR E0038
    t
}

fn main() {
}
