// Check that we correctly prevent users from making trait objects
// form traits that make use of `Self` in an argument or return
// position, unless `where Self : Sized` is present..

trait Bar {
    fn bar(&self, x: &Self);
}

trait Baz {
    fn bar(&self) -> Self;
}

trait Quux {
    fn get(&self, s: &Self) -> Self where Self : Sized;
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
        //~^ ERROR E0038
    loop { }
}

fn make_baz<T:Baz>(t: &T) -> &dyn Baz {
        //~^ ERROR E0038
    t
}

fn make_quux<T:Quux>(t: &T) -> &dyn Quux {
    t
}

fn make_quux_explicit<T:Quux>(t: &T) -> &dyn Quux {
    t as &dyn Quux
}

fn main() {
}
