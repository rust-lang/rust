// Tests that we probe receivers invariantly when using path-based method lookup.

struct B<T>(T);

impl B<fn(&'static ())> {
    fn method(self) {
        println!("hey");
    }
}

fn foo(y: B<fn(&'static ())>) {
    B::<for<'a> fn(&'a ())>::method(y);
    //~^ ERROR no associated function or constant named `method` found
}

fn main() {}
