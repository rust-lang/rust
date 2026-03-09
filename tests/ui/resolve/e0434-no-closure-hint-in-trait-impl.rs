// Verify that E0434 inside a trait impl method does not suggest
// "use the closure form instead" since trait methods cannot be closures.

trait T {
    fn t() -> impl FnOnce() -> ();
}

fn f(x: String) {
    struct S;
    impl T for S {
        fn t() -> impl FnOnce() -> () {
            move || {let _ = x;}
            //~^ ERROR can't capture dynamic environment in a fn item
        }
    }
}

fn main() {}
