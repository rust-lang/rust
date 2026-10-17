trait T {
    fn t() -> impl FnOnce() -> ();
}

fn f(x: String) {
    struct S;
    impl T for S {
        fn t() -> impl FnOnce() -> () {
            move || {
                let _ = x; //~ ERROR can't capture dynamic environment in a fn item
            }
        }
    }

    impl S {
        fn foo() {
            let bar = 42;

            fn inner() {
                let _ = bar; //~ ERROR can't capture dynamic environment in a fn item

                let _ = x; //~ ERROR can't capture dynamic environment in a fn item
            }
        }
    }
}

fn main() {}
