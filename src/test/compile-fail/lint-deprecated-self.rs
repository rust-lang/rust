#[forbid(deprecated_self)]
mod a {
    trait T {
        fn f(); //~ ERROR this method form is deprecated
    }

    struct S {
        x: int
    }

    impl S : T {
        fn f() {    //~ ERROR this method form is deprecated
        }
    }
}

fn main() {
}


