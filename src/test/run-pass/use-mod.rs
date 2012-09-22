use mod a::b;

mod a {
    #[legacy_exports];
    mod b {
        #[legacy_exports];
        fn f() {}
    }
}

fn main() {
    b::f();
}

