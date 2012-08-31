use mod a::b;

mod a {
    mod b {
        fn f() {}
    }
}

fn main() {
    b::f();
}

