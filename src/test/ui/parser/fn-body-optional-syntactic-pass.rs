// Ensures that all `fn` forms having or lacking a body are syntactically valid.

// check-pass

fn main() {}

#[cfg(FALSE)]
fn syntax() {
    fn f();
    fn f() {}

    trait X {
        fn f();
        fn f() {}
    }

    impl X for Y {
        fn f();
        fn f() {}
    }

    impl Y {
        fn f();
        fn f() {}
    }

    extern "C" {
        fn f();
        fn f();
    }
}
