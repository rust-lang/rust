// Tests the different rules for `fn` forms requiring the presence or lack of a body.

fn main() {
    fn f1(); //~ ERROR free function without a body
    fn f2() {} // OK.

    trait X {
        fn f1(); // OK.
        fn f2() {} // OK.
    }

    struct Y;
    impl X for Y {
        fn f1(); //~ ERROR associated function in `impl` without body
        fn f2() {} // OK.
    }

    impl Y {
        fn f3(); //~ ERROR associated function in `impl` without body
        fn f4() {} // OK.
    }

    extern "C" {
        fn f5(); // OK.
        fn f6() {} //~ ERROR incorrect function inside `extern` block
    }
}
