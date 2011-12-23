// error-pattern: attempted dynamic environment-capture
obj foo(x: int) {
    fn mth() {
        fn bar() { log(debug, x); }
    }
}

fn main() { foo(2); }
