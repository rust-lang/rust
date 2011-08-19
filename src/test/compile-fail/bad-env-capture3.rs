// error-pattern: attempted dynamic environment-capture
obj foo(x: int) {
    fn mth() {
        fn bar() { log x; }
    }
}

fn main() { foo(2); }
