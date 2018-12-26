// error-pattern: can't capture dynamic environment in a fn item
fn foo(x: isize) {
    fn mth() {
        fn bar() { log(debug, x); }
    }
}

fn main() { foo(2); }
