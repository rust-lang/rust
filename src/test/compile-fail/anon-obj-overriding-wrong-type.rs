//error-pattern: with one of a different type
use std;

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }

    let my_a = a();

    // Attempting to override a method with one of a different type.
    let my_b =
        obj () {
            fn foo() -> str { ret "hello"; }
            with
            my_a
        };

    log_err my_b.foo();
}
