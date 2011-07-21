//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3
use std;

fn main() {

    obj a() {
        fn foo() -> int {
            ret 2;
        }
        fn bar() -> int {
            ret self.foo();
        }
    }

    auto my_a = a();

    // This compiles and shouldn't.  You should only be able to
    // overload a method with one of the same type.  Issue #703.
    auto my_b = obj() {
        fn foo() -> str {
            ret "hello";
        }
        with my_a
    };

    log_err my_b.foo();
}
