

// xfail-stage0
// xfail-stage1
use std;

fn main() {
    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }
    auto my_a = a();
    // Extending an object with a new method

    auto my_b = anon obj;
    // Extending an object with a new field

    auto my_c = anon obj;
    // Should this be legal?

    auto my_d = anon obj;
}