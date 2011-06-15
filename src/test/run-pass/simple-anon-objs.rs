

// xfail-stage0
use std;

fn main() {
    obj a() {
        fn foo() -> int { ret 2; }
    }
    auto my_a = a();
    // Extending an object with a new method

    auto my_b = obj { 
        fn bar() -> int { 
            ret 3;
        }
        with my_a 
    };

    assert (my_a.foo() == 2);
    assert (my_b.bar() == 3);

    auto my_c = obj {
        fn baz() -> int {
            ret 4;
        }
        with my_b
    };

    assert (my_c.baz() == 4);

}

