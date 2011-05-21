// xfail-stage0
// xfail-stage1
use std;

fn main() {

    obj a() {
        fn foo() -> int {
            ret 2;
        }
    }

    auto my_a = a();

    // Extending an object with a new method
    auto my_b = obj { 
        fn bar() -> int { 
            ret 3;
        } 
        with my_a 
    };

    assert my_a.foo() == 2;

    // FIXME: these raise a runtime error
    //assert my_b.foo() == 2;
    assert my_b.bar() == 3;

}
