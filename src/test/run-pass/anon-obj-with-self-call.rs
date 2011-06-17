//xfail-stage0
//xfail-stage1
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

    // Extending an object with a new method that contains a simple
    // self-call.  Adding support for this is issue #540.
    auto my_b = obj { 
        fn baz() -> int { 
            ret self.foo(); 
        } 
        with my_a 
    };

    assert (my_b.baz() == 2);

}
