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

    // Extending an object with a new field.  Adding support for this
    // is issue #538.
    auto my_c = obj(int quux = 3) {
        fn baz() -> int {
            ret quux + 4;
        }
        with my_a 
    };

    assert (my_c.baz() == 7);

}
