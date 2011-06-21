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

    // Degenerate anonymous object: one that doesn't add any new
    // methods or fields.  Adding support for this is issue #539.
    // (Making this work will also ensure that calls to anonymous
    // objects "fall through" appropriately.)

    auto my_d = obj() { with my_a };

    // Right now, this fails with "unknown method 'foo' of obj".
    assert (my_d.foo() == 2);
    assert (my_d.bar() == 2);

}
