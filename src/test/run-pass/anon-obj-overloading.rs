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

    // An anonymous object that overloads the 'foo' method.  Adding
    // support for this is issue #543 (making this work in the
    // presence of self-calls is the tricky part).
    auto my_b = obj() { 
        fn foo() -> int {
            ret 3;
        }

        with my_a 
    };

    assert (my_b.foo() == 3);

    // The tricky part -- have to be sure to tie the knot in the right
    // place, so that bar() knows about the new foo().
    assert (my_b.bar() == 3);
}
