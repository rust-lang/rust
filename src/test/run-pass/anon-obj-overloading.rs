//xfail-stage0
//xfail-stage1
//xfail-stage2
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

    // Right now, this just fails with "unknown method 'bar' of obj",
    // but that's the easier of our worries; that'll be fixed when
    // issue #539 is fixed.  The bigger problem will be when we do
    // 'fall through' to bar() on the original object -- then we have
    // to be sure that self refers to the extended object.
    assert (my_b.bar() == 3);
}
