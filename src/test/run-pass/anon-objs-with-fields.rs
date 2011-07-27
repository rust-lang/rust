//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3
use std;

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }

    let my_a = a();

    // Extending an object with a new field.  Adding support for this
    // is issue #538.

    // Right now, this fails with "unresolved name: quux".
    let my_c =
        obj (quux: int = 3) {
            fn baz() -> int { ret quux + 4; }
            with
            my_a
        };

    assert (my_c.baz() == 7);

}