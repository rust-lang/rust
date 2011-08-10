//xfail-stage1
//xfail-stage2
//xfail-stage3
use std;

// Should we support this?  See issue #812.

fn main() {

    // Anonymous object that doesn't extend an existing one.
    let my_obj = obj () {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    };

    assert (my_obj.foo() == 2);
    assert (my_obj.bar() == 2);

}
