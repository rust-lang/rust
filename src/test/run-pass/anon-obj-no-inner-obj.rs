//xfail-test
use std;

// This is failing not because it's an anonymous object from nothing
// -- that park seems to work fine -- but, rather, because methods
// that are added to an object at the same time can't refer to each
// other (issue #822).

fn main() {

    // Anonymous object that doesn't extend an existing one.
    let my_obj = obj () {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    };

    assert (my_obj.foo() == 2);
    assert (my_obj.bar() == 2);

}
