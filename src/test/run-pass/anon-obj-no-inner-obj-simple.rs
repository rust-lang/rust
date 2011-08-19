use std;

fn main() {

    // Anonymous object that doesn't extend an existing one.
    let my_obj =
        obj () {
            fn foo() -> int { ret 2; }
            fn bar() -> int { ret 3; }
            fn baz() -> str { "hello!" }
        };

    assert (my_obj.foo() == 2);
    assert (my_obj.bar() == 3);
    assert (my_obj.baz() == "hello!");

    // Make sure the result is extendable.
    let my_ext_obj =
        obj () {
            fn foo() -> int { ret 3; }
            fn quux() -> str { ret self.baz(); }
            with
            my_obj
        };

    assert (my_ext_obj.foo() == 3);
    assert (my_ext_obj.bar() == 3);
    assert (my_ext_obj.baz() == "hello!");
    assert (my_ext_obj.quux() == "hello!");

    // And again.
    let my_ext_ext_obj =
        obj () {
            fn baz() -> str { "world!" }
            with
            my_ext_obj
        };

    assert (my_ext_ext_obj.foo() == 3);
    assert (my_ext_ext_obj.bar() == 3);
    assert (my_ext_ext_obj.baz() == "world!");
    assert (my_ext_ext_obj.quux() == "world!");
}
