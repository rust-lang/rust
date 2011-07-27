use std;

fn main() {
    obj normal() {
        fn foo() -> int { ret 2; }
    }
    let my_normal_obj = normal();

    // Extending an object with a new method
    let my_anon_obj =
        obj () {
            fn bar() -> int { ret 3; }
            with
            my_normal_obj
        };

    assert (my_normal_obj.foo() == 2);
    assert (my_anon_obj.bar() == 3);

    let another_anon_obj =
        obj () {
            fn baz() -> int { ret 4; }
            with
            my_anon_obj
        };

    assert (another_anon_obj.baz() == 4);

}