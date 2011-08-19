use std;

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }

    let my_a = a();

    let my_b =
        obj () {
            fn baz() -> int { ret self.foo(); }
            with
            my_a
        };

    assert (my_b.baz() == 2);

    let my_c =
        obj () {
            fn foo() -> int { ret 3; }
            fn baz() -> int { ret self.foo(); }
            with
            my_a
        };

    assert (my_c.baz() == 3);
    assert (my_c.bar() == 3);
}
