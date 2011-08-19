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

    assert (my_a.foo() == 2);
    assert (my_a.bar() == 2);
    assert (my_b.foo() == 2);
    assert (my_b.baz() == 2);
    assert (my_b.bar() == 2);

}
