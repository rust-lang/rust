use std;

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }

    let my_a = a();

    // An anonymous object that overloads the 'foo' method.
    let 

        my_b =
        obj () {
            fn foo() -> int { ret 3; }
            with
            my_a
        };

    assert (my_b.foo() == 3);
    assert (my_b.bar() == 3);

    let my_c =
        obj () {
            fn baz(x: int, y: int) -> int { ret x + y + self.foo(); }
            with
            my_b
        };

    let my_d =
        obj () {
            fn baz(x: int, y: int) -> int { ret x + y + self.foo(); }
            with
            my_a
        };

    assert (my_c.baz(1, 2) == 6);
    assert (my_d.baz(1, 2) == 5);
}