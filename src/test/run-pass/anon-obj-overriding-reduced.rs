// Reduced test case for issue #543.
fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
    }

    let my_a = a();

    let my_b =
        obj () {
            fn foo() -> int { ret 3; }
            with
            my_a
        };

    assert (my_b.foo() == 3);
}