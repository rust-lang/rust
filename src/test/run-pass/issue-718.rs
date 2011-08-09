fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
    }

    let my_a = a();

    let my_b = obj () { with my_a };

    assert (my_b.foo() == 2);

    let my_c = obj () { with my_b };

    assert (my_c.foo() == 2);

    // ...One more for good measure.
    let my_d = obj () { with my_b };

    assert (my_d.foo() == 2);
}

