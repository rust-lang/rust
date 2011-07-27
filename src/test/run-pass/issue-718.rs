//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
    }

    let my_a = a();

    let my_b = obj () { with my_a };

    assert (my_b.foo() == 2);

    let my_c = obj () { with my_b };

    assert (my_c.foo() == 2);
}

