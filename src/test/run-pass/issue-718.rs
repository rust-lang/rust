//xfail-stage0
//xfail-stage1
//xfail-stage2

fn main() {

    obj a() {
        fn foo() -> int {
            ret 2;
        }
    }

    auto my_a = a();

    auto my_b = obj() {
        with my_a
    };

    assert (my_b.foo() == 2);

    auto my_c = obj() {
        with my_b
    };

    assert (my_c.foo() == 2);
}

