use std;

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }

    let my_a = a();

    // Degenerate anonymous object: one that doesn't add any new
    // methods or fields.

    let my_d = obj () { with my_a };

    assert (my_d.foo() == 2);
    assert (my_d.bar() == 2);

}
