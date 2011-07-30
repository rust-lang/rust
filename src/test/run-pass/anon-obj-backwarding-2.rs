//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3
use std;

fn main() {

    obj a() {
        fn foo() -> int { ret 2; }
        fn bar() -> int { ret self.foo(); }
    }

    let my_a = a();

    let my_b = obj () {
        fn baz() -> int { ret self.foo(); }
        with my_a
    };

    // These should all be 2.
    log_err my_a.foo();
    log_err my_a.bar();
    log_err my_b.foo();

    // This works fine.  It sends us to foo on my_b, which forwards to
    // foo on my_a.
    log_err my_b.baz();

    // Currently segfaults.  It forwards us to bar on my_a, which
    // backwards us to foo on my_b, which forwards us to foo on my_a
    // -- or, at least, that's how it should work.
    log_err my_b.bar();

}
