//@ check-pass
// Expectations on dead code covered by a dead parent's diagnostic must still be fulfilled.

#[cfg_attr(all(), expect(unused))]
struct Foo {
    #[expect(unused)]
    x: u32,
}

#[expect(dead_code)]
struct Direct {
    #[expect(dead_code)]
    x: u32,
}

#[expect(unused)]
enum DeadEnum {
    Variant {
        #[expect(unused)]
        x: u32,
    },
}

enum PartiallyDead {
    Used,
    #[expect(unused)]
    Dead {
        #[expect(unused)]
        x: u32,
    },
}

#[expect(unused)]
fn dead_outer() {
    #[expect(unused)]
    fn dead_inner() {}
}

#[expect(unused)]
trait DeadTrait {
    #[expect(unused)]
    fn dead_method(&self);
}

// `y` is read, so this expectation must stay unfulfilled.
struct Live {
    #[expect(unused)]
    //~^ WARNING this lint expectation is unfulfilled
    //~| NOTE `#[warn(unfulfilled_lint_expectations)]` on by default
    y: u32,
}

fn main() {
    let _ = PartiallyDead::Used;
    let _ = Live { y: 1 }.y;
}
