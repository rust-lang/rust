//@ revisions: rpass1 rpass2

// Regression test for issue #92987
// Tests that we properly manage `DepNode`s during trait evaluation
// involing an auto-trait cycle.

#[cfg(rpass1)]
struct CycleOne(Box<CycleTwo>);

#[cfg(rpass2)]
enum CycleOne {
    Variant(Box<CycleTwo>)
}

struct CycleTwo(CycleOne);

fn assert_send<T: Send>() {}

fn bar() {
    assert_send::<CycleOne>();
    assert_send::<CycleTwo>();
}

fn main() {}
