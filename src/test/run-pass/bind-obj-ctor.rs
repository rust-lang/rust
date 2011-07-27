

fn main() {
    // Testcase for issue #59.

    obj simple(x: int, y: int) {
        fn sum() -> int { ret x + y; }
    }
    let obj0 = simple(1, 2);
    let ctor0 = bind simple(1, _);
    let ctor1 = bind simple(_, 2);
    let obj1 = ctor0(2);
    let obj2 = ctor1(1);
    assert (obj0.sum() == 3);
    assert (obj1.sum() == 3);
    assert (obj2.sum() == 3);
}