//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3

// Test case for issue #435.
obj foo() {
    fn add5(n: int) -> int {
        ret n + 5;
    }
}

fn add5(n: int) -> int {
    ret n + 5;
}

fn main() {
    let fiveplusseven = bind add5(7);
    assert add5(7) == 12;
    assert fiveplusseven() == 12;

    let my_foo = foo();
    let fiveplusseven_too = bind my_foo.add5(7);
    assert my_foo.add5(7) == 12;
    assert fiveplusseven_too() == 12;
}

