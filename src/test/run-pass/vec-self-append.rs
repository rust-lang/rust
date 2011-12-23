use std;
import vec;

fn test_heap_to_heap() {
    // a spills onto the heap
    let a = [0, 1, 2, 3, 4];
    a += a;
    assert (vec::len(a) == 10u);
    assert (a[0] == 0);
    assert (a[1] == 1);
    assert (a[2] == 2);
    assert (a[3] == 3);
    assert (a[4] == 4);
    assert (a[5] == 0);
    assert (a[6] == 1);
    assert (a[7] == 2);
    assert (a[8] == 3);
    assert (a[9] == 4);
}

fn test_stack_to_heap() {
    // a is entirely on the stack
    let a = [0, 1, 2];
    // a spills to the heap
    a += a;
    assert (vec::len(a) == 6u);
    assert (a[0] == 0);
    assert (a[1] == 1);
    assert (a[2] == 2);
    assert (a[3] == 0);
    assert (a[4] == 1);
    assert (a[5] == 2);
}

fn test_loop() {
    // Make sure we properly handle repeated self-appends.
    let a: [int] = [0];
    let i = 20;
    let expected_len = 1u;
    while i > 0 {
        log(error, vec::len(a));
        assert (vec::len(a) == expected_len);
        a += a;
        i -= 1;
        expected_len *= 2u;
    }
}

fn main() { test_heap_to_heap(); test_stack_to_heap(); test_loop(); }
