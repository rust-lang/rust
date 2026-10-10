//@ revisions: noopt opt opt_with_overflow_checks
//@[noopt]compile-flags: -C opt-level=0
//@[opt]compile-flags: -O
//@[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

//@ build-pass

#![allow(arithmetic_overflow)]

use std::mem;

const fn assert_static<T>(_: &'static T) {}

// Function calls in const on the "main path" (not inside conditionals)
// do get promoted.
const fn make_thing() -> i32 {
    42
}
const C: () = {
    assert_static(&make_thing());
    // Make sure this works even when there's other stuff (like function calls) above the relevant
    // call in the const initializer.
    assert_static(&make_thing());
};

fn main() {
    assert_static(&["a", "b", "c"]);
    assert_static(&["d", "e", "f"]);

    // make sure that this does not cause trouble despite overflowing
    assert_static(&(0u32 - 1));

    // div-by-non-0 (and also not MIN/-1) is okay
    assert_static(&(1/1));
    assert_static(&(0/1));
    assert_static(&(1/-1));
    assert_static(&(i32::MIN/1));
    assert_static(&(1%1));

    // in-bounds array access is okay
    assert_static(&([1, 2, 3][0] + 1));
    assert_static(&[[1, 2][1]]);

    // Top-level projections are not part of the promoted, so no error here.
    if false {
        #[allow(unconditional_panic)]
        assert_static(&[1, 2, 3][4]);
    }

    // More complicated case involving control flow and a `#[rustc_promotable]` function
    let decision = std::hint::black_box(true);
    let x: &'static usize = if decision { &mem::size_of::<usize>() } else { &0 };
}
