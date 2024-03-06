//@ revisions: noopt opt opt_with_overflow_checks
//@[noopt]compile-flags: -C opt-level=0
//@[opt]compile-flags: -O
//@[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

//@ build-pass

const fn assert_static<T>(_: &'static T) {}

#[allow(unconditional_panic)]
const fn fail() -> i32 {
    1/0
}
const C: i32 = {
    // Promoted that fails to evaluate in dead code -- this must work
    // (for backwards compatibility reasons).
    if false {
        assert_static(&fail());
    }
    42
};

fn main() {
    assert_static(&["a", "b", "c"]);
    assert_static(&["d", "e", "f"]);
    assert_eq!(C, 42);

    // make sure that this does not cause trouble despite overflowing
    assert_static(&(0-1));

    // div-by-non-0 (and also not MIN/-1) is okay
    assert_static(&(1/1));
    assert_static(&(0/1));
    assert_static(&(1/-1));
    assert_static(&(i32::MIN/1));
    assert_static(&(1%1));

    // in-bounds array access is okay
    assert_static(&([1,2,3][0] + 1));
    assert_static(&[[1,2][1]]);

    // Top-level projections are not part of the promoted, so no error here.
    if false {
        #[allow(unconditional_panic)]
        assert_static(&[1,2,3][4]);
    }
}
