// Test that miri respects the `target.min_global_align` value for the target.
//
// The only way to observe its effect currently is to test for the alignment of statics with a
// natural alignment of 1 on s390x. On that target, the `min_global_align` is 2 bytes.

fn main() {
    let min_align = if cfg!(target_arch = "s390x") { 2 } else { 1 };

    macro_rules! check {
        ($x:ident, $v:expr) => {
            static $x: bool = $v;
            assert!(core::ptr::from_ref(&$x).addr().is_multiple_of(min_align));
        };
    }

    check!(T0, true);
    check!(T1, true);
    check!(T2, true);
    check!(T3, true);
    check!(T4, true);
    check!(T5, true);
    check!(T6, true);
    check!(T7, true);
    check!(T8, true);
    check!(T9, true);

    check!(F0, false);
    check!(F1, false);
    check!(F2, false);
    check!(F3, false);
    check!(F4, false);
    check!(F5, false);
    check!(F6, false);
    check!(F7, false);
    check!(F8, false);
    check!(F9, false);
}
