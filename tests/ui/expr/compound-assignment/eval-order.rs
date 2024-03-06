// Test evaluation order of operands of the compound assignment operators

//@ run-pass

use std::ops::AddAssign;

enum Side {
    Lhs,
    Rhs,
}

// In the following tests, we place our value into a wrapper type so that we
// can do an element access as the outer place expression. If we just had the
// block expression, it'd be a value expression and not compile.
struct Wrapper<T>(T);

// Evaluation order for `a op= b` where typeof(a) and typeof(b) are primitives
// is first `b` then `a`.
fn primitive_compound() {
    let mut side_order = vec![];
    let mut int = Wrapper(0);

    {
        side_order.push(Side::Lhs);
        int
    }.0 += {
        side_order.push(Side::Rhs);
        0
    };

    assert!(matches!(side_order[..], [Side::Rhs, Side::Lhs]));
}

// Evaluation order for `a op=b` otherwise is first `a` then `b`.
fn generic_compound<T: AddAssign<T> + Default>() {
    let mut side_order = vec![];
    let mut add_assignable: Wrapper<T> = Wrapper(Default::default());

    {
        side_order.push(Side::Lhs);
        add_assignable
    }.0 += {
        side_order.push(Side::Rhs);
        Default::default()
    };

    assert!(matches!(side_order[..], [Side::Lhs, Side::Rhs]));
}

fn custom_compound() {
    struct Custom;

    impl AddAssign<()> for Custom {
        fn add_assign(&mut self, _: ()) {
            // this block purposely left blank
        }
    }

    let mut side_order = vec![];
    let mut custom = Wrapper(Custom);

    {
        side_order.push(Side::Lhs);
        custom
    }.0 += {
        side_order.push(Side::Rhs);
    };

    assert!(matches!(side_order[..], [Side::Lhs, Side::Rhs]));
}

fn main() {
    primitive_compound();
    generic_compound::<i32>();
    custom_compound();
}
