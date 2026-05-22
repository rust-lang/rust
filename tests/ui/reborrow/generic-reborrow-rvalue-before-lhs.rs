#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

struct CustomMut<'a>(PhantomData<&'a mut ()>);
impl<'a> Reborrow for CustomMut<'a> {}

fn index<'a>(_value: CustomMut<'a>) -> usize {
    0
}

fn rvalue_reborrow_is_live_while_lhs_is_evaluated<'a>(
    value: CustomMut<'a>,
    mut slots: [CustomMut<'a>; 1],
) {
    // Assignment evaluates the RHS before the assignee place. Rvalue lowering must materialize
    // the reborrow result before `index(value)` is evaluated, without reborrowing from an
    // artificial source temporary.
    slots[index(value)] = value;
    //~^ ERROR cannot borrow `value` as mutable more than once at a time
    //~| ERROR `value` does not live long enough
}

fn main() {}
