//@ revisions: without without_with_min_items with
//@[with] check-pass

#![feature(min_adt_const_params)]
#![cfg_attr(with, feature(gca_adts))]
// gca_min_const_items enables the gca! macro as well, which is vaguely interesting to test too
#![cfg_attr(without_with_min_items, feature(gca_min_const_items))]

use std::gca;
//[without]~^ ERROR use of unstable library feature `gca_min_const_items`

struct S<const A: [u32; 2]>;

fn main() {
    let _: S<gca!([1, 2])> = S::<gca!([1, 2])>;
    //[without_with_min_items]~^ ERROR complex const arguments must be placed inside of a `const` block
    //[without_with_min_items]~| ERROR complex const arguments must be placed inside of a `const` block
    //[without]~^^^ ERROR use of unstable library feature `gca_min_const_items`
    //[without]~| ERROR use of unstable library feature `gca_min_const_items`
    //[without]~| ERROR expected type, found `gca!()` constant
    //[without]~| ERROR expected type, found `gca!()` constant
    //[without]~| ERROR type provided when a constant was expected
    //[without]~| ERROR type provided when a constant was expected
}
