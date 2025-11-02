//@no-rustfix
#![warn(clippy::zero_repeat_side_effects)]
#![expect(clippy::diverging_sub_expression)]

fn issue_14998() {
    // unnameable types, don't suggest
    let _data = [|| 3i32; 0];
    //~^ zero_repeat_side_effects

    // unnameable type because `never_type` is not enabled, don't suggest
    let _data = [panic!(); 0];
    //~^ zero_repeat_side_effects
}
