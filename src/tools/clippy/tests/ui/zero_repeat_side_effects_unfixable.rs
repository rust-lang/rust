//@no-rustfix
#![warn(clippy::zero_repeat_side_effects)]

fn issue_14998() {
    // unnameable types, don't suggest
    let _data = [|| 3i32; 0];
    //~^ zero_repeat_side_effects
}
