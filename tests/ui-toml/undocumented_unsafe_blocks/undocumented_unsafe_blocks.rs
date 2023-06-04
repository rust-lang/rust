#![deny(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::missing_safety_doc)]

fn main() {
    // Safety: A safety comment
    let _some_variable_with_a_very_long_name_to_break_the_line =
        unsafe { a_function_with_a_very_long_name_to_break_the_line() };
}

pub unsafe fn a_function_with_a_very_long_name_to_break_the_line() -> u32 {
    1
}
