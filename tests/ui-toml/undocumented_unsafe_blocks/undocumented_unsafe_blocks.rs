#![deny(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::missing_safety_doc)]

fn main() {
    // Safety: A safety comment
    let _some_variable_with_a_very_long_name_to_break_the_line =
        unsafe { a_function_with_a_very_long_name_to_break_the_line() };

    // Safety: Another safety comment
    const _SOME_CONST_WITH_A_VERY_LONG_NAME_TO_BREAK_THE_LINE: u32 =
        unsafe { a_const_function_with_a_very_long_name_to_break_the_line() };

    // Safety: Yet another safety comment
    static _SOME_STATIC_WITH_A_VERY_LONG_NAME_TO_BREAK_THE_LINE: u32 =
        unsafe { a_const_function_with_a_very_long_name_to_break_the_line() };
}

pub unsafe fn a_function_with_a_very_long_name_to_break_the_line() -> u32 {
    1
}

pub const unsafe fn a_const_function_with_a_very_long_name_to_break_the_line() -> u32 {
    2
}
