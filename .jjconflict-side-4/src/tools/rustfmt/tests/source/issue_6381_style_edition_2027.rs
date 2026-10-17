// rustfmt-style_edition: 2027

fn my_function_no_wrap_at_exactly_100_characters_wide(
    my_long_impl_trait_parameter: impl Into<LongTypeNameThatMakesThisWholeLineExactly100Chars_____>,
) {
}

fn my_function_wraps_at_at_exactly_101_characters_wide(
    my_long_impl_trait_parameter: impl Into<LongTypeNameThatMakesThisWholeLineExactly101Chars______>,
) {
}
