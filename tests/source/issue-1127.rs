// rustfmt-max_width:120
// rustfmt-wrap_match_arms: false
// rustfmt-match_block_trailing_comma: true

fn a_very_very_very_very_very_very_very_very_very_very_very_long_function_name() -> i32 {
    42
}

enum TestEnum {
    AVeryVeryLongEnumName,
    AnotherVeryLongEnumName,
    TheLastVeryLongEnumName,
}

fn main() {
    let var = TestEnum::AVeryVeryLongEnumName;
    let num = match var {
        TestEnum::AVeryVeryLongEnumName => a_very_very_very_very_very_very_very_very_very_very_very_long_function_name(),
        TestEnum::AnotherVeryLongEnumName => a_very_very_very_very_very_very_very_very_very_very_very_long_function_name(),
        TestEnum::TheLastVeryLongEnumName => a_very_very_very_very_very_very_very_very_very_very_very_long_function_name(),
    };
}

