#![warn(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]
#![allow(clippy::let_unit_value, clippy::missing_safety_doc)]

mod unsafe_items_invalid_comment {
    // SAFETY:
    const CONST: u32 = 0;
    // SAFETY:
    static STATIC: u32 = 0;
    // SAFETY:
    struct Struct;
    // SAFETY:
    enum Enum {}
    // SAFETY:
    mod module {}
}

mod unnecessary_from_macro {
    trait T {}

    macro_rules! no_safety_comment {
        ($t:ty) => {
            impl T for $t {}
        };
    }

    // FIXME: This is not caught
    // Safety: unnecessary
    no_safety_comment!(());

    macro_rules! with_safety_comment {
        ($t:ty) => {
            // Safety: unnecessary
            impl T for $t {}
        };
    }

    with_safety_comment!(i32);
}

fn unnecessary_on_stmt_and_expr() -> u32 {
    // SAFETY: unnecessary
    let num = 42;

    // SAFETY: unnecessary
    if num > 24 {}

    // SAFETY: unnecessary
    24
}

fn main() {}
