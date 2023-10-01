#![warn(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]
#![allow(clippy::let_unit_value, clippy::missing_safety_doc, clippy::needless_if)]

mod unsafe_items_invalid_comment {
    // SAFETY:
    const CONST: u32 = 0;
    //~^ ERROR: constant item has unnecessary safety comment
    // SAFETY:
    static STATIC: u32 = 0;
    //~^ ERROR: static item has unnecessary safety comment
    // SAFETY:
    struct Struct;
    //~^ ERROR: struct has unnecessary safety comment
    // SAFETY:
    enum Enum {}
    //~^ ERROR: enum has unnecessary safety comment
    // SAFETY:
    mod module {}
    //~^ ERROR: module has unnecessary safety comment
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
    //~^ ERROR: statement has unnecessary safety comment

    // SAFETY: unnecessary
    if num > 24 {}
    //~^ ERROR: statement has unnecessary safety comment

    // SAFETY: unnecessary
    24
    //~^ ERROR: expression has unnecessary safety comment
}

mod issue_10084 {
    unsafe fn bar() -> i32 {
        42
    }

    macro_rules! foo {
        () => {
            // SAFETY: This is necessary
            unsafe { bar() }
        };
    }

    fn main() {
        foo!();
    }
}

fn main() {}
