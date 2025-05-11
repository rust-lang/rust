#![warn(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]
#![allow(clippy::let_unit_value, clippy::missing_safety_doc, clippy::needless_if)]

mod unsafe_items_invalid_comment {
    // SAFETY:
    const CONST: u32 = 0;
    //~^ unnecessary_safety_comment

    // SAFETY:
    static STATIC: u32 = 0;
    //~^ unnecessary_safety_comment

    // SAFETY:
    struct Struct;
    //~^ unnecessary_safety_comment

    // SAFETY:
    enum Enum {}
    //~^ unnecessary_safety_comment

    // SAFETY:
    mod module {}
    //~^ unnecessary_safety_comment
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
            //~^ unnecessary_safety_comment
        };
    }

    with_safety_comment!(i32);
}

fn unnecessary_on_stmt_and_expr() -> u32 {
    // SAFETY: unnecessary
    let num = 42;
    //~^ unnecessary_safety_comment

    // SAFETY: unnecessary
    if num > 24 {}
    //~^ unnecessary_safety_comment

    // SAFETY: unnecessary
    24
    //~^ unnecessary_safety_comment
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

mod issue_12048 {
    pub const X: u8 = 0;

    /// Returns a pointer to five.
    ///
    /// # Examples
    ///
    /// ```
    /// use foo::point_to_five;
    ///
    /// let five_pointer = point_to_five();
    /// // Safety: this pointer always points to a valid five.
    /// let five = unsafe { *five_pointer };
    /// assert_eq!(five, 5);
    /// ```
    pub fn point_to_five() -> *const u8 {
        static FIVE: u8 = 5;
        &FIVE
    }
}

fn main() {}
