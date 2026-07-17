//@ check-pass

// Regression test for issue #159323
// Field-level lint attributes must control `non_snake_case` diagnostics for that field.

#![deny(non_snake_case, unfulfilled_lint_expectations)]

pub struct Struct {
    #[expect(non_snake_case)]
    pub expected_Field: bool,

    #[allow(non_snake_case)]
    pub allowed_Field: bool,
}

#[expect(non_snake_case)]
pub struct ParentExpectation {
    pub expected_Field: bool,
}

pub enum Enum {
    Variant {
        #[expect(non_snake_case)]
        expected_Field: bool,
    },
}

pub union Union {
    #[expect(non_snake_case)]
    pub expected_Field: bool,
}

fn main() {}
