//! This file tests the `#[expect]` attribute implementation for tool lints. The same
//! file is used to test clippy and rustdoc. Any changes to this file should be synced
//! to the other test files as well.
//!
//! Expectations:
//! * rustc: only rustc lint expectations are emitted
//! * clippy: rustc and Clippy's expectations are emitted
//! * rustdoc: only rustdoc lint expectations are emitted
//!
//! This test can't cover every lint from Clippy, rustdoc and potentially other
//! tools that will be developed. This therefore only tests a small subset of lints
#![expect(rustdoc::missing_crate_level_docs)]
#![allow(clippy::needless_if)]

mod rustc_ok {
    //! See <https://doc.rust-lang.org/rustc/lints/index.html>

    #[expect(dead_code)]
    pub fn rustc_lints() {
        let x = 42.0;

        #[expect(invalid_nan_comparisons)]
        let _b = x == f32::NAN;
    }
}

mod rustc_warn {
    //! See <https://doc.rust-lang.org/rustc/lints/index.html>

    #[expect(dead_code)]
    //~^ ERROR: this lint expectation is unfulfilled
    //~| NOTE: `-D unfulfilled-lint-expectations` implied by `-D warnings`
    //~| HELP: to override `-D warnings` add `#[allow(unfulfilled_lint_expectations)]`
    pub fn rustc_lints() {
        let x = 42;

        #[expect(invalid_nan_comparisons)]
        //~^ ERROR: this lint expectation is unfulfilled
        //~| NOTE: duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
        //~| ERROR: this lint expectation is unfulfilled
        let _b = x == 5;
    }
}

pub mod rustdoc_ok {
    //! See <https://doc.rust-lang.org/rustdoc/lints.html>

    #[expect(rustdoc::broken_intra_doc_links)]
    /// I want to link to [`Nonexistent`] but it doesn't exist!
    pub fn foo() {}

    #[expect(rustdoc::invalid_html_tags)]
    /// <h1>
    pub fn bar() {}

    #[expect(rustdoc::bare_urls)]
    /// http://example.org
    pub fn baz() {}
}

pub mod rustdoc_warn {
    //! See <https://doc.rust-lang.org/rustdoc/lints.html>

    #[expect(rustdoc::broken_intra_doc_links)]
    /// I want to link to [`bar`] but it doesn't exist!
    pub fn foo() {}

    #[expect(rustdoc::invalid_html_tags)]
    /// <h1></h1>
    pub fn bar() {}

    #[expect(rustdoc::bare_urls)]
    /// <http://example.org>
    pub fn baz() {}
}

mod clippy_ok {
    //! See <https://rust-lang.github.io/rust-clippy/master/index.html>

    #[expect(clippy::almost_swapped)]
    fn foo() {
        let mut a = 0;
        let mut b = 9;
        a = b;
        b = a;
    }

    #[expect(clippy::bytes_nth)]
    fn bar() {
        let _ = "Hello".bytes().nth(3);
    }

    #[expect(clippy::if_same_then_else)]
    fn baz() {
        let _ = if true { 42 } else { 42 };
    }

    #[expect(clippy::overly_complex_bool_expr)]
    fn burger() {
        let a = false;
        let b = true;

        if a && b || a {}
    }
}

mod clippy_warn {
    //! See <https://rust-lang.github.io/rust-clippy/master/index.html>

    #[expect(clippy::almost_swapped)]
    //~^ ERROR: this lint expectation is unfulfilled
    fn foo() {
        let mut a = 0;
        let mut b = 9;
        a = b;
    }

    #[expect(clippy::bytes_nth)]
    //~^ ERROR: this lint expectation is unfulfilled
    fn bar() {
        let _ = "Hello".as_bytes().get(3);
    }

    #[expect(clippy::if_same_then_else)]
    //~^ ERROR: this lint expectation is unfulfilled
    fn baz() {
        let _ = if true { 33 } else { 42 };
    }

    #[expect(clippy::overly_complex_bool_expr)]
    //~^ ERROR: this lint expectation is unfulfilled
    fn burger() {
        let a = false;
        let b = true;
        let c = false;

        if a && b || c {}
    }
}

fn main() {
    rustc_warn::rustc_lints();
}
