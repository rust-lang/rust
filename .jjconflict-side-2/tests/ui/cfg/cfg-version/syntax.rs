//! Check `#[cfg(version(..))]` parsing.

#![feature(cfg_version)]

// Overall grammar
// ===============
//
// `#[cfg(version(..))]` accepts only the `version(VERSION_STRING_LITERAL)` predicate form, where
// only a single string literal is permitted.

#[cfg(version(42))]
//~^ ERROR expected a version literal
fn not_a_string_literal_simple() {}

#[cfg(version(1.20))]
//~^ ERROR expected a version literal
fn not_a_string_literal_semver_like() {}

#[cfg(version(false))]
//~^ ERROR expected a version literal
fn not_a_string_literal_other() {}

#[cfg(version("1.43", "1.44", "1.45"))]
//~^ ERROR expected single version literal
fn multiple_version_literals() {}

// The key-value form `cfg(version = "..")` is not considered a valid `cfg(version(..))` usage, but
// it will only trigger the `unexpected_cfgs` lint and not a hard error.

#[cfg(version = "1.43")]
//~^ WARN unexpected `cfg` condition name: `version`
fn key_value_form() {}

// Additional version string literal constraints
// =============================================
//
// The `VERSION_STRING_LITERAL` ("version literal") has additional constraints on its syntactical
// well-formedness.

// 1. A valid version literal can only constitute of numbers and periods (a "simple" semver version
// string). Non-semver strings or "complex" semver strings (such as build metadata) are not
// considered valid version literals, and will emit a non-lint warning "unknown version literal
// format".

#[cfg(version("1.43.0"))]
fn valid_major_minor_patch() {}

#[cfg(version("0.0.0"))]
fn valid_zero_zero_zero_major_minor_patch() {}

#[cfg(version("foo"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn not_numbers_or_periods() {}

#[cfg(version("1.20.0-stable"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn complex_semver_with_metadata() {}

// 2. "Shortened" version strings are permitted but *only* for the omission of the patch number.

#[cfg(version("1.0"))]
fn valid_major_minor_1() {}

#[cfg(version("1.43"))]
fn valid_major_minor_2() {}

#[cfg(not(version("1.44")))]
fn valid_major_minor_negated_smoke_test() {}

#[cfg(version("0.0"))]
fn valid_zero_zero_major_minor() {}

#[cfg(version("0.7"))]
fn valid_zero_major_minor() {}

// 3. Major-only, or other non-Semver-like strings are not permitted.

#[cfg(version("1"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn invalid_major_only() {}

#[cfg(version("0"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn invalid_major_only_zero() {}

#[cfg(version(".7"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn invalid_decimal_like() {}

// Misc parsing overflow/underflow edge cases
// ==========================================
//
// Check that we report "unknown version literal format" user-facing warnings and not ICEs.

#[cfg(version("-1"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn invalid_major_only_negative() {}

// Implementation detail: we store rustc version as `{ major: u16, minor: u16, patch: u16 }`.

#[cfg(version("65536"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn exceed_u16_major() {}

#[cfg(version("1.65536.0"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn exceed_u16_minor() {}

#[cfg(version("1.0.65536"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn exceed_u16_patch() {}

#[cfg(version("65536.0.65536"))]
//~^ WARN unknown version literal format, assuming it refers to a future version
fn exceed_u16_mixed() {}

// Usage as `cfg!()`
// =================

fn cfg_usage() {
    assert!(cfg!(version("1.0")));
    assert!(cfg!(version("1.43")));
    assert!(cfg!(version("1.43.0")));

    assert!(cfg!(version("foo")));
    //~^ WARN unknown version literal format, assuming it refers to a future version
    assert!(cfg!(version("1.20.0-stable")));
    //~^ WARN unknown version literal format, assuming it refers to a future version

    assert!(cfg!(version = "1.43"));
    //~^ WARN unexpected `cfg` condition name: `version`
}

fn main() {
    cfg_usage();

    // `cfg(version = "..")` is not a valid `cfg_version` form, but it only triggers
    // `unexpected_cfgs` lint, and `cfg(version = "..")` eval to `false`.
    key_value_form(); //~ ERROR cannot find function

    // Invalid version literal formats within valid `cfg(version(..))` form should also cause
    // `cfg(version(..))` eval to `false`.
    not_numbers_or_periods(); //~ ERROR cannot find function
    complex_semver_with_metadata(); //~ ERROR cannot find function
    invalid_major_only(); //~ ERROR cannot find function
    invalid_major_only_zero(); //~ ERROR cannot find function
    invalid_major_only_negative(); //~ ERROR cannot find function
    exceed_u16_major(); //~ ERROR cannot find function
    exceed_u16_minor(); //~ ERROR cannot find function
    exceed_u16_patch(); //~ ERROR cannot find function
    exceed_u16_mixed(); //~ ERROR cannot find function
}
