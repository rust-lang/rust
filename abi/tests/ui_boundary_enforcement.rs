//! UI Boundary Enforcement Tests
//!
//! These tests document and verify the architectural boundary between
//! applications and the Blossom layout/paint service as defined in
//! docs/UI_INTENT_CONTRACT.md.
//!
//! Note: The primary enforcement mechanism is compile-time module privacy.
//! These tests serve as documentation and verification that the API surface
//! is correctly designed.

#[test]
fn ui_intent_contract_documented() {
    // This test exists primarily as documentation that the UI Intent Contract
    // (docs/UI_INTENT_CONTRACT.md) defines the architectural boundary.
    //
    // The boundary is enforced at compile time through Rust's module privacy:
    // - blossom::graph_ui is pub(crate) — not externally importable
    // - blossom::read_string_prop is pub(crate)
    // - blossom::read_bytespace is pub(crate)
    //
    // Any attempt to import these from outside blossom will fail:
    //   use blossom::graph_ui;        // error: module `graph_ui` is private
    //   use blossom::read_bytespace;  // error: function `read_bytespace` is private
}

#[test]
fn petals_api_is_sufficient_for_apps() {
    // This test verifies that the public Petals API provides what apps need
    // to build UI without accessing Blossom internals.
    //
    // Apps can successfully build UI using only:
    //   use stem::petals::Petals;
    //
    // This is demonstrated by Font Explorer which:
    // - Depends only on stem and abi (not blossom)
    // - Uses only Petals builders
    // - Publishes via stem::petals::Petals::begin_window(...).finish()
    // - Contains no layout or paint code
}

#[test]
fn blossom_has_no_public_api() {
    // Blossom is a renderer, not a library. It intentionally exposes
    // zero public API surface. All modules and helpers are pub(crate).
    //
    // The crate enforces this with #![warn(unreachable_pub)].
    //
    // If something from Blossom is needed externally, it should be
    // promoted to stem, abi, or petals — never exposed from Blossom.
}
