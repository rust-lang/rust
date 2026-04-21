// Verify the `-Z dump-trait-graph` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) dumps computed
// trait graph info for root supertraits matching a filter to stderr. It is
// implemented in compiler/rustc_monomorphize/src/partitioning.rs
// (`dump_trait_graph`).
//
// This test verifies:
//   1. `-Zdump-trait-graph=all` emits the expected section headers, sub-trait
//      enumeration, and table layout for a simple trait hierarchy.
//   2. Filtering by a substring of the root trait name still produces the
//      dump (filter path exercised).
//   3. Filtering by a string that matches no root produces no dump
//      (negative case — no `=== Trait Graph:` header).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. filter = "all" --------------------------------------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-graph=all")
        .run()
        // Section header for the root supertrait.
        .assert_stderr_contains("=== Trait Graph:")
        // The root trait from test.rs should appear in a header.
        .assert_stderr_contains("GraphRoot")
        // Sub-trait enumeration ran.
        .assert_stderr_contains("Sub-traits (")
        // Concrete type enumeration ran.
        .assert_stderr_contains("Concrete types (")
        // Table layout was computed.
        .assert_stderr_contains("Table layout:")
        // At least one slot was emitted.
        .assert_stderr_contains("slot[0]:");

    // ---- 2. filter = substring of root name --------------------------------
    // "GraphRoot" is a substring of the root's printed type; the dump should
    // still fire.
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-graph=GraphRoot")
        .run()
        .assert_stderr_contains("=== Trait Graph:")
        .assert_stderr_contains("GraphRoot")
        .assert_stderr_contains("Table layout:")
        .assert_stderr_contains("slot[0]:");

    // ---- 3. filter matching no root (negative case) ------------------------
    // No root's printed type contains this string, so the dump must be silent.
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-graph=DefinitelyNotARealTrait")
        .run()
        .assert_stderr_not_contains("=== Trait Graph:");
}
