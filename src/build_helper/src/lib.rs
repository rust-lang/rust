//! Types and functions shared across tools in this workspace.

pub mod ci;
pub mod drop_bomb;
pub mod fs;
pub mod git;
pub mod metrics;
pub mod stage0_parser;
pub mod util;

/// The default set of crates for opt-dist to collect LLVM profiles.
pub const LLVM_PGO_CRATES: &[&str] = &[
    "syn-2.0.101",
    "cargo-0.87.1",
    "serde-1.0.219",
    "ripgrep-14.1.1",
    "regex-automata-0.4.8",
    "clap_derive-4.5.32",
    "hyper-1.6.0",
];

/// The default set of crates for opt-dist to collect rustc profiles.
pub const RUSTC_PGO_CRATES: &[&str] = &[
    "externs",
    "ctfe-stress-5",
    "cargo-0.87.1",
    "token-stream-stress",
    "match-stress",
    "tuple-stress",
    "diesel-2.2.10",
    "bitmaps-3.2.1",
];
