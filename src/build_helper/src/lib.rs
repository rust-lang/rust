//! Types and functions shared across tools in this workspace.

pub mod ci;
pub mod drop_bomb;
pub mod git;
pub mod metrics;
pub mod stage0_parser;
pub mod util;

/// The default set of crates for opt-dist to collect LLVM profiles.
pub const LLVM_PGO_CRATES: &[&str] = &[
    "syn-1.0.89",
    "cargo-0.60.0",
    "serde-1.0.136",
    "ripgrep-13.0.0",
    "regex-1.5.5",
    "clap-3.1.6",
    "hyper-0.14.18",
];

/// The default set of crates for opt-dist to collect rustc profiles.
pub const RUSTC_PGO_CRATES: &[&str] = &[
    "externs",
    "ctfe-stress-5",
    "cargo-0.60.0",
    "token-stream-stress",
    "match-stress",
    "tuple-stress",
    "diesel-1.4.8",
    "bitmaps-3.1.0",
];
