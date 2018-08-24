// This file is left intentionally empty (and not removed) to avoid an issue
// where this crate is always considered dirty due to compiler-builtins'
// `cargo:rerun-if-changed=build.rs` directive; since the path is relative, it
// refers to this file when this shim crate is being built, and the absence of
// this file is considered by cargo to be equivalent to it having changed.
