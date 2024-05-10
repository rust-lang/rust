use std::env;

#[cfg(miri)]
compile_error!("`miri` cfg should not be set in build script");

fn main() {
    // Cargo calls `miri --print=cfg` to populate the `CARGO_CFG_*` env vars.
    // Make sure that the "miri" flag is not set since we are building a procedural macro crate.
    assert!(env::var_os("CARGO_CFG_MIRI").is_none());
}
