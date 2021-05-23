/// Checks if we're in a nightly build.
///
/// The environment variable `CFG_RELEASE_CHANNEL` is set during the rustc bootstrap
/// to "stable", "beta", or "nightly" depending on what toolchain is being built.
/// If we are being built as part of the stable or beta toolchains, we want
/// to disable unstable configuration options.
///
/// If we're being built by cargo (e.g., `cargo +nightly install rustfmt-nightly`),
/// `CFG_RELEASE_CHANNEL` is not set. As we only support being built against the
/// nightly compiler when installed from crates.io, default to nightly mode.
#[macro_export]
macro_rules! is_nightly_channel {
    () => {
        option_env!("CFG_RELEASE_CHANNEL").map_or(true, |c| c == "nightly" || c == "dev")
    };
}
