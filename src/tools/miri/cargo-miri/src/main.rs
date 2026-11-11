#![allow(clippy::useless_format, clippy::derive_partial_eq_without_eq, rustc::internal)]

mod arg;
mod phases;
mod setup;
mod util;

use std::env;

use crate::arg::Args;
use crate::phases::*;
use crate::util::show_error;

/// Returns `true` if our flags look like they may be for rustdoc, i.e., this is cargo calling us to
/// be rustdoc. It's hard to be sure as cargo does not have a RUSTDOC_WRAPPER or an env var that
/// would let us get a clear signal.
fn looks_like_rustdoc(args: &Args) -> bool {
    // The `--test-run-directory` flag only exists for rustdoc and cargo always passes it. Perfect!
    // FIXME: use `has_arg_flag` instead?
    args.iter().any(|arg| arg == "--test-run-directory")
}

fn main() {
    // Rustc does not support non-UTF-8 arguments so we make no attempt either.
    // (We do support non-UTF-8 environment variables though.)
    let mut args = Args::from_env();

    // Dispatch to `cargo-miri` phase. Here is a rough idea of "who calls who".
    //
    // Initially, we are invoked as `cargo-miri miri run/test`. We first run the setup phase:
    // - We use `rustc-build-sysroot`, and set `RUSTC` back to us, together with `MIRI_CALLED_FROM_SETUP`,
    //   so that the sysroot build rustc invocations end up in `phase_rustc` with `RustcPhase::Setup`.
    //   There we then call the Miri driver with `MIRI_BE_RUSTC` to perform the actual build.
    //
    // Then we call `cargo run/test`, exactly forwarding all user flags, plus some configuration so
    // that we control every binary invoked by cargo:
    // - We set RUSTC_WRAPPER to ourselves, so for (almost) all rustc invocations, we end up in
    //   `phase_rustc` with `RustcPhase::Build`. This will in turn either determine that a
    //   dependency needs to be built (for which it invokes the Miri driver with `MIRI_BE_RUSTC`),
    //   or determine that this is a binary Miri should run, in which case we generate a JSON file
    //   with all the information needed to build and run this crate.
    //   (We don't run it yet since cargo thinks this is a build step, not a run step -- running the
    //   binary here would lead to a bad user experience.)
    // - We set RUSTC to the Miri driver and also set `MIRI_BE_RUSTC`, so that gets called by build
    //   scripts (and cargo uses it for the version query).
    // - We set `target.*.runner` to `cargo-miri runner`, which ends up calling `phase_runner` for
    //   `RunnerPhase::Cargo`. This parses the JSON file written in `phase_rustc` and then invokes
    //   the actual Miri driver for interpretation.
    // - We set RUSTDOC to ourselves, which ends up in `phase_rustdoc`. There we call regular
    //   rustdoc with some extra flags, and we set `MIRI_CALLED_FROM_RUSTDOC` to recognize this
    //   phase in our recursive invocations:
    //   - We set the `--test-builder` flag of rustdoc to ourselves, which ends up in `phase_rustc`
    //     with `RustcPhase::Rustdoc`. There we perform a check-build (needed to get the expected
    //     build failures for `compile_fail` doctests) and then store a JSON file with the
    //     information needed to run this test.
    //   - We also set `--test-runtool` to ourselves, which ends up in `phase_runner` with
    //     `RunnerPhase::Rustdoc`. There we parse the JSON file written in `phase_rustc` and invoke
    //     the Miri driver for interpretation.

    // Dispatch running as part of sysroot compilation.
    if env::var_os("MIRI_CALLED_FROM_SETUP").is_some() {
        phase_rustc(args, RustcPhase::Setup);
        return;
    }

    let Some(first) = args.first() else {
        show_error!(
            "`cargo-miri` called without first argument; please only invoke this binary through `cargo miri`"
        );
    };

    // The way rustdoc invokes rustc is indistinguishable from the way cargo invokes rustdoc by the
    // arguments alone. `phase_cargo_rustdoc` sets this environment variable to let us disambiguate.
    if env::var_os("MIRI_CALLED_FROM_RUSTDOC").is_some() {
        // ...however, we then also see this variable when rustdoc invokes us as the testrunner!
        // In that case the first argument is `runner` and there are no more arguments.
        match first {
            "runner" => {
                args.consume_first();
                phase_runner(args, RunnerPhase::Rustdoc);
            }
            flag if flag.starts_with("--") || flag.starts_with('@') => {
                // This is probably rustdoc invoking us to build the test.
                // Do NOT consume the first argument! It is meant for rustc.
                phase_rustc(args, RustcPhase::Rustdoc);
            }
            _ => {
                show_error!(
                    "`cargo-miri` failed to recognize which phase of the build process this is, please report a bug.\n\
                    We are inside MIRI_CALLED_FROM_RUSTDOC.\n\
                    The command-line arguments were: {:#?}",
                    args.as_slice(),
                );
            }
        }

        return;
    }

    match first {
        "miri" => {
            args.consume_first();
            phase_cargo_miri(args);
        }
        "runner" => {
            args.consume_first();
            phase_runner(args, RunnerPhase::Cargo);
        }
        arg if arg == env::var("RUSTC").unwrap_or_else(|_| {
            show_error!(
                "`cargo-miri` called without RUSTC set; please only invoke this binary through `cargo miri`"
            )
        }) => {
            // If the first arg is equal to the RUSTC env variable (which should be set at this
            // point), then we need to behave as rustc. This is the somewhat counter-intuitive
            // behavior of having both RUSTC and RUSTC_WRAPPER set
            // (see https://github.com/rust-lang/cargo/issues/10886).
            args.consume_first();
            phase_rustc(args, RustcPhase::Build);
        }
        _ if looks_like_rustdoc(&args) => {
            // This is probably rustdoc. Do NOT consume the first argument! It is meant for rustdoc.
            phase_rustdoc(args);
        }
        _ => {
            show_error!(
                "`cargo-miri` failed to recognize which phase of the build process this is, please report a bug.\nThe command-line arguments were: {:#?}",
                args.as_slice(),
            );
        }
    }
}
