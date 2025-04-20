//! Option `--test-args ARGS` has been deprecated in favor of `--test-arg ARG`.
//! Test that it's still accepted.
// FIXME: Use this test to check that using this flag will emit a
//        (hard) deprecation warning once it does.

//@ compile-flags: --test-args argument
//@ check-pass
