//@ legacy-makefile-test

//@ only-x86_64

//@ ignore-test
// FIXME(jieyouxu): This test never runs because the `ifeq` check in the Makefile
// compares `x86` to `x86_64`, which always evaluates to false.
// When the test does run, the compilation does not include `.note.gnu.property`.
// See https://github.com/rust-lang/rust/pull/126720 for more information.
