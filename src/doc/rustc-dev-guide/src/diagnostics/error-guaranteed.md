# `ErrorGuaranteed`
The previous sections have been about the error message that a user of the
compiler sees. But emitting an error can also have a second important side
effect within the compiler source code: it generates an
[`ErrorGuaranteed`][errorguar].

`ErrorGuaranteed` is a zero-sized type that is unconstructable outside of the
[`rustc_errors`][rerrors] crate. It is generated whenever an error is reported
to the user, so that if your compiler code ever encounters a value of type
`ErrorGuaranteed`, the compilation is _statically guaranteed to fail_. This is
useful for avoiding unsoundness bugs because you can statically check that an
error code path leads to a failure.

There are some important considerations about the usage of `ErrorGuaranteed`:

* It does _not_ convey information about the _kind_ of error. For example, the
  error may be due (indirectly) to a delayed bug or other compiler error.
  Thus, you should not rely on
  `ErrorGuaranteed` when deciding whether to emit an error, or what kind of error
  to emit.
* `ErrorGuaranteed` should not be used to indicate that a compilation _will
  emit_ an error in the future. It should be used to indicate that an error
  _has already been_ emitted -- that is, the [`emit()`][emit] function has
  already been called.  For example, if we detect that a future part of the
  compiler will error, we _cannot_ use `ErrorGuaranteed` unless we first emit
  an error or delayed bug ourselves.

Thankfully, in most cases, it should be statically impossible to abuse
`ErrorGuaranteed`.

[errorguar]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.ErrorGuaranteed.html
[rerrors]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html
[emit]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/diagnostic/struct.Diag.html#method.emit
