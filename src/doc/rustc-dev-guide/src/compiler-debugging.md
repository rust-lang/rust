# Debugging the compiler
[debugging]: #debugging

This chapter contains a few tips to debug the compiler. These tips aim to be
useful no matter what you are working on.  Some of the other chapters have
advice about specific parts of the compiler (e.g. the [Queries Debugging and
Testing
chapter](./incrcomp-debugging.html) or
the [LLVM Debugging chapter](./codegen/debugging.md)).

## `-Z` flags

The compiler has a bunch of `-Z` flags. These are unstable flags that are only
enabled on nightly. Many of them are useful for debugging. To get a full listing
of `-Z` flags, use `-Z help`.

One useful flag is `-Z verbose`, which generally enables printing more info that
could be useful for debugging.

## Getting a backtrace
[getting-a-backtrace]: #getting-a-backtrace

When you have an ICE (panic in the compiler), you can set
`RUST_BACKTRACE=1` to get the stack trace of the `panic!` like in
normal Rust programs.  IIRC backtraces **don't work** on Mac and on MinGW,
sorry. If you have trouble or the backtraces are full of `unknown`,
you might want to find some way to use Linux or MSVC on Windows.

In the default configuration, you don't have line numbers enabled, so the
backtrace looks like this:

```text
stack backtrace:
   0: std::sys::imp::backtrace::tracing::imp::unwind_backtrace
   1: std::sys_common::backtrace::_print
   2: std::panicking::default_hook::{{closure}}
   3: std::panicking::default_hook
   4: std::panicking::rust_panic_with_hook
   5: std::panicking::begin_panic
   (~~~~ LINES REMOVED BY ME FOR BREVITY ~~~~)
  32: rustc_typeck::check_crate
  33: <std::thread::local::LocalKey<T>>::with
  34: <std::thread::local::LocalKey<T>>::with
  35: rustc::ty::context::TyCtxt::create_and_enter
  36: rustc_driver::driver::compile_input
  37: rustc_driver::run_compiler
```

If you want line numbers for the stack trace, you can enable
`debuginfo-lines=true` or `debuginfo=true` in your config.toml and rebuild the
compiler. Then the backtrace will look like this:

```text
stack backtrace:
   (~~~~ LINES REMOVED BY ME FOR BREVITY ~~~~)
             at /home/user/rust/src/librustc_typeck/check/cast.rs:110
   7: rustc_typeck::check::cast::CastCheck::check
             at /home/user/rust/src/librustc_typeck/check/cast.rs:572
             at /home/user/rust/src/librustc_typeck/check/cast.rs:460
             at /home/user/rust/src/librustc_typeck/check/cast.rs:370
   (~~~~ LINES REMOVED BY ME FOR BREVITY ~~~~)
  33: rustc_driver::driver::compile_input
             at /home/user/rust/src/librustc_driver/driver.rs:1010
             at /home/user/rust/src/librustc_driver/driver.rs:212
  34: rustc_driver::run_compiler
             at /home/user/rust/src/librustc_driver/lib.rs:253
```

## Getting a backtrace for errors
[getting-a-backtrace-for-errors]: #getting-a-backtrace-for-errors

If you want to get a backtrace to the point where the compiler emits
an error message, you can pass the `-Z treat-err-as-bug=n`, which
will make the compiler skip `n` errors or `delay_span_bug` calls and then
panic on the next one. If you leave off `=n`, the compiler will assume `0` for
`n` and thus panic on the first error it encounters.

This can also help when debugging `delay_span_bug` calls - it will make
the first `delay_span_bug` call panic, which will give you a useful backtrace.

For example:

```bash
$ cat error.rs
fn main() {
    1 + ();
}
```

```bash
$ ./build/x86_64-unknown-linux-gnu/stage1/bin/rustc error.rs
error[E0277]: the trait bound `{integer}: std::ops::Add<()>` is not satisfied
 --> error.rs:2:7
  |
2 |     1 + ();
  |       ^ no implementation for `{integer} + ()`
  |
  = help: the trait `std::ops::Add<()>` is not implemented for `{integer}`

error: aborting due to previous error

$ # Now, where does the error above come from?
$ RUST_BACKTRACE=1 \
    ./build/x86_64-unknown-linux-gnu/stage1/bin/rustc \
    error.rs \
    -Z treat-err-as-bug
error[E0277]: the trait bound `{integer}: std::ops::Add<()>` is not satisfied
 --> error.rs:2:7
  |
2 |     1 + ();
  |       ^ no implementation for `{integer} + ()`
  |
  = help: the trait `std::ops::Add<()>` is not implemented for `{integer}`

error: internal compiler error: unexpected panic

note: the compiler unexpectedly panicked. this is a bug.

note: we would appreciate a bug report: https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md#bug-reports

note: rustc 1.24.0-dev running on x86_64-unknown-linux-gnu

note: run with `RUST_BACKTRACE=1` for a backtrace

thread 'rustc' panicked at 'encountered error with `-Z treat_err_as_bug',
/home/user/rust/src/librustc_errors/lib.rs:411:12
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose
backtrace.
stack backtrace:
  (~~~ IRRELEVANT PART OF BACKTRACE REMOVED BY ME ~~~)
   7: rustc::traits::error_reporting::<impl rustc::infer::InferCtxt<'a, 'gcx,
             'tcx>>::report_selection_error
             at /home/user/rust/src/librustc/traits/error_reporting.rs:823
   8: rustc::traits::error_reporting::<impl rustc::infer::InferCtxt<'a, 'gcx,
             'tcx>>::report_fulfillment_errors
             at /home/user/rust/src/librustc/traits/error_reporting.rs:160
             at /home/user/rust/src/librustc/traits/error_reporting.rs:112
   9: rustc_typeck::check::FnCtxt::select_obligations_where_possible
             at /home/user/rust/src/librustc_typeck/check/mod.rs:2192
  (~~~ IRRELEVANT PART OF BACKTRACE REMOVED BY ME ~~~)
  36: rustc_driver::run_compiler
             at /home/user/rust/src/librustc_driver/lib.rs:253
$ # Cool, now I have a backtrace for the error
```

## Getting logging output
[getting-logging-output]: #getting-logging-output

These crates are used in compiler for logging:

* [log]
* [env-logger]: check the link to see the full `RUSTC_LOG` syntax

[log]: https://docs.rs/log/0.4.6/log/index.html
[env-logger]: https://docs.rs/env_logger/0.4.3/env_logger/

The compiler has a lot of `debug!` calls, which print out logging information
at many points. These are very useful to at least narrow down the location of
a bug if not to find it entirely, or just to orient yourself as to why the
compiler is doing a particular thing.

To see the logs, you need to set the `RUSTC_LOG` environment variable to
your log filter, e.g. to get the logs for a specific module, you can run the
compiler as `RUSTC_LOG=module::path rustc my-file.rs`. All `debug!` output will
then appear in standard error.

**Note that unless you use a very strict filter, the logger will emit a lot of
output, so use the most specific module(s) you can (comma-separated if
multiple)**. It's typically a good idea to pipe standard error to a file and
look at the log output with a text editor.

So to put it together.

```bash
# This puts the output of all debug calls in `librustc/traits` into
# standard error, which might fill your console backscroll.
$ RUSTC_LOG=rustc::traits rustc +local my-file.rs

# This puts the output of all debug calls in `librustc/traits` in
# `traits-log`, so you can then see it with a text editor.
$ RUSTC_LOG=rustc::traits rustc +local my-file.rs 2>traits-log

# Not recommended. This will show the output of all `debug!` calls
# in the Rust compiler, and there are a *lot* of them, so it will be
# hard to find anything.
$ RUSTC_LOG=debug rustc +local my-file.rs 2>all-log

# This will show the output of all `info!` calls in `rustc_trans`.
#
# There's an `info!` statement in `trans_instance` that outputs
# every function that is translated. This is useful to find out
# which function triggers an LLVM assertion, and this is an `info!`
# log rather than a `debug!` log so it will work on the official
# compilers.
$ RUSTC_LOG=rustc_trans=info rustc +local my-file.rs
```

### How to keep or remove `debug!` and `trace!` calls from the resulting binary

While calls to `error!`, `warn!` and `info!` are included in every build of the compiler,
calls to `debug!` and `trace!` are only included in the program if
`debug-assertions=yes` is turned on in config.toml (it is
turned off by default), so if you don't see `DEBUG` logs, especially
if you run the compiler with `RUSTC_LOG=rustc rustc some.rs` and only see
`INFO` logs, make sure that `debug-assertions=yes` is turned on in your
config.toml.

I also think that in some cases just setting it will not trigger a rebuild,
so if you changed it and you already have a compiler built, you might
want to call `x.py clean` to force one.

### Logging etiquette and conventions

Because calls to `debug!` are removed by default, in most cases, don't worry
about adding "unnecessary" calls to `debug!` and leaving them in code you
commit - they won't slow down the performance of what we ship, and if they
helped you pinning down a bug, they will probably help someone else with a
different one.

A loosely followed convention is to use `debug!("foo(...)")` at the _start_ of
a function `foo` and `debug!("foo: ...")` _within_ the function. Another
loosely followed convention is to use the `{:?}` format specifier for debug
logs.

One thing to be **careful** of is **expensive** operations in logs.

If in the module `rustc::foo` you have a statement

```Rust
debug!("{:?}", random_operation(tcx));
```

Then if someone runs a debug `rustc` with `RUSTC_LOG=rustc::bar`, then
`random_operation()` will run.

This means that you should not put anything too expensive or likely to crash
there - that would annoy anyone who wants to use logging for their own module.
No-one will know it until someone tries to use logging to find *another* bug.

## Formatting Graphviz output (.dot files)
[formatting-graphviz-output]: #formatting-graphviz-output

Some compiler options for debugging specific features yield graphviz graphs -
e.g. the `#[rustc_mir(borrowck_graphviz_postflow="suffix.dot")]` attribute
dumps various borrow-checker dataflow graphs.

These all produce `.dot` files. To view these files, install graphviz (e.g.
`apt-get install graphviz`) and then run the following commands:

```bash
$ dot -T pdf maybe_init_suffix.dot > maybe_init_suffix.pdf
$ firefox maybe_init_suffix.pdf # Or your favorite pdf viewer
```

## Narrowing (Bisecting) Regressions

The [cargo-bisect-rustc][bisect] tool can be used as a quick and easy way to
find exactly which PR caused a change in `rustc` behavior. It automatically
downloads `rustc` PR artifacts and tests them against a project you provide
until it finds the regression. You can then look at the PR to get more context
on *why* it was changed.  See [this tutorial][bisect-tutorial] on how to use
it.

[bisect]: https://github.com/rust-lang-nursery/cargo-bisect-rustc
[bisect-tutorial]: https://github.com/rust-lang-nursery/cargo-bisect-rustc/blob/master/TUTORIAL.md

## Downloading Artifacts from Rust's CI

The [rustup-toolchain-install-master][rtim] tool by kennytm can be used to
download the artifacts produced by Rust's CI for a specific SHA1 -- this
basically corresponds to the successful landing of some PR -- and then sets
them up for your local use. This also works for artifacts produced by `@bors
try`. This is helpful when you want to examine the resulting build of a PR
without doing the build yourself.

[rtim]: https://github.com/kennytm/rustup-toolchain-install-master
