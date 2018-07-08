**Note: This is copied from the 
[rust-forge](https://github.com/rust-lang-nursery/rust-forge). If anything needs
 updating, please open an issue or make a PR on the github repo.**

# Debugging the compiler
[debugging]: #debugging

Here are a few tips to debug the compiler:

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
an error message, you can pass the `-Z treat-err-as-bug`, which
will make the compiler panic on the first error it sees.

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

The compiler has a lot of `debug!` calls, which print out logging information
at many points. These are very useful to at least narrow down the location of
a bug if not to find it entirely, or just to orient yourself as to why the 
compiler is doing a particular thing.

To see the logs, you need to set the `RUST_LOG` environment variable to
your log filter, e.g. to get the logs for a specific module, you can run the
compiler as `RUST_LOG=module::path rustc my-file.rs`. The Rust logs are
powered by [env-logger], and you can look at the docs linked there to see
the full `RUST_LOG` syntax. All `debug!` output will then appear in
standard error.

Note that unless you use a very strict filter, the logger will emit a *lot*
of output - so it's typically a good idea to pipe standard error to a file
and look at the log output with a text editor.

So to put it together.

```bash
# This puts the output of all debug calls in `librustc/traits` into
# standard error, which might fill your console backscroll.
$ RUST_LOG=rustc::traits rustc +local my-file.rs

# This puts the output of all debug calls in `librustc/traits` in
# `traits-log`, so you can then see it with a text editor.
$ RUST_LOG=rustc::traits rustc +local my-file.rs 2>traits-log

# Not recommended. This will show the output of all `debug!` calls
# in the Rust compiler, and there are a *lot* of them, so it will be
# hard to find anything.
$ RUST_LOG=debug rustc +local my-file.rs 2>all-log

# This will show the output of all `info!` calls in `rustc_trans`.
#
# There's an `info!` statement in `trans_instance` that outputs
# every function that is translated. This is useful to find out
# which function triggers an LLVM assertion, and this is an `info!`
# log rather than a `debug!` log so it will work on the official
# compilers.
$ RUST_LOG=rustc_trans=info rustc +local my-file.rs
```

While calls to `info!` are included in every build of the compiler,
calls to `debug!` are only included in the program if the
`debug-assertions=yes` is turned on in config.toml (it is
turned off by default), so if you don't see `DEBUG` logs, especially
if you run the compiler with `RUST_LOG=rustc rustc some.rs` and only see
`INFO` logs, make sure that `debug-assertions=yes` is turned on in your
config.toml.

I also think that in some cases just setting it will not trigger a rebuild,
so if you changed it and you already have a compiler built, you might
want to call `x.py clean` to force one.

### Logging etiquette

Because calls to `debug!` are removed by default, in most cases, don't worry
about adding "unnecessary" calls to `debug!` and leaving them in code you 
commit - they won't slow down the performance of what we ship, and if they 
helped you pinning down a bug, they will probably help someone else with a 
different one.

However, there are still a few concerns that you might care about:

### Expensive operations in logs

A note of caution: the expressions *within* the `debug!` call are run
whenever RUST_LOG is set, even if the filter would exclude the log. This means 
that if in the module `rustc::foo` you have a statement

```Rust
debug!("{:?}", random_operation(tcx));
```

Then if someone runs a debug `rustc` with `RUST_LOG=rustc::bar`, then 
`random_operation()` will still run - even while it's output will never be 
needed!

This means that you should not put anything too expensive or likely
to crash there - that would annoy anyone who wants to use logging for their own 
module. Note that if `RUST_LOG` is unset (the default), then the code will not 
run - this means that if your logging code panics, then no-one will know it 
until someone tries to use logging to find *another* bug.

If you *need* to do an expensive operation in a log, be aware that while log 
expressions are *evaluated* even if logging is not enabled in your module, 
they are not *formatted* unless it *is*. This means you can put your 
expensive/crashy operations inside an `fmt::Debug` impl, and they will not be 
run unless your log is enabled:

```Rust
use std::fmt;

struct ExpensiveOperationContainer<'a, 'gcx, 'tcx>
    where 'tcx: 'gcx, 'a: 'tcx
{
    tcx: TyCtxt<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> fmt::Debug for ExpensiveOperationContainer<'a, 'gcx, 'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let value = random_operation(tcx);
        fmt::Debug::fmt(&value, fmt)
    }
}

debug!("{:?}", ExpensiveOperationContainer { tcx });
```

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

## Debugging LLVM
[debugging-llvm]: #debugging-llvm

LLVM is a big project on its own that probably needs to have its own debugging
document (not that I could find one). But here are some tips that are important
in a rustc context:

The official compilers (including nightlies) have LLVM assertions disabled,
which means that LLVM assertion failures can show up as compiler crashes (not
ICEs but "real" crashes) and other sorts of weird behavior. If you are
encountering these, it is a good idea to try using a compiler with LLVM
assertions enabled - either an "alt" nightly or a compiler you build yourself
by setting `[llvm] assertions=true` in your config.toml - and
see whether anything turns up.

The rustc build process builds the LLVM tools into 
`./build/<host-triple>/llvm/bin`. They can be called directly.

The default rustc compilation pipeline has multiple codegen units, which is hard
to replicate manually and means that LLVM is called multiple times in parallel.
If you can get away with it (i.e. if it doesn't make your bug disappear),
passing `-C codegen-units=1` to rustc will make debugging easier.

If you want to play with the optimization pipeline, you can use the opt tool 
from `./build/<host-triple>/llvm/bin/` with the the LLVM IR emitted by rustc. 
Note that rustc emits different IR depending on whether `-O` is enabled, even 
without LLVM's optimizations, so if you want to play with the IR rustc emits, 
you should:

```bash
$ rustc +local my-file.rs --emit=llvm-ir -O -C no-prepopulate-passes \
    -C codegen-units=1
$ OPT=./build/$TRIPLE/llvm/bin/opt
$ $OPT -S -O2 < my-file.ll > my
```

If you just want to get the LLVM IR during the LLVM pipeline, to e.g. see which
IR causes an optimization-time assertion to fail, or to see when
LLVM performs a particular optimization, you can pass the rustc flag
`-C llvm-args=-print-after-all`, and possibly add
`-C llvm-args='-filter-print-funcs=EXACT_FUNCTION_NAME` (e.g.
`-C llvm-args='-filter-print-funcs=_ZN11collections3str21_$LT$impl$u20$str$GT$\
    7replace17hbe10ea2e7c809b0bE'`).

That produces a lot of output into standard error, so you'll want to pipe
that to some file. Also, if you are using neither `-filter-print-funcs` nor
`-C codegen-units=1`, then, because the multiple codegen units run in parallel,
the printouts will mix together and you won't be able to read anything.

If you want just the IR for a specific function (say, you want to see
why it causes an assertion or doesn't optimize correctly), you can use
`llvm-extract`, e.g.

```bash
$ ./build/$TRIPLE/llvm/bin/llvm-extract \
    -func='_ZN11collections3str21_$LT$impl$u20$str$GT$7replace17hbe10ea2e7c809b0bE' \
    -S \
    < unextracted.ll \
    > extracted.ll
```

[env-logger]: https://docs.rs/env_logger/0.4.3/env_logger/
