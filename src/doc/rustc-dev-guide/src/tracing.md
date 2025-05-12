# Using tracing to debug the compiler

<!-- toc -->

The compiler has a lot of [`debug!`] (or `trace!`) calls, which print out logging information
at many points. These are very useful to at least narrow down the location of
a bug if not to find it entirely, or just to orient yourself as to why the
compiler is doing a particular thing.

[`debug!`]: https://docs.rs/tracing/0.1/tracing/macro.debug.html

To see the logs, you need to set the `RUSTC_LOG` environment variable to your
log filter. The full syntax of the log filters can be found in the [rustdoc
of `tracing-subscriber`](https://docs.rs/tracing-subscriber/0.2.24/tracing_subscriber/filter/struct.EnvFilter.html#directives).

## Function level filters

Lots of functions in rustc are annotated with

```
#[instrument(level = "debug", skip(self))]
fn foo(&self, bar: Type) {}
```

which allows you to use

```
RUSTC_LOG=[foo]
```

to do the following all at once

* log all function calls to `foo`
* log the arguments (except for those in the `skip` list)
* log everything (from anywhere else in the compiler) until the function returns

### I don't want everything

Depending on the scope of the function, you may not want to log everything in its body.
As an example: the `do_mir_borrowck` function will dump hundreds of lines even for trivial
code being borrowchecked.

Since you can combine all filters, you can add a crate/module path, e.g.

```
RUSTC_LOG=rustc_borrowck[do_mir_borrowck]
```

### I don't want all calls

If you are compiling libcore, you likely don't want *all* borrowck dumps, but only one
for a specific function. You can filter function calls by their arguments by regexing them.

```
RUSTC_LOG=[do_mir_borrowck{id=\.\*from_utf8_unchecked\.\*}]
```

will only give you the logs of borrowchecking `from_utf8_unchecked`. Note that you will
still get a short message per ignored `do_mir_borrowck`, but none of the things inside those
calls. This helps you in looking through the calls that are happening and helps you adjust
your regex if you mistyped it.

## Query level filters

Every [query](query.md) is automatically tagged with a logging span so that
you can display all log messages during the execution of the query. For
example, if you want to log everything during type checking:

```
RUSTC_LOG=[typeck]
```

The query arguments are included as a tracing field which means that you can
filter on the debug display of the arguments. For example, the `typeck` query
has an argument `key: LocalDefId` of what is being checked. You can use a
regex to match on that `LocalDefId` to log type checking for a specific
function:

```
RUSTC_LOG=[typeck{key=.*name_of_item.*}]
```

Different queries have different arguments. You can find a list of queries and
their arguments in
[`rustc_middle/src/query/mod.rs`](https://github.com/rust-lang/rust/blob/master/compiler/rustc_middle/src/query/mod.rs#L18).

## Broad module level filters

You can also use filters similar to the `log` crate's filters, which will enable
everything within a specific module. This is often too verbose and too unstructured,
so it is recommended to use function level filters.

Your log filter can be just `debug` to get all `debug!` output and
higher (e.g., it will also include `info!`), or `path::to::module` to get *all*
output (which will include `trace!`) from a particular module, or
`path::to::module=debug` to get `debug!` output and higher from a particular
module.

For example, to get the `debug!` output and higher for a specific module, you
can run the compiler with `RUSTC_LOG=path::to::module=debug rustc my-file.rs`.
All `debug!` output will then appear in standard error.

Note that you can use a partial path and the filter will still work. For
example, if you want to see `info!` output from only
`rustdoc::passes::collect_intra_doc_links`, you could use
`RUSTDOC_LOG=rustdoc::passes::collect_intra_doc_links=info` *or* you could use
`RUSTDOC_LOG=rustdoc::passes::collect_intra=info`.

If you are developing rustdoc, use `RUSTDOC_LOG` instead. If you are developing
Miri, use `MIRI_LOG` instead. You get the idea :)

See the [`tracing`] crate's docs, and specifically the docs for [`debug!`] to
see the full syntax you can use. (Note: unlike the compiler, the [`tracing`]
crate and its examples use the `RUST_LOG` environment variable. rustc, rustdoc,
and other tools set custom environment variables.)

**Note that unless you use a very strict filter, the logger will emit a lot of
output, so use the most specific module(s) you can (comma-separated if
multiple)**. It's typically a good idea to pipe standard error to a file and
look at the log output with a text editor.

So, to put it together:

```bash
# This puts the output of all debug calls in `rustc_middle/src/traits` into
# standard error, which might fill your console backscroll.
$ RUSTC_LOG=rustc_middle::traits=debug rustc +stage1 my-file.rs

# This puts the output of all debug calls in `rustc_middle/src/traits` in
# `traits-log`, so you can then see it with a text editor.
$ RUSTC_LOG=rustc_middle::traits=debug rustc +stage1 my-file.rs 2>traits-log

# Not recommended! This will show the output of all `debug!` calls
# in the Rust compiler, and there are a *lot* of them, so it will be
# hard to find anything.
$ RUSTC_LOG=debug rustc +stage1 my-file.rs 2>all-log

# This will show the output of all `info!` calls in `rustc_codegen_ssa`.
#
# There's an `info!` statement in `codegen_instance` that outputs
# every function that is codegen'd. This is useful to find out
# which function triggers an LLVM assertion, and this is an `info!`
# log rather than a `debug!` log so it will work on the official
# compilers.
$ RUSTC_LOG=rustc_codegen_ssa=info rustc +stage1 my-file.rs

# This will show all logs in `rustc_codegen_ssa` and `rustc_resolve`.
$ RUSTC_LOG=rustc_codegen_ssa,rustc_resolve rustc +stage1 my-file.rs

# This will show the output of all `info!` calls made by rustdoc
# or any rustc library it calls.
$ RUSTDOC_LOG=info rustdoc +stage1 my-file.rs

# This will only show `debug!` calls made by rustdoc directly,
# not any `rustc*` crate.
$ RUSTDOC_LOG=rustdoc=debug rustdoc +stage1 my-file.rs
```

## Log colors

By default, rustc (and other tools, like rustdoc and Miri) will be smart about
when to use ANSI colors in the log output. If they are outputting to a terminal,
they will use colors, and if they are outputting to a file or being piped
somewhere else, they will not. However, it's hard to read log output in your
terminal unless you have a very strict filter, so you may want to pipe the
output to a pager like `less`. But then there won't be any colors, which makes
it hard to pick out what you're looking for!

You can override whether to have colors in log output with the `RUSTC_LOG_COLOR`
environment variable (or `RUSTDOC_LOG_COLOR` for rustdoc, or `MIRI_LOG_COLOR`
for Miri, etc.). There are three options: `auto` (the default), `always`, and
`never`. So, if you want to enable colors when piping to `less`, use something
similar to this command:

```bash
# The `-R` switch tells less to print ANSI colors without escaping them.
$ RUSTC_LOG=debug RUSTC_LOG_COLOR=always rustc +stage1 ... | less -R
```

Note that `MIRI_LOG_COLOR` will only color logs that come from Miri, not logs
from rustc functions that Miri calls. Use `RUSTC_LOG_COLOR` to color logs from
rustc.

## How to keep or remove `debug!` and `trace!` calls from the resulting binary

While calls to `error!`, `warn!` and `info!` are included in every build of the compiler,
calls to `debug!` and `trace!` are only included in the program if
`debug-logging=true` is turned on in bootstrap.toml (it is
turned off by default), so if you don't see `DEBUG` logs, especially
if you run the compiler with `RUSTC_LOG=rustc rustc some.rs` and only see
`INFO` logs, make sure that `debug-logging=true` is turned on in your
bootstrap.toml.

## Logging etiquette and conventions

Because calls to `debug!` are removed by default, in most cases, don't worry
about the performance of adding "unnecessary" calls to `debug!` and leaving them in code you
commit - they won't slow down the performance of what we ship.

That said, there can also be excessive tracing calls, especially
when they are redundant with other calls nearby or in functions called from
here. There is no perfect balance to hit here, and is left to the reviewer's
discretion to decide whether to let you leave `debug!` statements in or whether to ask
you to remove them before merging.

It may be preferable to use `trace!` over `debug!` for very noisy logs.

A loosely followed convention is to use `#[instrument(level = "debug")]`
([also see the attribute's documentation](https://docs.rs/tracing-attributes/0.1.17/tracing_attributes/attr.instrument.html))
in favour of `debug!("foo(...)")` at the start of a function `foo`.
Within functions, prefer `debug!(?variable.field)` over `debug!("xyz = {:?}", variable.field)`
and `debug!(bar = ?var.method(arg))` over `debug!("bar = {:?}", var.method(arg))`.
The documentation for this syntax can be found [here](https://docs.rs/tracing/0.1.28/tracing/#recording-fields).

One thing to be **careful** of is **expensive** operations in logs.

If in the module `rustc::foo` you have a statement

```Rust
debug!(x = ?random_operation(tcx));
```

Then if someone runs a debug `rustc` with `RUSTC_LOG=rustc::foo`, then
`random_operation()` will run. `RUSTC_LOG` filters that do not enable this
debug statement will not execute `random_operation`.

This means that you should not put anything too expensive or likely to crash
there - that would annoy anyone who wants to use logging for that module.
No-one will know it until someone tries to use logging to find *another* bug.

[`tracing`]: https://docs.rs/tracing
