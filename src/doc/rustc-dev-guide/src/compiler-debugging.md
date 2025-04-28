# Debugging the compiler

<!-- toc -->

This chapter contains a few tips to debug the compiler. These tips aim to be
useful no matter what you are working on.  Some of the other chapters have
advice about specific parts of the compiler (e.g. the [Queries Debugging and
Testing chapter](./incrcomp-debugging.html) or the [LLVM Debugging
chapter](./backend/debugging.md)).

## Configuring the compiler

By default, rustc is built without most debug information. To enable debug info,
set `debug = true` in your bootstrap.toml.

Setting `debug = true` turns on many different debug options (e.g., `debug-assertions`,
`debug-logging`, etc.) which can be individually tweaked if you want to, but many people
simply set `debug = true`.

If you want to use GDB to debug rustc, please set `bootstrap.toml` with options:

```toml
[rust]
debug = true
debuginfo-level = 2
```

> NOTE:
> This will use a lot of disk space
> (upwards of <!-- date-check Aug 2022 --> 35GB),
> and will take a lot more compile time.
> With `debuginfo-level = 1` (the default when `debug = true`),
> you will be able to track the execution path,
> but will lose the symbol information for debugging.

The default configuration will enable `symbol-mangling-version` v0.
This requires at least GDB v10.2,
otherwise you need to disable new symbol-mangling-version in `bootstrap.toml`.

```toml
[rust]
new-symbol-mangling = false
```

> See the comments in `bootstrap.example.toml` for more info.

You will need to rebuild the compiler after changing any configuration option.

## Suppressing the ICE file

By default, if rustc encounters an Internal Compiler Error (ICE) it will dump the ICE contents to an
ICE file within the current working directory named `rustc-ice-<timestamp>-<pid>.txt`. If this is
not desirable, you can prevent the ICE file from being created with `RUSTC_ICE=0`.

## Getting a backtrace
[getting-a-backtrace]: #getting-a-backtrace

When you have an ICE (panic in the compiler), you can set
`RUST_BACKTRACE=1` to get the stack trace of the `panic!` like in
normal Rust programs. IIRC backtraces **don't work** on MinGW,
sorry. If you have trouble or the backtraces are full of `unknown`,
you might want to find some way to use Linux, Mac, or MSVC on Windows.

In the default configuration (without `debug` set to `true`), you don't have line numbers
enabled, so the backtrace looks like this:

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

If you set `debug = true`, you will get line numbers for the stack trace.
Then the backtrace will look like this:

```text
stack backtrace:
   (~~~~ LINES REMOVED BY ME FOR BREVITY ~~~~)
             at /home/user/rust/compiler/rustc_typeck/src/check/cast.rs:110
   7: rustc_typeck::check::cast::CastCheck::check
             at /home/user/rust/compiler/rustc_typeck/src/check/cast.rs:572
             at /home/user/rust/compiler/rustc_typeck/src/check/cast.rs:460
             at /home/user/rust/compiler/rustc_typeck/src/check/cast.rs:370
   (~~~~ LINES REMOVED BY ME FOR BREVITY ~~~~)
  33: rustc_driver::driver::compile_input
             at /home/user/rust/compiler/rustc_driver/src/driver.rs:1010
             at /home/user/rust/compiler/rustc_driver/src/driver.rs:212
  34: rustc_driver::run_compiler
             at /home/user/rust/compiler/rustc_driver/src/lib.rs:253
```

## `-Z` flags

The compiler has a bunch of `-Z *` flags. These are unstable flags that are only
enabled on nightly. Many of them are useful for debugging. To get a full listing
of `-Z` flags, use `-Z help`.

One useful flag is `-Z verbose-internals`, which generally enables printing more
info that could be useful for debugging.

Right below you can find elaborate explainers on a selected few.

### Getting a backtrace for errors
[getting-a-backtrace-for-errors]: #getting-a-backtrace-for-errors

If you want to get a backtrace to the point where the compiler emits an
error message, you can pass the `-Z treat-err-as-bug=n`, which will make
the compiler panic on the `nth` error. If you leave off `=n`, the compiler will
assume `1` for `n` and thus panic on the first error it encounters.

For example:

```bash
$ cat error.rs
```

```rust
fn main() {
    1 + ();
}
```

```
$ rustc +stage1 error.rs
error[E0277]: cannot add `()` to `{integer}`
 --> error.rs:2:7
  |
2 |       1 + ();
  |         ^ no implementation for `{integer} + ()`
  |
  = help: the trait `Add<()>` is not implemented for `{integer}`

error: aborting due to previous error
```

Now, where does the error above come from?

```
$ RUST_BACKTRACE=1 rustc +stage1 error.rs -Z treat-err-as-bug
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
/home/user/rust/compiler/rustc_errors/src/lib.rs:411:12
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose
backtrace.
stack backtrace:
  (~~~ IRRELEVANT PART OF BACKTRACE REMOVED BY ME ~~~)
   7: rustc::traits::error_reporting::<impl rustc::infer::InferCtxt<'a, 'tcx>>
             ::report_selection_error
             at /home/user/rust/compiler/rustc_middle/src/traits/error_reporting.rs:823
   8: rustc::traits::error_reporting::<impl rustc::infer::InferCtxt<'a, 'tcx>>
             ::report_fulfillment_errors
             at /home/user/rust/compiler/rustc_middle/src/traits/error_reporting.rs:160
             at /home/user/rust/compiler/rustc_middle/src/traits/error_reporting.rs:112
   9: rustc_typeck::check::FnCtxt::select_obligations_where_possible
             at /home/user/rust/compiler/rustc_typeck/src/check/mod.rs:2192
  (~~~ IRRELEVANT PART OF BACKTRACE REMOVED BY ME ~~~)
  36: rustc_driver::run_compiler
             at /home/user/rust/compiler/rustc_driver/src/lib.rs:253
```

Cool, now I have a backtrace for the error!

### Debugging delayed bugs

The `-Z eagerly-emit-delayed-bugs` option makes it easy to debug delayed bugs.
It turns them into normal errors, i.e. makes them visible. This can be used in
combination with `-Z treat-err-as-bug` to stop at a particular delayed bug and
get a backtrace.

### Getting the error creation location

`-Z track-diagnostics` can help figure out where errors are emitted. It uses `#[track_caller]`
for this and prints its location alongside the error:

```
$ RUST_BACKTRACE=1 rustc +stage1 error.rs -Z track-diagnostics
error[E0277]: cannot add `()` to `{integer}`
 --> src\error.rs:2:7
  |
2 |     1 + ();
  |       ^ no implementation for `{integer} + ()`
-Ztrack-diagnostics: created at compiler/rustc_trait_selection/src/traits/error_reporting/mod.rs:638:39
  |
  = help: the trait `Add<()>` is not implemented for `{integer}`
  = help: the following other types implement trait `Add<Rhs>`:
            <&'a f32 as Add<f32>>
            <&'a f64 as Add<f64>>
            <&'a i128 as Add<i128>>
            <&'a i16 as Add<i16>>
            <&'a i32 as Add<i32>>
            <&'a i64 as Add<i64>>
            <&'a i8 as Add<i8>>
            <&'a isize as Add<isize>>
          and 48 others

For more information about this error, try `rustc --explain E0277`.
```

This is similar but different to `-Z treat-err-as-bug`:
- it will print the locations for all errors emitted
- it does not require a compiler built with debug symbols
- you don't have to read through a big stack trace.

## Getting logging output

The compiler uses the [`tracing`] crate for logging.

[`tracing`]: https://docs.rs/tracing

For details see [the guide section on tracing](./tracing.md)

## Narrowing (Bisecting) Regressions

The [cargo-bisect-rustc][bisect] tool can be used as a quick and easy way to
find exactly which PR caused a change in `rustc` behavior. It automatically
downloads `rustc` PR artifacts and tests them against a project you provide
until it finds the regression. You can then look at the PR to get more context
on *why* it was changed.  See [this tutorial][bisect-tutorial] on how to use
it.

[bisect]: https://github.com/rust-lang/cargo-bisect-rustc
[bisect-tutorial]: https://rust-lang.github.io/cargo-bisect-rustc/tutorial.html

## Downloading Artifacts from Rust's CI

The [rustup-toolchain-install-master][rtim] tool by kennytm can be used to
download the artifacts produced by Rust's CI for a specific SHA1 -- this
basically corresponds to the successful landing of some PR -- and then sets
them up for your local use. This also works for artifacts produced by `@bors
try`. This is helpful when you want to examine the resulting build of a PR
without doing the build yourself.

[rtim]: https://github.com/kennytm/rustup-toolchain-install-master

## `#[rustc_*]` TEST attributes

The compiler defines a whole lot of internal (perma-unstable) attributes some of which are useful
for debugging by dumping extra compiler-internal information. These are prefixed with `rustc_` and
are gated behind the internal feature `rustc_attrs` (enabled via e.g. `#![feature(rustc_attrs)]`).

For a complete and up to date list, see [`builtin_attrs`]. More specifically, the ones marked `TEST`.
Here are some notable ones:

| Attribute | Description |
|----------------|-------------|
| `rustc_def_path` | Dumps the [`def_path_str`] of an item. |
| `rustc_dump_def_parents` | Dumps the chain of `DefId` parents of certain definitions. |
| `rustc_dump_item_bounds` | Dumps the [`item_bounds`] of an item. |
| `rustc_dump_predicates` | Dumps the [`predicates_of`] an item. |
| `rustc_dump_vtable` | Dumps the vtable layout of an impl, or a type alias of a dyn type. |
| `rustc_hidden_type_of_opaques` | Dumps the [hidden type of each opaque types][opaq] in the crate. |
| `rustc_layout` | [See this section](#debugging-type-layouts). |
| `rustc_object_lifetime_default` | Dumps the [object lifetime defaults] of an item. |
| `rustc_outlives` | Dumps implied bounds of an item. More precisely, the [`inferred_outlives_of`] an item. |
| `rustc_regions` | Dumps NLL closure region requirements. |
| `rustc_symbol_name` | Dumps the mangled & demangled [`symbol_name`] of an item. |
| `rustc_variances` | Dumps the [variances] of an item. |

Right below you can find elaborate explainers on a selected few.

[`builtin_attrs`]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_feature/src/builtin_attrs.rs
[`def_path_str`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.def_path_str
[`inferred_outlives_of`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.inferred_outlives_of
[`item_bounds`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.item_bounds
[`predicates_of`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.predicates_of
[`symbol_name`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.symbol_name
[object lifetime defaults]: https://doc.rust-lang.org/reference/lifetime-elision.html#default-trait-object-lifetimes
[opaq]: ./opaque-types-impl-trait-inference.md
[variances]: ./variance.md

### Formatting Graphviz output (.dot files)
[formatting-graphviz-output]: #formatting-graphviz-output

Some compiler options for debugging specific features yield graphviz graphs -
e.g. the `#[rustc_mir(borrowck_graphviz_postflow="suffix.dot")]` attribute
on a function dumps various borrow-checker dataflow graphs in conjunction with
`-Zdump-mir-dataflow`.

These all produce `.dot` files. To view these files, install graphviz (e.g.
`apt-get install graphviz`) and then run the following commands:

```bash
$ dot -T pdf maybe_init_suffix.dot > maybe_init_suffix.pdf
$ firefox maybe_init_suffix.pdf # Or your favorite pdf viewer
```

### Debugging type layouts

The internal attribute `#[rustc_layout]` can be used to dump the [`Layout`] of
the type it is attached to. For example:

```rust
#![feature(rustc_attrs)]

#[rustc_layout(debug)]
type T<'a> = &'a u32;
```

Will emit the following:

```text
error: layout_of(&'a u32) = Layout {
    fields: Primitive,
    variants: Single {
        index: 0,
    },
    abi: Scalar(
        Scalar {
            value: Pointer,
            valid_range: 1..=18446744073709551615,
        },
    ),
    largest_niche: Some(
        Niche {
            offset: Size {
                raw: 0,
            },
            scalar: Scalar {
                value: Pointer,
                valid_range: 1..=18446744073709551615,
            },
        },
    ),
    align: AbiAndPrefAlign {
        abi: Align {
            pow2: 3,
        },
        pref: Align {
            pow2: 3,
        },
    },
    size: Size {
        raw: 8,
    },
}
 --> src/lib.rs:4:1
  |
4 | type T<'a> = &'a u32;
  | ^^^^^^^^^^^^^^^^^^^^^

error: aborting due to previous error
```

[`Layout`]: https://doc.rust-lang.org/nightly/nightly-rustc/stable_mir/abi/struct.Layout.html


## Configuring CodeLLDB for debugging `rustc`

If you are using VSCode, and have edited your `bootstrap.toml` to request debugging
level 1 or 2 for the parts of the code you're interested in, then you should be
able to use the [CodeLLDB] extension in VSCode to debug it.

Here is a sample `launch.json` file, being used to run a stage 1 compiler direct
from the directory where it is built (does not have to be "installed"):

```javascript
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
      {
        "type": "lldb",
        "request": "launch",
        "name": "Launch",
        "args": [],  // array of string command-line arguments to pass to compiler
        "program": "${workspaceFolder}/build/host/stage1/bin/rustc",
        "windows": {  // applicable if using windows
            "program": "${workspaceFolder}/build/host/stage1/bin/rustc.exe"
        },
        "cwd": "${workspaceFolder}",  // current working directory at program start
        "stopOnEntry": false,
        "sourceLanguages": ["rust"]
      }
    ]
  }
```

[CodeLLDB]: https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb
