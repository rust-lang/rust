# Debugging bootstrap

There are two main ways to debug bootstrap itself. The first is through println logging, and the second is through the `tracing` feature.

> FIXME: this section should be expanded

## `println` logging

Bootstrap has extensive unstructured logging. Most of it is gated behind the `--verbose` flag (pass `-vv` for even more detail).

If you want to know which `Step` ran a command, you could invoke bootstrap like so:

```
$ ./x dist rustc --dry-run -vv
learning about cargo
running: RUSTC_BOOTSTRAP="1" "/home/jyn/src/rust2/build/x86_64-unknown-linux-gnu/stage0/bin/cargo" "metadata" "--format-version" "1" "--no-deps" "--manifest-path" "/home/jyn/src/rust2/Cargo.toml" (failure_mode=Exit) (created at src/bootstrap/src/core/metadata.rs:81:25, executed at src/bootstrap/src/core/metadata.rs:92:50)
running: RUSTC_BOOTSTRAP="1" "/home/jyn/src/rust2/build/x86_64-unknown-linux-gnu/stage0/bin/cargo" "metadata" "--format-version" "1" "--no-deps" "--manifest-path" "/home/jyn/src/rust2/library/Cargo.toml" (failure_mode=Exit) (created at src/bootstrap/src/core/metadata.rs:81:25, executed at src/bootstrap/src/core/metadata.rs:92:50)
> Assemble { target_compiler: Compiler { stage: 1, host: x86_64-unknown-linux-gnu } }
  > Libdir { compiler: Compiler { stage: 1, host: x86_64-unknown-linux-gnu }, target: x86_64-unknown-linux-gnu }
    > Sysroot { compiler: Compiler { stage: 1, host: x86_64-unknown-linux-gnu }, force_recompile: false }
Removing sysroot /home/jyn/src/rust2/build/tmp-dry-run/x86_64-unknown-linux-gnu/stage1 to avoid caching bugs
    < Sysroot { compiler: Compiler { stage: 1, host: x86_64-unknown-linux-gnu }, force_recompile: false }
  < Libdir { compiler: Compiler { stage: 1, host: x86_64-unknown-linux-gnu }, target: x86_64-unknown-linux-gnu }
...
```

This will go through all the recursive dependency calculations, where `Step`s internally call `builder.ensure()`, without actually running cargo or the compiler.

In some cases, even this may not be enough logging (if so, please add more!). In that case, you can omit `--dry-run`, which will show the normal output inline with the debug logging:

```
      c Sysroot { compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu }, force_recompile: false }
using sysroot /home/jyn/src/rust2/build/x86_64-unknown-linux-gnu/stage0-sysroot
Building stage0 library artifacts (x86_64-unknown-linux-gnu)
running: cd "/home/jyn/src/rust2" && env ... RUSTC_VERBOSE="2" RUSTC_WRAPPER="/home/jyn/src/rust2/build/bootstrap/debug/rustc" "/home/jyn/src/rust2/build/x86_64-unknown-linux-gnu/stage0/bin/cargo" "build" "--target" "x86_64-unknown-linux-gnu" "-Zbinary-dep-depinfo" "-Zroot-dir=/home/jyn/src/rust2" "-v" "-v" "--manifest-path" "/home/jyn/src/rust2/library/sysroot/Cargo.toml" "--message-format" "json-render-diagnostics"
   0.293440230s  INFO prepare_target{force=false package_id=sysroot v0.0.0 (/home/jyn/src/rust2/library/sysroot) target="sysroot"}: cargo::core::compiler::fingerprint: fingerprint error for sysroot v0.0.0 (/home/jyn/src/rust2/library/sysroot)/Build/TargetInner { name_inferred: true, ..: lib_target("sysroot", ["lib"], "/home/jyn/src/rust2/library/sysroot/src/lib.rs", Edition2021) }
...
```

In most cases this should not be necessary.

TODO: we should convert all this to structured logging so it's easier to control precisely.

## `tracing` in bootstrap

Bootstrap has conditional [`tracing`][tracing] setup to provide structured logging.

[tracing]: https://docs.rs/tracing/0.1.41/tracing/index.html

### Enabling `tracing` output

Bootstrap will conditionally build `tracing` support and enable `tracing` output if the `BOOTSTRAP_TRACING` env var is set.

#### Basic usage

Example basic usage[^just-trace]:

[^just-trace]: It is not recommended to use *just* `BOOTSTRAP_TRACING=TRACE` because that will dump *everything* at `TRACE` level, including logs intentionally gated behind custom targets as they are too verbose even for `TRACE` level by default.

```bash
$ BOOTSTRAP_TRACING=bootstrap=TRACE ./x build library --stage 1
```

Example output[^unstable]:

```
$ BOOTSTRAP_TRACING=bootstrap=TRACE ./x check src/bootstrap/
Building bootstrap
   Compiling bootstrap v0.0.0 (/home/joe/repos/rust/src/bootstrap)
    Finished `dev` profile [unoptimized] target(s) in 2.74s
 DEBUG bootstrap parsing flags
 bootstrap::core::config::flags::Flags::parse args=["check", "src/bootstrap/"]
 DEBUG bootstrap parsing config based on flags
 DEBUG bootstrap creating new build based on config
 bootstrap::Build::build
  TRACE bootstrap setting up job management
  TRACE bootstrap downloading rustfmt early
   bootstrap::handling hardcoded subcommands (Format, Suggest, Perf)
    DEBUG bootstrap not a hardcoded subcommand; returning to normal handling, cmd=Check { all_targets: false }
  DEBUG bootstrap handling subcommand normally
   bootstrap::executing real run
     bootstrap::(1) executing dry-run sanity-check
     bootstrap::(2) executing actual run
Checking stage0 library artifacts (x86_64-unknown-linux-gnu)
    Finished `release` profile [optimized + debuginfo] target(s) in 0.04s
Checking stage0 compiler artifacts {rustc-main, rustc_abi, rustc_arena, rustc_ast, rustc_ast_ir, rustc_ast_lowering, rustc_ast_passes, rustc_ast_pretty, rustc_attr_data_structures, rustc_attr_parsing, rustc_baked_icu_data, rustc_borrowck, rustc_builtin_macros, rustc_codegen_llvm, rustc_codegen_ssa, rustc_const_eval, rustc_data_structures, rustc_driver, rustc_driver_impl, rustc_error_codes, rustc_error_messages, rustc_errors, rustc_expand, rustc_feature, rustc_fluent_macro, rustc_fs_util, rustc_graphviz, rustc_hir, rustc_hir_analysis, rustc_hir_pretty, rustc_hir_typeck, rustc_incremental, rustc_index, rustc_index_macros, rustc_infer, rustc_interface, rustc_lexer, rustc_lint, rustc_lint_defs, rustc_llvm, rustc_log, rustc_macros, rustc_metadata, rustc_middle, rustc_mir_build, rustc_mir_dataflow, rustc_mir_transform, rustc_monomorphize, rustc_next_trait_solver, rustc_parse, rustc_parse_format, rustc_passes, rustc_pattern_analysis, rustc_privacy, rustc_query_impl, rustc_query_system, rustc_resolve, rustc_sanitizers, rustc_serialize, rustc_session, rustc_smir, rustc_span, rustc_symbol_mangling, rustc_target, rustc_trait_selection, rustc_traits, rustc_transmute, rustc_ty_utils, rustc_type_ir, rustc_type_ir_macros, stable_mir} (x86_64-unknown-linux-gnu)
    Finished `release` profile [optimized + debuginfo] target(s) in 0.23s
Checking stage0 bootstrap artifacts (x86_64-unknown-linux-gnu)
    Checking bootstrap v0.0.0 (/home/joe/repos/rust/src/bootstrap)
    Finished `release` profile [optimized + debuginfo] target(s) in 0.64s
  DEBUG bootstrap checking for postponed test failures from `test  --no-fail-fast`
Build completed successfully in 0:00:08
```

#### Controlling tracing output

The env var `BOOTSTRAP_TRACING` accepts a [`tracing` env-filter][tracing-env-filter].

There are two orthogonal ways to control which kind of tracing logs you want:

1. You can specify the log **level**, e.g. `DEBUG` or `TRACE`.
2. You can also control the log **target**, e.g. `bootstrap` or `bootstrap::core::config` vs custom targets like `CONFIG_HANDLING`.
    - Custom targets are used to limit what is output when `BOOTSTRAP_TRACING=bootstrap=TRACE` is used, as they can be too verbose even for `TRACE` level by default. Currently used custom targets:
        - `CONFIG_HANDLING`

The `TRACE` filter will enable *all* `trace` level or less verbose level tracing output.

You can of course combine them (custom target logs are typically gated behind `TRACE` log level additionally):

```bash
$ BOOTSTRAP_TRACING=CONFIG_HANDLING=TRACE ./x build library --stage 1
```

[^unstable]: This output is always subject to further changes.

[tracing-env-filter]: https://docs.rs/tracing-subscriber/0.3.19/tracing_subscriber/filter/struct.EnvFilter.html

##### FIXME(#96176): specific tracing for `compiler()` vs `compiler_for()`

The additional targets `COMPILER` and `COMPILER_FOR` are used to help trace what
`builder.compiler()` and `builder.compiler_for()` does. They should be removed
if [#96176][cleanup-compiler-for] is resolved.

[cleanup-compiler-for]: https://github.com/rust-lang/rust/issues/96176

### Using `tracing` in bootstrap

Both `tracing::*` macros and the `tracing::instrument` proc-macro attribute need to be gated behind `tracing` feature. Examples:

```rs
#[cfg(feature = "tracing")]
use tracing::instrument;

struct Foo;

impl Step for Foo {
    type Output = ();

    #[cfg_attr(feature = "tracing", instrument(level = "trace", name = "Foo::should_run", skip_all))]
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        trace!(?run, "entered Foo::should_run");

        todo!()
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "Foo::run",
            skip_all,
            fields(compiler = ?builder.compiler),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        trace!(?run, "entered Foo::run");

        todo!()
    }    
}
```

For `#[instrument]`, it's recommended to:

- Gate it behind `trace` level for fine-granularity, possibly `debug` level for core functions.
- Explicitly pick an instrumentation name via `name = ".."` to distinguish between e.g. `run` of different steps.
- Take care to not cause diverging behavior via tracing, e.g. building extra things only when tracing infra is enabled.

### Profiling bootstrap

You can use the `COMMAND` tracing target to trace execution of most commands spawned by bootstrap. If you also use the `BOOTSTRAP_PROFILE=1` environment variable, bootstrap will generate a Chrome JSON trace file, which can be visualized in Chrome's `chrome://tracing` page or on https://ui.perfetto.dev.

```bash
$ BOOTSTRAP_TRACING=COMMAND=trace BOOTSTRAP_PROFILE=1 ./x build library
```

### rust-analyzer integration?

Unfortunately, because bootstrap is a `rust-analyzer.linkedProjects`, you can't ask r-a to check/build bootstrap itself with `tracing` feature enabled to get relevant completions, due to lack of support as described in <https://github.com/rust-lang/rust-analyzer/issues/8521>.
