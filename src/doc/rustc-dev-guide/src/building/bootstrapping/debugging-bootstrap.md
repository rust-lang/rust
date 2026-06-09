# Debugging bootstrap

There are two main ways of debugging (and profiling bootstrap). The first is through println logging, and the second is through the `tracing` feature.

## `println` logging

Bootstrap has extensive unstructured logging. Most of it is gated behind the `--verbose` flag (pass `-vv` for even more detail).

If you want to see verbose output of executed Cargo commands and other kinds of detailed logs, pass `-v` or `-vv` when invoking bootstrap. Note that the logs are unstructured and may be overwhelming.

```
$ ./x dist rustc --dry-run -vv
learning about cargo
running: RUSTC_BOOTSTRAP="1" "/home/jyn/src/rust2/build/x86_64-unknown-linux-gnu/stage0/bin/cargo" "metadata" "--format-version" "1" "--no-deps" "--manifest-path" "/home/jyn/src/rust2/Cargo.toml" (failure_mode=Exit) (created at src/bootstrap/src/core/metadata.rs:81:25, executed at src/bootstrap/src/core/metadata.rs:92:50)
running: RUSTC_BOOTSTRAP="1" "/home/jyn/src/rust2/build/x86_64-unknown-linux-gnu/stage0/bin/cargo" "metadata" "--format-version" "1" "--no-deps" "--manifest-path" "/home/jyn/src/rust2/library/Cargo.toml" (failure_mode=Exit) (created at src/bootstrap/src/core/metadata.rs:81:25, executed at src/bootstrap/src/core/metadata.rs:92:50)
...
```

## `tracing` in bootstrap

Bootstrap has a conditional `tracing` feature, which provides the following features:
- It enables structured logging using [`tracing`][tracing] events and spans.
- It generates a [Chrome trace file] that can be used to visualize the hierarchy and durations of executed steps and commands.
  - You can open the generated `chrome-trace.json` file using Chrome, on the `chrome://tracing` tab, or e.g. using [Perfetto].
- It generates [GraphViz] graphs that visualize the dependencies between executed steps.
  - You can open the generated `step-graph-*.dot` file using e.g. [xdot] to visualize the step graph, or use e.g. `dot -Tsvg` to convert the GraphViz file to an SVG file.
- It generates a command execution summary, which shows which commands were executed, how many of their executions were cached, and what commands were the slowest to run.
  - The generated `command-stats.txt` file is in a simple human-readable format.

The structured logs will be written to standard error output (`stderr`), while the other outputs will be stored in files in the `<build-dir>/bootstrap-trace/<pid>` directory. For convenience, bootstrap will also create a symlink to the latest generated trace output directory at `<build-dir>/bootstrap-trace/latest`.

> Note that if you execute bootstrap with `--dry-run`, the tracing output directory might change. Bootstrap will always print a path where the tracing output files were stored at the end of its execution.

[tracing]: https://docs.rs/tracing/0.1.41/tracing/index.html
[Chrome trace file]: https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/
[Perfetto]: https://ui.perfetto.dev/
[GraphViz]: https://graphviz.org/doc/info/lang.html
[xdot]: https://github.com/jrfonseca/xdot.py

### Enabling `tracing` output

To enable the conditional `tracing` feature, run bootstrap with the `BOOTSTRAP_TRACING` environment variable.

[tracing_subscriber filter]: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html

```bash
$ BOOTSTRAP_TRACING=trace ./x build library --stage 1
```

Example output[^unstable]:

```
$ BOOTSTRAP_TRACING=trace ./x build library --stage 1 --dry-run
Building bootstrap
    Finished `dev` profile [unoptimized] target(s) in 0.05s
15:56:52.477  INFO > tool::LibcxxVersionTool {target: x86_64-unknown-linux-gnu} (builder/mod.rs:1715)
15:56:52.575  INFO > compile::Assemble {target_compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }} (builder/mod.rs:1715)
15:56:52.575  INFO > tool::Compiletest {compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, target: x86_64-unknown-linux-gnu} (builder/mod.rs:1715)
15:56:52.576  INFO  > tool::ToolBuild {build_compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, target: x86_64-unknown-linux-gnu, tool: "compiletest", path: "src/tools/compiletest", mode: ToolBootstrap, source_type: InTree, extra_features: [], allow_features: "internal_output_capture", cargo_args: [], artifact_kind: Binary} (builder/mod.rs:1715)
15:56:52.576  INFO   > builder::Libdir {compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, target: x86_64-unknown-linux-gnu} (builder/mod.rs:1715)
15:56:52.576  INFO    > compile::Sysroot {compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, force_recompile: false} (builder/mod.rs:1715)
15:56:52.578  INFO > compile::Assemble {target_compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }} (builder/mod.rs:1715)
15:56:52.578  INFO > tool::Compiletest {compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, target: x86_64-unknown-linux-gnu} (builder/mod.rs:1715)
15:56:52.578  INFO  > tool::ToolBuild {build_compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, target: x86_64-unknown-linux-gnu, tool: "compiletest", path: "src/tools/compiletest", mode: ToolBootstrap, source_type: InTree, extra_features: [], allow_features: "internal_output_capture", cargo_args: [], artifact_kind: Binary} (builder/mod.rs:1715)
15:56:52.578  INFO   > builder::Libdir {compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, target: x86_64-unknown-linux-gnu} (builder/mod.rs:1715)
15:56:52.578  INFO    > compile::Sysroot {compiler: Compiler { stage: 0, host: x86_64-unknown-linux-gnu, forced_compiler: false }, force_recompile: false} (builder/mod.rs:1715)
    Finished `release` profile [optimized] target(s) in 0.11s
Tracing/profiling output has been written to <src-root>/build/bootstrap-trace/latest
Build completed successfully in 0:00:00
```

[^unstable]: This output is always subject to further changes.

#### Controlling tracing output

The environment variable `BOOTSTRAP_TRACING` accepts a [`tracing_subscriber` filter][tracing-env-filter]. If you set `BOOTSTRAP_TRACING=trace`, you will enable all logs, but that can be overwhelming. You can thus use the filter to reduce the amount of data logged.

There are two orthogonal ways to control which kind of tracing logs you want:

1. You can specify the log **level**, e.g. `debug` or `trace`.
   - If you select a level, all events/spans with an equal or higher priority level will be shown.
2. You can also control the log **target**, e.g. `bootstrap` or `bootstrap::core::config` or a custom target like `CONFIG_HANDLING` or `STEP`.
    - Custom targets are used to limit what kinds of spans you are interested in, as the `BOOTSTRAP_TRACING=trace` output can be quite verbose. Currently, you can use the following custom targets:
        - `CONFIG_HANDLING`: show spans related to config handling.
        - `STEP`: show all executed steps. Executed commands have `info` event level.
        - `COMMAND`: show all executed commands. Executed commands have `trace` event level.
        - `IO`: show performed I/O operations. Executed commands have `trace` event level.
            - Note that many I/O are currently not being traced.

You can of course combine them (custom target logs are typically gated behind `TRACE` log level additionally):

```bash
$ BOOTSTRAP_TRACING=CONFIG_HANDLING=trace,STEP=info,COMMAND=trace ./x build library --stage 1
```

[tracing-env-filter]: https://docs.rs/tracing-subscriber/0.3.19/tracing_subscriber/filter/struct.EnvFilter.html

Note that the level that you specify using `BOOTSTRAP_TRACING` also has an effect on the spans that will be recorded in the Chrome trace file.

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

### rust-analyzer integration?

Unfortunately, because bootstrap is a `rust-analyzer.linkedProjects`, you can't ask r-a to check/build bootstrap itself with `tracing` feature enabled to get relevant completions, due to lack of support as described in <https://github.com/rust-lang/rust-analyzer/issues/8521>.
