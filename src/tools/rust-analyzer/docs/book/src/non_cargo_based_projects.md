# Non-Cargo Based Projects

rust-analyzer does not require Cargo. However, if you use some other
build system, you’ll have to describe the structure of your project for
rust-analyzer in the `rust-project.json` format:

```typescript
interface ProjectJson {
    /// Path to the sysroot directory.
    ///
    /// The sysroot is where rustc looks for the
    /// crates that are built-in to rust, such as
    /// std.
    ///
    /// https://doc.rust-lang.org/rustc/command-line-arguments.html#--sysroot-override-the-system-root
    ///
    /// To see the current value of sysroot, you
    /// can query rustc:
    ///
    /// ```
    /// $ rustc --print sysroot
    /// /Users/yourname/.rustup/toolchains/stable-x86_64-apple-darwin
    /// ```
    sysroot?: string;
    /// Path to the directory with *source code* of
    /// sysroot crates.
    ///
    /// By default, this is `lib/rustlib/src/rust/library`
    /// relative to the sysroot.
    ///
    /// It should point to the directory where std,
    /// core, and friends can be found:
    ///
    /// https://github.com/rust-lang/rust/tree/HEAD/library.
    ///
    /// If provided, rust-analyzer automatically adds
    /// dependencies on sysroot crates. Conversely,
    /// if you omit this path, you can specify sysroot
    /// dependencies yourself and, for example, have
    /// several different "sysroots" in one graph of
    /// crates.
    sysroot_src?: string;
    /// A ProjectJson describing the crates of the sysroot.
    sysroot_project?: ProjectJson;

    /// List of groups of common cfg values, to allow
    /// sharing them between crates.
    ///
    /// Maps from group name to its cfgs. Cfg follow
    /// the same format as `Crate.cfg`.
    cfg_groups?: { [key: string]: string[]; };
    /// The set of crates comprising the current
    /// project. Must include all transitive
    /// dependencies as well as sysroot crate (libstd,
    /// libcore and such).
    crates: Crate[];
    /// Configuration for CLI commands.
    ///
    /// These are used for running and debugging binaries
    /// and tests without encoding build system-specific
    /// knowledge into rust-analyzer.
    ///
    /// # Example
    ///
    /// Below is an example of a test runnable. `{label}` and `{test_id}`
    /// are explained in `Runnable::args`'s documentation below.
    ///
    /// ```json
    /// {
    ///     "program": "buck",
    ///     "args": [
    ///         "test",
    ///          "{label}",
    ///          "--",
    ///          "{test_id}",
    ///          "--print-passing-details"
    ///     ],
    ///     "cwd": "/home/user/repo-root/",
    ///     "kind": "testOne"
    /// }
    /// ```
    runnables?: Runnable[];
}

interface Crate {
    /// Optional crate name used for display purposes,
    /// without affecting semantics. See the `deps`
    /// key for semantically-significant crate names.
    display_name?: string;
    /// Path to the root module of the crate.
    root_module: string;
    /// Edition of the crate.
    edition: '2015' | '2018' | '2021' | '2024';
    /// The version of the crate. Used for calculating
    /// the correct docs.rs URL.
    version?: string;
    /// Dependencies
    deps: Dep[];
    /// Should this crate be treated as a member of
    /// current "workspace".
    ///
    /// By default, inferred from the `root_module`
    /// (members are the crates which reside inside
    /// the directory opened in the editor).
    ///
    /// Set this to `false` for things like standard
    /// library and 3rd party crates to enable
    /// performance optimizations (rust-analyzer
    /// assumes that non-member crates don't change).
    is_workspace_member?: boolean;
    /// Optionally specify the (super)set of `.rs`
    /// files comprising this crate.
    ///
    /// By default, rust-analyzer assumes that only
    /// files under `root_module.parent` can belong
    /// to a crate. `include_dirs` are included
    /// recursively, unless a subdirectory is in
    /// `exclude_dirs`.
    ///
    /// Different crates can share the same `source`.
    ///
    /// If two crates share an `.rs` file in common,
    /// they *must* have the same `source`.
    /// rust-analyzer assumes that files from one
    /// source can't refer to files in another source.
    source?: {
        include_dirs: string[];
        exclude_dirs: string[];
    };
    /// List of cfg groups this crate inherits.
    ///
    /// All cfg in these groups will be concatenated to
    /// `cfg`. It is impossible to replace a value from
    /// the groups.
    cfg_groups?: string[];
    /// The set of cfgs activated for a given crate, like
    /// `["unix", "feature=\"foo\"", "feature=\"bar\""]`.
    cfg: string[];
    /// Target tuple for this Crate.
    ///
    /// Used when running `rustc --print cfg`
    /// to get target-specific cfgs.
    target?: string;
    /// Environment variables, used for
    /// the `env!` macro
    env: { [key: string]: string; };
    /// Extra crate-level attributes applied to this crate.
    ///
    /// rust-analyzer will behave as if these attributes
    /// were present before the first source line of the
    /// crate root.
    ///
    /// Each string should contain the contents of a `#![...]`
    /// crate-level attribute, without the surrounding `#![]`.
    crate_attrs?: string[];

    /// Whether the crate is a proc-macro crate.
    is_proc_macro: boolean;
    /// For proc-macro crates, path to compiled
    /// proc-macro (.so file).
    proc_macro_dylib_path?: string;

    /// Repository, matching the URL that would be used
    /// in Cargo.toml.
    repository?: string;

    /// Build-specific data about this crate.
    build?: BuildInfo;
}

interface Dep {
    /// Index of a crate in the `crates` array.
    crate: number;
    /// Name as should appear in the (implicit)
    /// `extern crate name` declaration.
    name: string;
}

interface BuildInfo {
    /// The name associated with this crate.
    ///
    /// This is determined by the build system that produced
    /// the `rust-project.json` in question. For instance, if buck were used,
    /// the label might be something like `//ide/rust/rust-analyzer:rust-analyzer`.
    ///
    /// Do not attempt to parse the contents of this string; it is a build system-specific
    /// identifier similar to `Crate::display_name`.
    label: string;
    /// Path corresponding to the build system-specific file defining the crate.
    build_file: string;
    /// The kind of target.
    ///
    /// This information is used to determine what sort
    /// of runnable codelens to provide, if any.
    target_kind: 'bin' | 'lib' | 'test';
}

interface Runnable {
    /// The program invoked by the runnable.
    ///
    /// For example, this might be `cargo`, `buck`, or `bazel`.
    program: string;
    /// The arguments passed to `program`.
    args: string[];
    /// The current working directory of the runnable.
    cwd: string;
    /// Maps a runnable to a piece of rust-analyzer functionality.
    ///
    /// - `testOne`: This runnable will be used when the user clicks the 'Run Test'
    ///   CodeLens above a test.
    /// - `run`: This runnable will be used when the user clicks the 'Run' CodeLens
    ///    above a main function or triggers a run command.
    /// - `flycheck`: This is run to provide check-on-save diagnostics when the user
    ///    saves a file. It must emit rustc JSON diagnostics that rust-analyzer can
    ///    parse. If this runnable is not specified, we may try to use `cargo check -p`.
    ///    This is only run for a single crate that the user saved a file in. The
    ///    {label} syntax is replaced with `BuildInfo::label`.
    ///    Alternatively, you may use `{saved_file}` and figure out which crate
    ///    to produce diagnostics for based on that.
    ///
    /// The args for testOne can contain two template strings:
    /// `{label}` and `{test_id}`. `{label}` will be replaced
    /// with the `BuildInfo::label` and `{test_id}` will be replaced
    /// with the test name.
    kind: 'testOne' | 'run' | 'flycheck' | string;
}
```

This format is provisional and subject to change. Specifically, the
`roots` setup will be different eventually.

### Providing a JSON project to rust-analyzer

There are four ways to feed `rust-project.json` to rust-analyzer:

-   Use
    [`"rust-analyzer.workspace.discoverConfig": … }`](./configuration.md#workspace.discoverConfig)
    to specify a workspace discovery command to generate project descriptions
    on-the-fly. Please note that the command output is message-oriented and must
    output JSONL [as described in the configuration docs](./configuration.md#workspace.discoverConfig).

-   Place `rust-project.json` file at the root of the project, and
    rust-analyzer will discover it.

-   Specify
    `"rust-analyzer.linkedProjects": [ "path/to/rust-project.json" ]` in
    the settings (and make sure that your LSP client sends settings as a
    part of initialize request).

-   Specify
    `"rust-analyzer.linkedProjects": [ { "roots": […​], "crates": […​] }]`
    inline.

Relative paths are interpreted relative to `rust-project.json` file
location or (for inline JSON) relative to `rootUri`.

You can set the `RA_LOG` environment variable to `rust_analyzer=info` to
inspect how rust-analyzer handles config and project loading.

### Flycheck support

Rust-analyzer has functionality to run an actual build of a crate when the user saves a file, to
fill in diagnostics it does not implement natively. This is known as "flycheck".

**Flycheck is disabled when using `rust-project.json` unless explicitly configured**, so compilation
errors and warnings will no longer be sent to your LSP client by default. To enable these
compilation errors you will need to specify explicitly what command rust-analyzer should run to
perform the checks. There are two ways to do this:

- `rust-project.json` may contain a `runnables` field. The `flycheck` runnable may be used to
  configure a check command. See above for documentation.

- Using the [`rust-analyzer.check.overrideCommand`](./configuration.md#check.overrideCommand)
  configuration. This will also override anything in `rust-project.json`. As an example, the
  following configuration explicitly sets `cargo check` as the `check` command.

  ```json
  { "rust-analyzer.check.overrideCommand": ["cargo", "check", "--message-format=json"] }
  ```

  Note also that this works with cargo projects.

Either option requires the command specified to output JSON error messages for rust-analyzer to
consume. The `--message-format=json` flag does this for `cargo check` so whichever command you use
must also output errors in this format.

Either option also supports two syntaxes within each argument:

- `{label}` will be replaced with the `BuildInfo::label` of the crate
  containing a saved file, if `BuildInfo` is provided. In the case of `check.overrideCommand` being
  used in a Cargo project, this will be the cargo package ID, which can be used with `cargo check -p`.
- `{saved_file}` will be replaced with an absolute path to the saved file. This can be queried against a
  build system to find targets that include the file.

For example:

```json
{ "rust-analyzer.check.overrideCommand": ["custom_crate_checker", "{label}"] }
```

If you do use `{label}` or `{saved_file}`, the command will not be run unless the relevant value can
be substituted.


#### Flycheck considerations

##### Diagnostic output on error

A flycheck command using a complex build orchestrator like `"bazel", "build", "{label}"`, even with
a tweak to return JSON messages, is often insufficient. Such a command will typically succeed if
there are warnings, but if there are errors, it might "fail to compile" the diagnostics and not
produce any output. You must build a package in such a way that the build succeeds even if `rustc`
exits with an error, and prints the JSON build messages in every case.

##### Diagnostics for upstream crates

`cargo check -p` re-prints any errors and warnings in crates higher up in the dependency graph
than the one requested. We do clear all diagnostics when flychecking, so if you manage to
replicate this behaviour, diagnostics for crates other than the one being checked will show up in
the editor. If you do not, then users may be confused that diagnostics are "stuck" or disappear
entirely when there is a build error in an upstream crate.

##### Compiler options

`cargo check` invokes rustc differently from `cargo build`. It turns off codegen (with `rustc
--emit=metadata`), which results in lower latency to get to diagnostics. If your build system can
configure this, it is recommended.

If your build tool can configure rustc for incremental compiles, this is also recommended.

##### Locking and pre-emption

In any good build system, including Cargo, build commands sometimes block each other. Running a
flycheck will (by default) frequently block you from running other build commands. Generally this is
undesirable. Users will have to (unintuitively) press save again in the editor to cancel a
flycheck, so that some other command may proceed.

If your build system has the ability to isolate any rust-analyzer-driven flychecks and prevent lock
contention, for example a separate build output directory and/or daemon instance, this is
recommended. Alternatively, consider using a feature if available that can set the priority of
various build invocations and automatically cancel lower-priority ones when needed. Flychecks should
be set to a lower priority than general direct build invocations.
