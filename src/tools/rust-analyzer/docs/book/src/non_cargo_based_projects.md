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
    /// https://github.com/rust-lang/rust/tree/master/library.
    ///
    /// If provided, rust-analyzer automatically adds
    /// dependencies on sysroot crates. Conversely,
    /// if you omit this path, you can specify sysroot
    /// dependencies yourself and, for example, have
    /// several different "sysroots" in one graph of
    /// crates.
    sysroot_src?: string;
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
    /// Used to decide what code lens to offer.
    ///
    /// `testOne`: This runnable will be used when the user clicks the 'Run Test'
    /// CodeLens above a test.
    ///
    /// The args for testOne can contain two template strings:
    /// `{label}` and `{test_id}`. `{label}` will be replaced
    /// with the `Build::label` and `{test_id}` will be replaced
    /// with the test name.
    kind: 'testOne' | string;
}
```

This format is provisional and subject to change. Specifically, the
`roots` setup will be different eventually.

There are three ways to feed `rust-project.json` to rust-analyzer:

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

Note that calls to `cargo check` are disabled when using
`rust-project.json` by default, so compilation errors and warnings will
no longer be sent to your LSP client. To enable these compilation errors
you will need to specify explicitly what command rust-analyzer should
run to perform the checks using the
`rust-analyzer.check.overrideCommand` configuration. As an example, the
following configuration explicitly sets `cargo check` as the `check`
command.

    { "rust-analyzer.check.overrideCommand": ["cargo", "check", "--message-format=json"] }

`check.overrideCommand` requires the command specified to output json
error messages for rust-analyzer to consume. The `--message-format=json`
flag does this for `cargo check` so whichever command you use must also
output errors in this format. See the [Configuration](#_configuration)
section for more information.
