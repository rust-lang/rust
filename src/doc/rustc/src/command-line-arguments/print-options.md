# Print Options

All of these options are passed to `rustc` via the `--print` flag.

Those options prints out various information about the compiler. Multiple options can be
specified, and the information is printed in the order the options are specified.

Specifying an option will usually disable the [`--emit`](../command-line-arguments.md#option-emit)
step and will only print the requested information.

A filepath may optionally be specified for each requested information kind, in the format
`--print KIND=PATH`, just like for `--emit`. When a path is specified, information will be
written there instead of to stdout.

## `crate-name`

The name of the crate.

Generally coming from either from the `#![crate_name = "..."]` attribute,
[`--crate-name` flag](../command-line-arguments.md#option-crate-name) or the filename.

Example:

```bash
$ rustc --print crate-name --crate-name my_crate a.rs
my_crate
```

## `file-names`

The names of the files created by the `link` emit kind.

## `sysroot`

Absolute path to the sysroot.

Example (with rustup and the stable toolchain):

```bash
$ rustc --print sysroot a.rs
/home/[REDACTED]/.rustup/toolchains/stable-x86_64-unknown-linux-gnu
```

## `target-libdir`

Path to the target libdir.

Example (with rustup and the stable toolchain):

```bash
$ rustc --print target-libdir a.rs
/home/[REDACTED]/.rustup/toolchains/beta-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib
```

## `host-tuple`

The target-tuple string of the host compiler.

Example:

```bash
$ rustc --print host-tuple a.rs
x86_64-unknown-linux-gnu
```

Example with the `--target` flag:

```bash
$ rustc --print host-tuple --target "armv7-unknown-linux-gnueabihf" a.rs
x86_64-unknown-linux-gnu
```

## `cfg`

List of cfg values. See [conditional compilation] for more information about cfg values.

Example (for `x86_64-unknown-linux-gnu`):

```bash
$ rustc --print cfg a.rs
debug_assertions
panic="unwind"
target_abi=""
target_arch="x86_64"
target_endian="little"
target_env="gnu"
target_family="unix"
target_feature="fxsr"
target_feature="sse"
target_feature="sse2"
target_has_atomic="16"
target_has_atomic="32"
target_has_atomic="64"
target_has_atomic="8"
target_has_atomic="ptr"
target_os="linux"
target_pointer_width="64"
target_vendor="unknown"
unix
```

## `target-list`

List of known targets. The target may be selected with the `--target` flag.

## `target-cpus`

List of available CPU values for the current target. The target CPU may be selected with
the [`-C target-cpu=val` flag](../codegen-options/index.md#target-cpu).

## `target-features`

List of available target features for the *current target*.

Target features may be enabled with the **unsafe**
[`-C target-feature=val` flag](../codegen-options/index.md#target-feature).

See [known issues](../targets/known-issues.md) for more details.

## `relocation-models`

List of relocation models. Relocation models may be selected with the
[`-C relocation-model=val` flag](../codegen-options/index.md#relocation-model).

Example:

```bash
$ rustc --print relocation-models a.rs
Available relocation models:
    static
    pic
    pie
    dynamic-no-pic
    ropi
    rwpi
    ropi-rwpi
    default
```

## `code-models`

List of code models. Code models may be selected with the
[`-C code-model=val` flag](../codegen-options/index.md#code-model).

Example:

```bash
$ rustc --print code-models a.rs
Available code models:
    tiny
    small
    kernel
    medium
    large
```

## `tls-models`

List of Thread Local Storage models supported. The model may be selected with the
`-Z tls-model=val` flag.

Example:

```bash
$ rustc --print tls-models a.rs
Available TLS models:
    global-dynamic
    local-dynamic
    initial-exec
    local-exec
    emulated
```

## `native-static-libs`

This may be used when creating a `staticlib` crate type.

If this is the only flag, it will perform a full compilation and include a diagnostic note
that indicates the linker flags to use when linking the resulting static library.

The note starts with the text `native-static-libs:` to make it easier to fetch the output.

Example:

```bash
$ rustc --print native-static-libs --crate-type staticlib a.rs
note: link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.

note: native-static-libs: -lgcc_s -lutil [REDACTED] -lpthread -lm -ldl -lc
```

## `link-args`

This flag does not disable the `--emit` step. This can be useful when debugging linker options.

When linking, this flag causes `rustc` to print the full linker invocation in a human-readable
form. The exact format of this debugging output is not a stable guarantee, other than that it
will include the linker executable and the text of each command-line argument passed to the
linker.

## `deployment-target`

The currently selected [deployment target] (or minimum OS version) for the selected Apple
platform target.

This value can be used or passed along to other components alongside a Rust build that need
this information, such as C compilers. This returns rustc's minimum supported deployment target
if no `*_DEPLOYMENT_TARGET` variable is present in the environment, or otherwise returns the
variable's parsed value.

[conditional compilation]: ../../reference/conditional-compilation.html
[deployment target]: https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/cross_development/Configuring/configuring.html
