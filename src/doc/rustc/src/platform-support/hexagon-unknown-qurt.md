# `hexagon-unknown-qurt`

**Tier: 3**

Rust for Hexagon QuRT (Qualcomm Real-Time OS).

| Target               | Description |
| -------------------- | ------------|
| hexagon-unknown-qurt | Hexagon 32-bit QuRT |

## Target maintainers

[@androm3da](https://github.com/androm3da)

## Requirements

This target is cross-compiled. There is support for `std`. The target uses
QuRT's standard library and runtime.

By default, code generated with this target should run on Hexagon DSP hardware
running the QuRT real-time operating system.

- `-Ctarget-cpu=hexagonv69` targets Hexagon V69 architecture (default)
- `-Ctarget-cpu=hexagonv73` adds support for instructions defined up to Hexagon V73

Functions marked `extern "C"` use the [Hexagon architecture calling convention](https://lists.llvm.org/pipermail/llvm-dev/attachments/20190916/21516a52/attachment-0001.pdf).

This target generates position-independent ELF binaries by default, making it
suitable for both static images and dynamic shared objects.

The [Hexagon SDK](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK) is
required for building programs for this target.

## Linking

This target selects `rust-lld` by default.  Another option to use is
[eld](https://github.com/qualcomm/eld), which is also provided with
[the opensource hexagon toolchain](https://github.com/quic/toolchain_for_hexagon)
and the Hexagon SDK.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `bootstrap.toml`:

```toml
[build]
build-stage = 1
host = ["<target for your host>"]
target = ["<target for your host>", "hexagon-unknown-qurt"]

[target.hexagon-unknown-qurt]
cc = "hexagon-clang"
cxx = "hexagon-clang++"
ranlib = "llvm-ranlib"
ar = "llvm-ar"
llvm-libunwind = 'in-tree'
```

Replace `<target for your host>` with `x86_64-unknown-linux-gnu` or whatever
else is appropriate for your host machine.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Static Image Targeting

For static executables that run directly on QuRT, use the default target
configuration with additional linker flags:

```sh
# Build a static executable for QuRT
cargo rustc --target hexagon-unknown-qurt -- \
    -C link-args="-static -nostdlib" \
    -C link-args="-L/opt/Hexagon_SDK/6.3.0.0/rtos/qurt/computev69/lib" \
    -C link-args="-lqurt -lc"
```

This approach is suitable for:
- Standalone QuRT applications
- System-level services
- Boot-time initialization code
- Applications that need deterministic memory layout

## User-Loadable Shared Object Targeting

For shared libraries that can be dynamically loaded by QuRT applications:

```sh
# Build a shared object for QuRT
cargo rustc --target hexagon-unknown-qurt \
    --crate-type=cdylib -- \
    -C link-args="-shared -fPIC" \
    -C link-args="-L/opt/Hexagon_SDK/6.3.0.0/rtos/qurt/computev69/lib"
```

This approach is suitable for:
- Plugin architectures
- Runtime-loadable modules
- Libraries shared between multiple applications
- Code that needs to be updated without system restart

## Configuration Options

The target can be customized for different use cases:

### For Static Images
```toml
# In .cargo/config.toml
[target.hexagon-unknown-qurt]
rustflags = [
    "-C", "link-args=-static",
    "-C", "link-args=-nostdlib",
    "-C", "target-feature=-small-data"
]
```

### For Shared Objects
```toml
# In .cargo/config.toml
[target.hexagon-unknown-qurt]
rustflags = [
    "-C", "link-args=-shared",
    "-C", "link-args=-fPIC",
    "-C", "relocation-model=pic"
]
```

## Testing

Since `hexagon-unknown-qurt` requires the QuRT runtime environment, testing requires
either:
- Hexagon hardware with QuRT
- `hexagon-sim`
- QEMU (`qemu-system-hexagon`)

## Cross-compilation toolchains and C code

This target requires the proprietary [Hexagon SDK toolchain for C interoperability](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK):

- **Sample SDK Path**: `/opt/Hexagon_SDK/6.3.0.0/`
- **Toolchain**: Use `hexagon-clang` from the Hexagon SDK
- **Libraries**: Link against QuRT system libraries as needed

### C Interoperability Example

```rust
// lib.rs
#![no_std]
extern crate std;

#[unsafe(no_mangle)]
pub extern "C" fn rust_function() -> i32 {
    // Your Rust code here
    42
}

fn main() {
    // Example usage
    let result = rust_function();
    assert_eq!(result, 42);
}
```

```c
// wrapper.c
extern int rust_function(void);

int main() {
    return rust_function();
}
```

The target supports both static linking for standalone applications and dynamic
linking for modular architectures, making it flexible for various QuRT
deployment scenarios.
