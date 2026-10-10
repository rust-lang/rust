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
QuRT's POSIX-like threading and Dinkumware C library.

By default, code generated with this target should run on Hexagon DSP hardware
running the QuRT real-time operating system, or on `qemu-system-hexagon`.

- `-Ctarget-cpu=hexagonv69` targets Hexagon V69 architecture (default)
- `-Ctarget-cpu=hexagonv73` adds support for instructions defined up to Hexagon V73

Functions marked `extern "C"` use the [Hexagon architecture calling convention](https://lists.llvm.org/pipermail/llvm-dev/attachments/20190916/21516a52/attachment-0001.pdf).

The [Hexagon SDK](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK) is
required for building and running programs for this target. It provides:

- `hexagon-clang` (compiler and linker)
- QuRT runtime libraries (`libqurt.a`, `libposix.a`, etc.)
- CRT startup objects (`crt1.o`, `crt0.o`, `init.o`, `fini.o`, `debugmon.o`)
- `qemu-system-hexagon` emulator

Programs require the `restricted_std` feature gate:

```rust
#![feature(restricted_std)]
```

## SDK setup

Source the SDK environment script to set `HEXAGON_SDK_ROOT` and tool paths:

```sh
source /opt/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
```

This exports `HEXAGON_SDK_ROOT`, `DEFAULT_HEXAGON_TOOLS_ROOT`, and
`DEFAULT_QURT_PATH`. All paths below reference these variables.

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

## Linking

QuRT executables require specific CRT startup objects and system libraries.
A `build.rs` script can derive the library paths from environment variables set
by the SDK's `setup_sdk_env.source`.

### Example `build.rs`

```rust
use std::env;
use std::path::PathBuf;

fn main() {
    // Only apply QuRT link configuration when targeting hexagon-unknown-qurt
    let target = env::var("TARGET").unwrap_or_default();
    if target != "hexagon-unknown-qurt" {
        return;
    }

    // Derive paths from Hexagon SDK environment variables.
    // These are set by: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
    let sdk = env::var("HEXAGON_SDK_ROOT")
        .expect("HEXAGON_SDK_ROOT not set — source setup_sdk_env.source");
    let tools_root = env::var("DEFAULT_HEXAGON_TOOLS_ROOT")
        .unwrap_or_else(|_| format!("{sdk}/tools/HEXAGON_Tools/19.0.04"));
    let qurt_path = env::var("DEFAULT_QURT_PATH")
        .unwrap_or_else(|_| format!("{sdk}/rtos/qurt"));

    let arch = "v69";   // match -Ctarget-cpu
    let hexlib = PathBuf::from(&tools_root)
        .join("Tools/target/hexagon/lib").join(arch).join("G0");
    let qurtlib = PathBuf::from(&qurt_path)
        .join(format!("compute{arch}")).join("lib");

    // CRT startup objects
    println!("cargo:rustc-link-arg={}", qurtlib.join("crt1.o").display());
    println!("cargo:rustc-link-arg={}", hexlib.join("crt0.o").display());
    println!("cargo:rustc-link-arg={}", hexlib.join("init.o").display());
    println!("cargo:rustc-link-arg={}", qurtlib.join("debugmon.o").display());

    // QuRT's ELF loader requires the program's load address to fall within
    // the virtual memory pool (starting at page 0x40 = address 0x40000).
    println!("cargo:rustc-link-arg=-Wl,--section-start=.start=0x40000");

    // Stub symbols not available on QuRT
    for sym in ["sched_yield", "unsetenv", "_Unwind_Backtrace", "_Unwind_GetIPInfo"] {
        println!("cargo:rustc-link-arg=-Wl,--defsym={sym}=abort");
    }

    // Library search paths
    println!("cargo:rustc-link-search=native={}", qurtlib.display());
    println!("cargo:rustc-link-search=native={}", hexlib.display());

    // QuRT system libraries (use --start-group for circular deps)
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for lib in ["qurt", "posix", "qurtcfs", "timer_main", "timer_island"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    // Exception handling and C runtime
    println!("cargo:rustc-link-lib=static=c_eh");
    println!("cargo:rustc-link-lib=static=c");
    println!("cargo:rustc-link-lib=static=qcc");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // CRT finalization
    println!("cargo:rustc-link-arg={}", hexlib.join("fini.o").display());

    // Re-run if SDK path changes
    println!("cargo:rerun-if-env-changed=HEXAGON_SDK_ROOT");
    println!("cargo:rerun-if-env-changed=DEFAULT_HEXAGON_TOOLS_ROOT");
    println!("cargo:rerun-if-env-changed=DEFAULT_QURT_PATH");
}
```

### Compiling and linking

With the `build.rs` above, build with:

```sh
source ${HEXAGON_SDK_ROOT}/setup_sdk_env.source

cargo build --target hexagon-unknown-qurt \
    -Zbuild-std=core,alloc,std,panic_abort \
    -Zbuild-std-features=restricted-std
```

Or with `rustc` directly, passing all link flags explicitly (set `HEXLIB` and
`QURTLIB` from the SDK environment as shown in the `build.rs` above):

```sh
HEXLIB="${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/v69/G0"
QURTLIB="${DEFAULT_QURT_PATH}/computev69/lib"

rustc program.rs \
    --target hexagon-unknown-qurt \
    --edition 2021 \
    -C linker=hexagon-clang \
    -C panic=abort \
    -C "link-args=-nostdlib" \
    -C "link-args=${QURTLIB}/crt1.o ${HEXLIB}/crt0.o ${HEXLIB}/init.o ${QURTLIB}/debugmon.o" \
    -C "link-args=-Wl,--section-start=.start=0x40000" \
    -C "link-args=-Wl,--defsym=sched_yield=abort" \
    -C "link-args=-Wl,--defsym=unsetenv=abort" \
    -C "link-args=-Wl,--defsym=_Unwind_Backtrace=abort" \
    -C "link-args=-Wl,--defsym=_Unwind_GetIPInfo=abort" \
    -C "link-args=-L${QURTLIB} -L${HEXLIB}" \
    -C "link-args=-Wl,--start-group" \
    -C "link-args=-lqurt -lposix -lqurtcfs -ltimer_main -ltimer_island" \
    -C "link-args=${HEXLIB}/libc_eh.a -lc -lqcc" \
    -C "link-args=-Wl,--end-group" \
    -C "link-args=${HEXLIB}/fini.o" \
    -o program
```

The above use hexagon-clang/ld.qcld, but an alternative linker is available:
- [eld](https://github.com/qualcomm/eld), which is provided with both
  [the opensource hexagon toolchain](https://github.com/quic/toolchain_for_hexagon)
  and the Hexagon SDK

## Testing

Programs can be tested using `qemu-system-hexagon` from the Hexagon SDK.

### Running a static executable on QEMU

For programs linked as static executables (as shown in the linking examples
above), pass the program directly to `runelf.pbn`:

```sh
${HEXAGON_SDK_ROOT}/tools/Tools/QEMUHexagon/bin/qemu-system-hexagon \
    -machine V69NA_1024 \
    -kernel ${DEFAULT_QURT_PATH}/computev69/sdksim_bin/runelf.pbn \
    -append "/path/to/program"
```

The QuRT boot loader (`runelf.pbn`) is passed as `-kernel` and it loads the
user program specified via `-append`. No configuration files or cosim plugins
are needed — the machine model includes timer and interrupt controller
emulation.

### Running a shared object on QEMU

The Hexagon SDK provides `run_main_on_hexagon_sim`, a QuRT program that
dynamically loads a user shared object and calls its `main()`. This is the
standard approach used by the SDK's build system for running tests.

First, build the Rust program as a shared object:

```sh
rustc program.rs \
    --target hexagon-unknown-qurt \
    --edition 2021 \
    --crate-type cdylib \
    -C linker=hexagon-clang \
    -C panic=abort \
    -o libprogram.so
```

Then run it using `run_main_on_hexagon_sim`:

```sh
RUN_MAIN="${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/hexagon_toolv19_v69/run_main_on_hexagon_sim"

${HEXAGON_SDK_ROOT}/tools/Tools/QEMUHexagon/bin/qemu-system-hexagon \
    -machine V69NA_1024 \
    -kernel ${DEFAULT_QURT_PATH}/computev69/sdksim_bin/runelf.pbn \
    -append "${RUN_MAIN} -- libprogram.so"
```

Arguments after the `.so` filename are passed as `argc`/`argv` to `main()`:

```sh
    -append "${RUN_MAIN} -- libprogram.so arg1 arg2"
```

The `run_main_on_hexagon_sim` approach is useful for:
- Programs that need to be loaded dynamically (plugin architectures)
- Matching the SDK's standard test workflow
- Testing shared libraries built with `--crate-type cdylib`

## Qualcomm Hexagon Libraries (QHL)

The Hexagon SDK includes optimized math, BLAS, and DSP libraries that can be
called from Rust via `extern "C"` declarations:

- **qhmath** — scalar and array math: `qhmath_sin_f`, `qhmath_cos_f`,
  `qhmath_sqrt_f`, `qhmath_exp_f`, `qhmath_sigmoid_f`, etc.
- **qhblas** — BLAS operations: `qhblas_vector_add_af`,
  `qhblas_f_vector_dot_af`, `qhblas_vector_scaling_af`, etc.
- **qhblas_hvx** — HVX-accelerated BLAS: `qhblas_hvx_vector_add_af`,
  `qhblas_hvx_f_vector_dot_af`, `qhblas_hvx_vector_hadamard_af`, etc.
- **qhmath_hvx** — HVX-accelerated math: `qhmath_hvx_sin_af`,
  `qhmath_hvx_cos_af`, `qhmath_hvx_sqrt_af`
- **qhdsp** — signal processing: `qhdsp_crc32_poly`, FFT, FIR/IIR filters

To link QHL libraries, add these paths and libraries (in `build.rs` or as
`-C link-args`):

```rust,ignore (snippet-missing-imports-and-context)
// In build.rs, inside the hexagon-unknown-qurt block:
let qhl = PathBuf::from(&sdk).join("libs/qhl/prebuilt/hexagon_toolv19_v69");
let qhl_hvx = PathBuf::from(&sdk).join("libs/qhl_hvx/prebuilt/hexagon_toolv19_v69");
println!("cargo:rustc-link-search=native={}", qhl.display());
println!("cargo:rustc-link-search=native={}", qhl_hvx.display());
for lib in ["qhmath", "qhblas", "qhdsp", "qhcomplex",
            "qhmath_hvx", "qhblas_hvx", "qhdsp_hvx"] {
    println!("cargo:rustc-link-lib=static={lib}");
}
```

Example Rust usage:

```rust,ignore (requires-qhl-libraries-to-link)
extern "C" {
    fn qhmath_sqrt_f(x: f32) -> f32;
    fn qhblas_hvx_vector_add_af(
        i1: *const f32, i2: *const f32, out: *mut f32, size: u32,
    ) -> i32;
}

unsafe {
    let sqrt4 = qhmath_sqrt_f(4.0);   // 2.0
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut c = [0.0f32; 4];
    qhblas_hvx_vector_add_af(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 4);
    // c ≈ [11.0, 22.0, 33.0, 44.0]  (HVX float has ~1e-5 precision)
}
```

## Working `std` functionality

The following `std` features are expected to work:

- **Heap allocation**: `Vec`, `String`, `Box`, `HashMap`, `BTreeMap`, `VecDeque`,
  `Rc`, `Arc`
- **Formatting/IO**: `println!`, `eprintln!`, `format!`, `write!`,
  `stdout().write_all()`, `stderr().write_all()`
- **Synchronization**: `Mutex`, `RwLock`, `Condvar`, `Once`, `OnceLock`,
  `AtomicI32`, `AtomicU32`, `AtomicBool` (max atomic width is 32 bits)
- **Threading**: `thread::spawn`, `thread::Builder` (set stack size),
  `thread::sleep`, `thread_local!`, `thread::current().id()`
- **Time**: `Instant::now()`, `SystemTime::now()`, `Duration` arithmetic
- **File I/O**: `File::create`, `File::open`, `fs::remove_file`
- **Environment**: `env::current_dir()`, `env::temp_dir()`, `env::var()`
  (read-only)
- **Error handling**: `Result`, `Option` combinators
- **HVX SIMD**: `core::arch::hexagon` intrinsics (128-byte vectors via
  `#![feature(stdarch_hexagon)]`)

## Known limitations

- **`panic=unwind` not functional at runtime**: The target compiles with
  `panic=unwind` but panics abort instead of unwinding. Use `-C panic=abort`.
- **No process spawning**: `Command` / `process::exit` are not available
- **No networking**: Socket APIs are not supported
- **32-bit atomics maximum**: Use `AtomicU32`/`AtomicI32`, not
  `AtomicU64`/`AtomicUsize` on this 32-bit target
- **Thread stack size**: QuRT's default heap is limited (~512 KB); use
  `thread::Builder::new().stack_size(8192)` or similar small values to
  avoid out-of-memory failures
- **Environment variables**: `env::set_var` and `env::remove_var` are not
  functional; `env::var` works for reading pre-set variables
- **File I/O quirks**: QuRT's CFS (cosim filesystem) has known issues:
  `write()` may report one extra byte written, `read()` may return 0 bytes
  in the emulator, and `stat()` is not supported
- **`thread::yield_now()`**: Calls `sched_yield` which is stubbed to `abort`;
  use `thread::sleep(Duration::from_millis(0))` as an alternative
- **`_Unwind_Backtrace`**: Stubbed to `abort`; backtraces are not available

## Cross-compilation toolchains and C code

This target requires the [Hexagon SDK](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK)
for C interoperability:

- **Compiler**: `hexagon-clang` / `hexagon-clang++`
- **QuRT libraries**: `${DEFAULT_QURT_PATH}/computev69/lib/`
- **Hex tools libraries**: `${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/v69/G0/`
- **QHL libraries**: `${HEXAGON_SDK_ROOT}/libs/qhl/prebuilt/hexagon_toolv19_v69/`

### Simple C Interoperability Example

```rust
#![feature(restricted_std)]

#[unsafe(no_mangle)]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = rust_add(2, 3);
    println!("result = {result}");
}
```

```c
// call_from_c.c
extern int rust_add(int a, int b);

int use_rust(void) {
    return rust_add(2, 3);
}
```
