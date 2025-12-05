# nto-qnx

**Tier: 3**

The [QNX®][qnx.com] Neutrino (nto) Real-time operating system. Known as QNX OS
from version 8 onwards.

This support has been implemented jointly by [Elektrobit Automotive GmbH][Elektrobit]
and [QNX][qnx.com].

[qnx.com]: https://blackberry.qnx.com
[Elektrobit]: https://www.elektrobit.com

## Target maintainers

[@flba-eb](https://github.com/flba-eb)
[@gh-tr](https://github.com/gh-tr)
[@jonathanpallant](https://github.com/jonathanpallant)
[@japaric](https://github.com/japaric)

## Requirements

Currently, the following QNX versions and compilation targets are supported:

| Target Tuple                        | QNX Version                   | Target Architecture | Full support | `no_std` support |
| ----------------------------------- | ----------------------------- | ------------------- | :----------: | :--------------: |
| `aarch64-unknown-nto-qnx800`        | QNX OS 8.0                    | AArch64             |      ?       |        ✓         |
| `x86_64-pc-nto-qnx800`              | QNX OS 8.0                    | x86_64              |      ?       |        ✓         |
| `aarch64-unknown-nto-qnx710`        | QNX Neutrino 7.1 with io-pkt  | AArch64             |      ✓       |        ✓         |
| `x86_64-pc-nto-qnx710`              | QNX Neutrino 7.1 with io-pkt  | x86_64              |      ✓       |        ✓         |
| `aarch64-unknown-nto-qnx710_iosock` | QNX Neutrino 7.1 with io-sock | AArch64             |      ?       |        ✓         |
| `x86_64-pc-nto-qnx710_iosock`       | QNX Neutrino 7.1 with io-sock | x86_64              |      ?       |        ✓         |
| `aarch64-unknown-nto-qnx700`        | QNX Neutrino 7.0              | AArch64             |      ?       |        ✓         |
| `i686-pc-nto-qnx700`                | QNX Neutrino 7.0              | x86                 |              |        ✓         |

On QNX Neutrino 7.0 and 7.1, `io-pkt` is used as network stack by default.
QNX Neutrino 7.1 includes the optional network stack `io-sock`.
QNX OS 8.0 always uses `io-sock`. QNX OS 8.0 support is currently work in progress.

Adding other architectures that are supported by QNX is possible.

In the table above, 'full support' indicates support for building Rust applications with the full standard library. A '?' means that support is in-progress.
'`no_std` support' is for building `#![no_std]` applications where only `core` and `alloc` are available.

For building or using the Rust toolchain for QNX, the
[QNX Software Development Platform (SDP)](https://blackberry.qnx.com/en/products/foundation-software/qnx-software-development-platform)
must be installed and initialized.
Initialization is usually done by sourcing `qnxsdp-env.sh` (this will be installed as part of the SDP, see also installation instruction provided with the SDP).
Afterwards [`qcc`](https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.utilities/topic/q/qcc.html) (QNX C/C++ compiler)
should be available (in the `$PATH` variable).
`qcc` will be called e.g. for linking executables.

When linking `no_std` applications, they must link against `libc.so` (see example). This is
required because applications always link against the `crt` library and `crt` depends on `libc.so`.
This is done automatically when using the standard library.

### Disabling RELocation Read-Only (RELRO)

While not recommended by default, some QNX kernel setups may require the `RELRO` to be disabled with `-C relro_level=off`, e.g. by adding it to the `.cargo/config.toml` file:

```toml
[target.aarch64-unknown-nto-qnx700]
rustflags = ["-C", "relro_level=off"]
```

If your QNX kernel does not allow it, and `relro` is not disabled, running compiled binary would fail with `syntax error: ... unexpected` or similar.  This is due to kernel trying to interpret compiled binary with `/bin/sh`, and obviously failing.  To verify that this is really the case, run your binary with the `DL_DEBUG=all` env var, and look for this output. If you see it, you should disable `relro` as described above.

```text
Resolution scope for Executable->/bin/sh:
        Executable->/bin/sh
        libc.so.4->/usr/lib/ldqnx-64.so.2
```

### Small example application

Small `no_std` example is shown below. Applications using the standard library work as well.

```rust,ignore (platform-specific)
#![no_std]
#![no_main]
#![feature(lang_items)]

// We must always link against libc, even if no external functions are used
// "extern C" - Block can be empty but must be present
#[link(name = "c")]
extern "C" {
    pub fn printf(format: *const core::ffi::c_char, ...) -> core::ffi::c_int;
}

#[no_mangle]
pub extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    const HELLO: &'static str = "Hello World, the answer is %d\n\0";
    unsafe {
        printf(HELLO.as_ptr() as *const _, 42);
    }
    0
}

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_panic: &PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
```

The QNX support in Rust has been tested with QNX Neutrino 7.0 and 7.1. Support for QNX OS 8.0 is a work in progress.

There are no further known requirements.

## Conditional compilation

For conditional compilation, following QNX specific attributes are defined:

- `target_os` = `"nto"`
- `target_env` = `"nto71"` (for QNX Neutrino 7.1 with "classic" network stack "io_pkt")
- `target_env` = `"nto71_iosock"` (for QNX Neutrino 7.1 with network stack "io_sock")
- `target_env` = `"nto70"` (for QNX Neutrino 7.0)
- `target_env` = `"nto80"` (for QNX OS 8.0)

## Building the target

1. Create a `bootstrap.toml`

    Example content:

    ```toml
    profile = "compiler"
    change-id = 999999
    ```

2. Compile the Rust toolchain for an `x86_64-unknown-linux-gnu` host

    Compiling the Rust toolchain requires the same environment variables used for compiling C binaries.
    Refer to the [QNX developer manual](https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.prog/topic/devel_OS_version.html).

    To compile for QNX, environment variables must be set to use the correct tools and compiler switches:

    - `CC_<target>=qcc`
    - `CFLAGS_<target>=<nto_cflag>`
    - `CXX_<target>=qcc`
    - `AR_<target>=<nto_ar>`

    With:

    - `<target>` target triplet using underscores instead of hyphens, e.g. `aarch64_unknown_nto_qnx710`
    - `<nto_cflag>`

      - `-Vgcc_ntox86_cxx` for x86 (32 bit)
      - `-Vgcc_ntox86_64_cxx` for x86_64 (64 bit)
      - `-Vgcc_ntoaarch64le_cxx` for Aarch64 (64 bit)

    - `<nto_ar>`

      - `ntox86-ar` for x86 (32 bit)
      - `ntox86_64-ar` for x86_64 (64 bit)
      - `ntoaarch64-ar` for Aarch64 (64 bit)

    Example to build the Rust toolchain including a standard library for x86_64-linux-gnu and Aarch64-QNX-7.1:

    ```bash
    export build_env='
        CC_aarch64_unknown_nto_qnx710=qcc
        CFLAGS_aarch64_unknown_nto_qnx710=-Vgcc_ntoaarch64le_cxx
        CXX_aarch64_unknown_nto_qnx710=qcc
        AR_aarch64_unknown_nto_qnx710=ntoaarch64-ar
        '

    env $build_env \
        ./x.py build \
            --target x86_64-unknown-linux-gnu,aarch64-unknown-nto-qnx710 \
            rustc library/core library/alloc library/std
    ```

## Running the Rust test suite

The test suites of the Rust compiler and standard library can be executed much like other Rust targets.
The environment for testing should match the one used during compiler compilation (refer to `build_env` and `qcc`/`PATH` above) with the
addition of the TEST_DEVICE_ADDR environment variable.
The TEST_DEVICE_ADDR variable controls the remote runner and should point to the target, despite localhost being shown in the following example.
Note that some tests are failing which is why they are currently excluded by the target maintainers which can be seen in the following example.

To run all tests on a x86_64 QNX Neutrino 7.1 target:

```bash
export TEST_DEVICE_ADDR="localhost:12345" # must address the test target, can be a SSH tunnel
export build_env=<see above>

# Disable tests that only work on the host or don't make sense for this target.
# See also:
# - src/ci/docker/host-x86_64/i686-gnu/Dockerfile
# - https://rust-lang.zulipchat.com/#narrow/stream/182449-t-compiler.2Fhelp/topic/Running.20tests.20on.20remote.20target
# - .github/workflows/ci.yml
export exclude_tests='
    --exclude src/bootstrap
    --exclude src/tools/error_index_generator
    --exclude src/tools/linkchecker
    --exclude tests/ui-fulldeps
    --exclude rustc
    --exclude rustdoc'

env $build_env \
    ./x.py test \
        $exclude_tests \
        --stage 1 \
        --target x86_64-pc-nto-qnx710
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you must either build Rust with the target enabled (see "Building the target" above),
or build your own copy of `core` by using `build-std` or similar.

## Testing

Compiled executables can run directly on QNX.

### Rust std library test suite

The target needs sufficient resources to execute all tests. The commands below assume that a QEMU image
is used.

* Ensure that the temporary directory used by `remote-test-server` has enough free space and inodes.
  5GB of free space and 40000 inodes are known to be sufficient (the test will create more than 32k files).
  To create a QEMU image in an empty directory, run this command inside the directory:

  ```bash
  mkqnximage --type=qemu --ssh-ident=$HOME/.ssh/id_ed25519.pub --data-size=5000 --data-inodes=40000
  ```

  `/data` should have enough free resources.
  Set the `TMPDIR` environment variable accordingly when running `remote-test-server`, e.g.:
  ```bash
  TMPDIR=/data/tmp/rust remote-test-server --bind 0.0.0.0:12345
  ```

* Ensure the TCP stack can handle enough parallel connections (default is 200, should be 300 or higher).
  After creating an image (see above), edit the file `output/build/startup.sh`:
  1. Search for `io-pkt-v6-hc`
  2. Add the parameter `-ptcpip threads_max=300`, e.g.:
     ```text
     io-pkt-v6-hc -U 33:33 -d e1000 -ptcpip threads_max=300
     ```
  3. Update the image by running `mkqnximage` again with the same parameters as above for creating it.

* Running and stopping the virtual machine

  To start the virtual machine, run inside the directory of the VM:

  ```bash
  mkqnximage --run=-h
  ```

  To stop the virtual machine, run inside the directory of the VM:

  ```bash
  mkqnximage --stop
  ```

* Ensure local networking

  Ensure that 'localhost' is getting resolved to 127.0.0.1. If you can't ping the localhost, some tests may fail.
  Ensure it's appended to /etc/hosts (if first `ping` command fails).
  Commands have to be executed inside the virtual machine!

  ```bash
  $ ping localhost
  ping: Cannot resolve "localhost" (Host name lookup failure)

  $ echo "127.0.0.1 localhost" >> /etc/hosts

  $ ping localhost
  PING localhost (127.0.0.1): 56 data bytes
  64 bytes from 127.0.0.1: icmp_seq=0 ttl=255 time=1 ms
  ```

## Cross-compilation toolchains and C code

Compiling C code requires the same environment variables to be set as compiling the Rust toolchain (see above),
to ensure `qcc` is used with proper arguments.
To ensure compatibility, do not specify any further arguments that for example change calling conventions or memory layout.
