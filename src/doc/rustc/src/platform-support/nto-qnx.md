# nto-qnx

**Tier: 3**

[QNX®][BlackBerry] Neutrino (nto) Real-time operating system.
The support has been implemented jointly by [Elektrobit Automotive GmbH][Elektrobit]
and [Blackberry QNX][BlackBerry].

[BlackBerry]: https://blackberry.qnx.com
[Elektrobit]: https://www.elektrobit.com

## Target maintainers

- Florian Bartels, `Florian.Bartels@elektrobit.com`, https://github.com/flba-eb
- Tristan Roach, `TRoach@blackberry.com`, https://github.com/gh-tr

## Requirements

Currently, only cross-compilation for QNX Neutrino on AArch64 and x86_64 are supported (little endian).
Adding other architectures that are supported by QNX Neutrino is possible.

The standard library, including `core` and `alloc` (with default allocator) are supported.

For building or using the Rust toolchain for QNX Neutrino, the
[QNX Software Development Platform (SDP)](https://blackberry.qnx.com/en/products/foundation-software/qnx-software-development-platform)
must be installed and initialized.
Initialization is usually done by sourcing `qnxsdp-env.sh` (this will be installed as part of the SDP, see also installation instruction provided with the SDP).
Afterwards [`qcc`](https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.utilities/topic/q/qcc.html) (QNX C/C++ compiler)
should be available (in the `$PATH` variable).
`qcc` will be called e.g. for linking executables.

When linking `no_std` applications, they must link against `libc.so` (see example). This is
required because applications always link against the `crt` library and `crt` depends on `libc.so`.
This is done automatically when using the standard library.

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
pub extern "C" fn main(_argc: isize, _argv: *const *const u8) -> isize {
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

The QNX Neutrino support of Rust has been tested with QNX Neutrino 7.1.

There are no further known requirements.

## Conditional compilation

For conditional compilation, following QNX Neutrino specific attributes are defined:

- `target_os` = `"nto"`
- `target_env` = `"nto71"` (for QNX Neutrino 7.1)

## Building the target

1. Create a `config.toml`

Example content:

```toml
profile = "compiler"
changelog-seen = 2
```

2. Compile the Rust toolchain for an `x86_64-unknown-linux-gnu` host (for both `aarch64` and `x86_64` targets)

Compiling the Rust toolchain requires the same environment variables used for compiling C binaries.
Refer to the [QNX developer manual](https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.prog/topic/devel_OS_version.html).

To compile for QNX Neutrino (aarch64 and x86_64) and Linux (x86_64):

```bash
export build_env='
    CC_aarch64-unknown-nto-qnx710=qcc
    CFLAGS_aarch64-unknown-nto-qnx710=-Vgcc_ntoaarch64le_cxx
    CXX_aarch64-unknown-nto-qnx710=qcc
    AR_aarch64_unknown_nto_qnx710=ntoaarch64-ar
    CC_x86_64-pc-nto-qnx710=qcc
    CFLAGS_x86_64-pc-nto-qnx710=-Vgcc_ntox86_64_cxx
    CXX_x86_64-pc-nto-qnx710=qcc
    AR_x86_64_pc_nto_qnx710=ntox86_64-ar'

env $build_env \
    ./x.py build \
        --target aarch64-unknown-nto-qnx710 \
        --target x86_64-pc-nto-qnx710 \
        --target x86_64-unknown-linux-gnu \
        rustc library/core library/alloc
```

## Running the Rust test suite

The test suites of the Rust compiler and standard library can be executed much like other Rust targets.
The environment for testing should match the one used during compiler compilation (refer to `build_env` and `qcc`/`PATH` above) with the
addition of the TEST_DEVICE_ADDR environment variable.
The TEST_DEVICE_ADDR variable controls the remote runner and should point to the target, despite localhost being shown in the following example.
Note that some tests are failing which is why they are currently excluded by the target maintainers which can be seen in the following example.

To run all tests on a x86_64 QNX Neutrino target:

```bash
export TEST_DEVICE_ADDR="localhost:12345" # must address the test target, can be a SSH tunnel
export build_env='
    CC_aarch64-unknown-nto-qnx710=qcc
    CFLAGS_aarch64-unknown-nto-qnx710=-Vgcc_ntoaarch64le_cxx
    CXX_aarch64-unknown-nto-qnx710=qcc
    AR_aarch64_unknown_nto_qnx710=ntoaarch64-ar
    CC_x86_64-pc-nto-qnx710=qcc
    CFLAGS_x86_64-pc-nto-qnx710=-Vgcc_ntox86_64_cxx
    CXX_x86_64-pc-nto-qnx710=qcc
    AR_x86_64_pc_nto_qnx710=ntox86_64-ar'

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
    --exclude rustdoc
    --exclude tests/run-make-fulldeps'

env $build_env \
    ./x.py test -j 1 \
        $exclude_tests \
        --stage 1 \
        --target x86_64-pc-nto-qnx710
```

Currently, only one thread can be used when testing due to limitations in `libc::fork` and `libc::posix_spawnp`.
See [fork documentation](https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.lib_ref/topic/f/fork.html)
(error section) for more information.
This can be achieved by using the `-j 1` parameter in the `x.py` call.
This issue is being researched and we will try to allow parallelism in the future.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you must either build Rust with the target enabled (see "Building the target" above),
or build your own copy of `core` by using `build-std` or similar.

## Testing

Compiled executables can run directly on QNX Neutrino.

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
