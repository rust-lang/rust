# QNX

**Tier: 3**

Support for the [QNX®](https://qnx.com) [QNX Software Development Platform (SDP)], version 7.0, 7.1 and 8.0.

[QNX Software Development Platform (SDP)]: https://qnx.software/en/software/products-and-solutions/qnx-software-development-platform

The [QNX Software Development Platform (SDP)] is a development environment that
you download and install on a host computer. It includes a C toolchain for your
host, an IDE, and various board support packages for different target platforms.
You can then use QNX SDP to build a custom run-time environment which you deploy
onto an embedded device. That run-time environment will include a microkernel,
whatever services you have selected, and perhaps one or more applications
written in Rust.

In QNX SDP 7.x the run-time environment is based on QNX Neutrino RTOS 7.x, while
in QNX SDP 8.0 the run-time environment is based on QNX OS 8.0. The name change
reflects architectural changes in the RTOS, but both use a microkernel design.

## Target maintainers

[@flba-eb](https://github.com/flba-eb)
[@gh-tr](https://github.com/gh-tr)
[@jonathanpallant](https://github.com/jonathanpallant)
[@japaric](https://github.com/japaric)

## Requirements

The following QNX SDP versions and compilation targets are supported:

| Target Tuple                        | QNX Version                   | Target Architecture | Full support | `no_std` support |
| ----------------------------------- | ----------------------------- | ------------------- | :----------: | :--------------: |
| `aarch64-unknown-qnx`               | QNX SDP 8.0+                  | AArch64             |      ?       |        ✓         |
| `x86_64-pc-qnx`                     | QNX SDP 8.0+                  | x86_64              |      ?       |        ✓         |
| `aarch64-unknown-nto-qnx710_iosock` | QNX SDP 7.1 with io-sock      | AArch64             |      ?       |        ✓         |
| `x86_64-pc-nto-qnx710_iosock`       | QNX SDP 7.1 with io-sock      | x86_64              |      ?       |        ✓         |
| `aarch64-unknown-nto-qnx710`        | QNX SDP 7.1 with io-pkt       | AArch64             |      ✓       |        ✓         |
| `x86_64-pc-nto-qnx710`              | QNX SDP 7.1 with io-pkt       | x86_64              |      ✓       |        ✓         |
| `aarch64-unknown-nto-qnx700`        | QNX SDP 7.0                   | AArch64             |      ?       |        ✓         |
| `i686-pc-nto-qnx700`                | QNX SDP 7.0                   | x86                 |      -       |        ✓         |

* QNX SDP 7.0 only offers the `io-pkt` network stack
* QNX SDP 7.1 uses the `io-pkt` network stack by default, but also includes the optional `io-sock` network stack
* QNX SDP 8.0 only offers the `io-sock` network stack

In the table above, 'full support' indicates support for building Rust
applications with the full standard library. A '?' means that support is
in-progress. `no_std` support is for building `#![no_std]` applications where
only `core` and `alloc` are available.

For building or using the Rust toolchain for QNX, the relevant version of the
[QNX Software Development Platform (SDP)] must be installed and initialized.
Initialization is usually done by sourcing `qnxsdp-env.sh` (this will be
installed as part of the SDP, so see the installation instruction provided with
the SDP). Afterwards [`qcc`] (the QNX C/C++ compiler) should be available in
your system PATH because it will be called during Rust compilation (e.g. for
linking executables).

[`qcc`]: https://www.qnx.com/developers/docs/latest/com.qnx.doc.neutrino.utilities/topic/q/qcc.html

When linking `no_std` applications, they must link against `libc.so` (see
example). This is required because applications always link against the `crt`
library and `crt` depends on `libc.so`. This is done automatically when using
the standard library.

## Conditional compilation

For conditional compilation, the following QNX specific attributes are defined:

- `target_os` = `"nto"`
    - `target_env` = `"nto70"` (for QNX SDP 7.0)
    - `target_env` = `"nto71"` (for QNX SDP 7.1 with "classic" network stack "io_pkt")
    - `target_env` = `"nto71_iosock"` (for QNX SDP 7.1 with "new" network stack "io_sock")
- `target_os` = `"qnx"` (for QNX SDP 8.0 or higher)

## Building the target

1. Create a `bootstrap.toml`

    Example content:

    ```toml
    profile = "compiler"
    change-id = 999999

    [build]
    host = ["x86_64-unknown-linux-gnu"]
    target = ["x86_64-unknown-linux-gnu", "aarch64-unknown-nto-qnx710"]
    ```

2. Compile the Rust toolchain with the QNX SDP environment loaded

    As noted above, we need the right environment variables for QNX SDP to work,
    and for `qcc` to be in your system PATH. Typically this is done by sourcing
    the `qnxsdp-env.sh` file (or equivalent for your host platform).

    To build on Linux, you would run:

    ```bash
    source ~/qnx710/qnxsdp-env.sh
    ./x.py build rustc library/core library/alloc library/std
    ```

## Building Rust programs

Rust does not ship pre-compiled artifacts for this target. To compile for this
target, you must either build Rust with the target enabled (see "Building the
target" above), or build your own copy of `core` by using `build-std` or
similar.

Compiled executables can run directly on QNX, either by including them in the
disk image, or copying them over the network to a running system.

Compiling C code requires the same environment variables to be set as compiling
the Rust toolchain (see above), to ensure `qcc` is used with proper arguments.
To ensure compatibility, do not specify any further arguments that for example
change calling conventions or memory layout.

## Running the Rust test suite

The test suites of the Rust compiler and standard library can be executed much
like other Rust targets. The environment for testing should match the one used
during compiler compilation (refer to notes on `qnxsdp-env.sh` above) with
the addition of the `TEST_DEVICE_ADDR` environment variable. The
`TEST_DEVICE_ADDR` variable controls the remote runner and should point to a
target running the `remote-test-server` executable.

Note that some tests are failing which is why they are currently excluded by the
target maintainers which can be seen in the following example.

To run all tests on a x86_64 QNX Neutrino 7.1 target:

```bash
source ~/qnx710/qnxsdp-env.sh
export TEST_DEVICE_ADDR="1.2.3.4:12345" # must address the test target, can be a SSH tunnel

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

./x.py test \
    $exclude_tests \
    --stage 1 \
    --target x86_64-pc-nto-qnx710
```

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

## Disabling RELocation Read-Only (RELRO)

While not recommended by default, some QNX kernel setups may require the `RELRO`
to be disabled with `-C relro_level=off`, e.g. by adding it to the
`.cargo/config.toml` file:

```toml
[target.aarch64-unknown-nto-qnx700]
rustflags = ["-C", "relro_level=off"]
```

If your QNX kernel does not allow it, and `relro` is not disabled, running the
compiled binary would fail with `syntax error: ... unexpected` or similar. This
is due to kernel trying to interpret the compiled binary with `/bin/sh`, and
obviously failing. To verify that this is really the case, run your binary with
the `DL_DEBUG=all` env var, and look for this output. If you see it, you should
disable `relro` as described above.

```text
Resolution scope for Executable->/bin/sh:
        Executable->/bin/sh
        libc.so.4->/usr/lib/ldqnx-64.so.2
```
