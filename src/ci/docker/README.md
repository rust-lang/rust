# Docker images for CI

This folder contains a bunch of docker images used by the continuous integration
(CI) of Rust. A script is accompanied (`run.sh`) with these images to actually
execute them.

Note that a single Docker image can be used by multiple CI jobs, so the job name
is the important thing that you should know. You can examine the existing CI jobs in
the [`jobs.yml`](../github-actions/jobs.yml) file.

To run a specific CI job locally, you can use the `citool` Rust crate:

```
cargo run --manifest-path src/ci/citool/Cargo.toml run-local <job-name>
```

For example, to run the `x86_64-gnu-llvm-19-1` job:
```
cargo run --manifest-path src/ci/citool/Cargo.toml run-local x86_64-gnu-llvm-19-1
```

The job will output artifacts in an `obj/<image-name>` dir at the root of a repository. Note
that the script will overwrite the contents of this directory. `<image-name>` is set based on the
Docker image executed in the given CI job.

**NOTE**: In CI, the script outputs the artifacts to the `obj` directory,
while locally, to the `obj/<image-name>` directory. This is primarily to prevent
strange linker errors when using multiple Docker images.

For some Linux workflows (for example `x86_64-gnu-llvm-19-N`), the process is more involved. You will need to see which script is executed for the given workflow inside the [`jobs.yml`](../github-actions/jobs.yml) file and pass it through the `DOCKER_SCRIPT` environment variable. For example, to reproduce the `x86_64-gnu-llvm-19-3` workflow, you can run the following script:

```
DOCKER_SCRIPT=x86_64-gnu-llvm3.sh ./src/ci/docker/run.sh x86_64-gnu-llvm-19
```

## Local Development

Refer to the [dev guide](https://rustc-dev-guide.rust-lang.org/tests/docker.html) for more information on testing locally.

## Filesystem layout

- Each host architecture has its own `host-{arch}` directory, and those
  directories contain a subdirectory for each Docker image (plus the `disabled`
  subdirectory).
- `host-{arch}/disabled` contains images that are not built on CI.
- `scripts` contains files shared by multiple Docker images.

## Docker Toolbox on Windows

For Windows before Windows 10, the docker images can be run on Windows via
[Docker Toolbox]. There are several preparation needs to be made before running
a Docker image.

1. Stop the virtual machine from the terminal with `docker-machine stop`

2. If your Rust source is placed outside of `C:\Users\**`, e.g. if you place the
    repository in the `E:\rust` folder, please add a shared folder from
    VirtualBox by:

    1. Select the "default" virtual machine inside VirtualBox, then click
        "Settings"
    2. Go to "Shared Folders", click "Add shared folder" (the folder icon with
        a plus sign), fill in the following information, then click "OK":

        * Folder path: `E:\rust`
        * Folder name: `e/rust`
        * Read-only: ☐ *unchecked*
        * Auto-mount: ☑ *checked*
        * Make Permanent: ☑ *checked*

3. VirtualBox might not support creating symbolic links inside a shared folder
    by default. You can enable it manually by running these from `cmd.exe`:

    ```bat
    cd "C:\Program Files\Oracle\VirtualBox"
    VBoxManage setextradata default VBoxInternal2/SharedFoldersEnableSymlinksCreate/e/rust 1
    ::                                                                              ^~~~~~
    ::                                                                              folder name
    ```

4. Restart the virtual machine from terminal with `docker-machine start`.

To run the image,

1. Launch the "Docker Quickstart Terminal".
2. Execute `./src/ci/docker/run.sh $image_name` as explained at the beginning.

[Docker Toolbox]: https://www.docker.com/products/docker-toolbox

## Cross toolchains

A number of these images take quite a long time to compile as they're building
whole gcc toolchains to do cross builds with. Much of this is relatively
self-explanatory but some images use [crosstool-ng] which isn't quite as self
explanatory. Below is a description of where these `*.defconfig` files come form,
how to generate them, and how the existing ones were generated.

[crosstool-ng]: https://github.com/crosstool-ng/crosstool-ng

### Generating a `.defconfig` file

**NOTE:** Existing Dockerfiles can also be a good guide for the process and order
of script execution.

If you have a `linux-cross` image lying around you can use that and skip the
next two steps.

- First we spin up a container and copy all scripts into it. All
  these steps are outside the container:

```
# Note: We use ubuntu:22.04 because that's the "base" of linux-cross Docker
# image, or simply run ./src/ci/docker/run.sh once, which will download the correct
# one and you can check it out with `docker images`
$ docker run -it ubuntu:22.04 bash
# in another terminal:
$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
cfbec05ed730        ubuntu:22.04        "bash"              16 seconds ago      Up 15 seconds                           drunk_murdock
$ docker cp src/ci/docker/scripts drunk_murdock:/tmp/
```

- Then inside the container we build crosstool-ng by simply calling the bash
  script we copied in the previous step:

```
$ cd /tmp/scripts
# Download packages necessary for building
$ bash ./cross-apt-packages.sh
# Download and build crosstool-ng
$ bash ./crosstool-ng.sh
```

- In case you want to adjust or start from an existing config, copy that
  to the container. `crosstool-ng` will automatically load `./.config` if
  present. Otherwise one can use the TUI to load any config-file.

```
$ docker cp arm-linux-gnueabi.defconfig drunk_murdock:/tmp/.config
```

- Now, inside the container run the following command to configure the
  toolchain. To get a clue of which options need to be changed check the next
  section and come back.

```
$ cd /tmp/
$ ct-ng olddefconfig
$ ct-ng menuconfig
$ ct-ng savedefconfig
```

- Finally, we retrieve the `defconfig` file from the container and give it a
  meaningful name. This is done outside the container.

```
$ docker cp drunk_murdock:/tmp/defconfig arm-linux-gnueabi.defconfig
```

- Now you can shutdown the container or repeat the two last steps to generate a
  new `.defconfig` file.

### Toolchain configuration

Changes on top of the default toolchain configuration used to generate the
`.defconfig` files in this directory. The changes are formatted as follows:

```
$category > $option = $value -- $comment
```

### `arm-linux-gnueabi.defconfig`

For targets: `arm-unknown-linux-gnueabi`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = arm
- Target options > Architecture level = armv6 -- (+)
- Target options > Floating point = software (no FPU) -- (\*)
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.17.0
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `arm-linux-gnueabihf.defconfig`

For targets: `arm-unknown-linux-gnueabihf`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = arm
- Target options > Architecture level = armv6 -- (+)
- Target options > Use specific FPU = vfp -- (+)
- Target options > Floating point = hardware (FPU) -- (\*)
- Target options > Default instruction set mode = arm -- (+)
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.17.0
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `armv7-linux-gnueabihf.defconfig`

For targets: `armv7-unknown-linux-gnueabihf`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = arm
- Target options > Suffix to the arch-part = v7
- Target options > Architecture level = armv7-a -- (+)
- Target options > Use specific FPU = vfpv3-d16 -- (\*)
- Target options > Floating point = hardware (FPU) -- (\*)
- Target options > Default instruction set mode = thumb -- (\*)
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.17.0
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

(\*) These options have been selected to match the configuration of the arm
      toolchains shipped with Ubuntu 15.10
(+) These options have been selected to match the gcc flags we use to compile C
    libraries like jemalloc. See the mk/cfg/arm(v7)-unknown-linux-gnueabi{,hf}.mk
    file in Rust's source code.

### `aarch64-linux-gnu.defconfig`

For targets: `aarch64-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = arm
- Target options > Bitness = 64-bit
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 4.1.49
- Binary utilities > Version of binutils = 2.29.1
- C-library > glibc version = 2.17 -- aarch64 support was introduced in this version
- C compiler > gcc version = 13.2.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `i586-linux-gnu.defconfig`

For targets: `i586-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Target options > Target Architecture = x86
- Target options > Architecture level = i586
- Target options > Target CFLAGS = -Wa,-mrelax-relocations=no
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.32
- Binary utilities > binutils extra config = --enable-compressed-debug-sections=none -- (\*)
- C-library > glibc version = 2.17.0
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE

(\*) Compressed debug is enabled by default for gas (assembly) on Linux/x86 targets,
     but that makes our `compiler_builtins` incompatible with binutils < 2.32.

### `loongarch64-linux-gnu.defconfig`

For targets: `loongarch64-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = loongarch
- Target options > Bitness = 64-bit
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 5.19.16
- Binary utilities > Version of binutils = 2.42
- C-library > glibc version = 2.36
- C compiler > gcc version = 14.2.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `loongarch64-linux-musl.defconfig`

For targets: `loongarch64-unknown-linux-musl`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = loongarch
- Target options > Bitness = 64-bit
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 5.19.16
- Binary utilities > Version of binutils = 2.42
- C-library > musl version = 1.2.5
- C compiler > gcc version = 14.2.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `mips-linux-gnu.defconfig`

For targets: `mips-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = mips
- Target options > ABI = o32
- Target options > Endianness = Big endian
- Target options > Bitness = 32-bit
- Target options > Architecture level = mips32r2
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 4.4.302
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.23
- C compiler > gcc version = 8.5.0
- C compiler > gcc extra config = --with-fp-32=xx --with-odd-spreg-32=no
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `mipsel-linux-gnu.defconfig`

For targets: `mipsel-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = mips
- Target options > ABI = o32
- Target options > Endianness = Little endian
- Target options > Bitness = 32-bit
- Target options > Architecture level = mips32r2
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 4.4.302
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.23
- C compiler > gcc version = 8.5.0
- C compiler > gcc extra config = --with-fp-32=xx --with-odd-spreg-32=no
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `mips64-linux-gnu.defconfig`

For targets: `mips64-unknown-linux-gnuabi64`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = mips
- Target options > ABI = n64
- Target options > Endianness = Big endian
- Target options > Bitness = 64-bit
- Target options > Architecture level = mips64r2
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 4.4.302
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.23
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `mips64el-linux-gnu.defconfig`

For targets: `mips64el-unknown-linux-gnuabi64`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = mips
- Target options > ABI = n64
- Target options > Endianness = Little endian
- Target options > Bitness = 64-bit
- Target options > Architecture level = mips64r2
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 4.4.302
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.23
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `powerpc-linux-gnu.defconfig`

For targets: `powerpc-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = powerpc
- Target options > Emit assembly for CPU = powerpc -- pure 32-bit PowerPC
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.30
- C-library > glibc version = 2.17 -- ~RHEL7 glibc
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `powerpc64-linux-gnu.defconfig`

For targets: `powerpc64-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = powerpc
- Target options > Bitness = 64-bit
- Target options > Emit assembly for CPU = power4 -- (+)
- Target options > Tune for CPU = power6 -- (+)
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.17 -- ~RHEL7 glibc
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

(+) These CPU options match the configuration of the toolchains in RHEL6.

### `riscv64-unknown-linux-gnu.defconfig`

For targets: `riscv64-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = riscv
- Target options > Bitness = 64-bit
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 4.20.17
- Binary utilities > Version of binutils = 2.36.1
- C-library > glibc version = 2.29
- C compiler > gcc version = 8.5.0
- C compiler > C++ = ENABLE -- to cross compile LLVM

### `s390x-linux-gnu.defconfig`

For targets: `s390x-unknown-linux-gnu`

- Path and misc options > Prefix directory = /x-tools/${CT\_TARGET}
- Path and misc options > Use a mirror = ENABLE
- Path and misc options > Base URL = https://ci-mirrors.rust-lang.org/rustc
- Target options > Target Architecture = s390
- Target options > Bitness = 64-bit
- Operating System > Target OS = linux
- Operating System > Linux kernel version = 3.2.101
- Binary utilities > Version of binutils = 2.32
- C-library > glibc version = 2.17 -- ~RHEL7 glibc
- C compiler > gcc version = 8.5.0
- C compiler > gcc extra config = --with-arch=z10 -- LLVM's minimum support
- C compiler > C++ = ENABLE -- to cross compile LLVM
