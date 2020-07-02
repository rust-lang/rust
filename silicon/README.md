# Setup

1. Download and install Xcode 12 beta, then set it as your default

    ```
    xcode-select -s /Applications/Xcode-beta.app/Contents/Developer/
    ```

1. Install CMake

    Install this however you like. A simple way is to download it and add it to your path;

    ```
    curl -L 'https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Darwin-x86_64.tar.gz' -o cmake.tgz
    tar -xf cmake.tgz
    export PATH="${PATH}:${PWD}/cmake-3.18.0-Darwin-x86_64/CMake.app/Contents/bin"
    ```

- Clone [this fork of rust][fork], checkout the "silicon" branch.

[fork]: https://github.com/shepmaster/rust

# Cross-compile the compiler for the DTK

The current Rust compiler doesn't know how to target the DTK. We use
some advanced build system configuration to build an enhanced version
that can.

This section can be simplified when an appropriate target
specification is added to the standard Rust bootstrapping compiler.

## Compiling on x86_64-apple-darwin (a.k.a. not the DTK)

1. `cd silicon/cross`

1. Configure the compiler

    ```
    ../../configure
    ```

1. Cross-compile the compiler

    ```
    SDKROOT=$(xcrun -sdk macosx11.0 --show-sdk-path) \
    MACOSX_DEPLOYMENT_TARGET=$(xcrun -sdk macosx11.0 --show-sdk-platform-version) \
    DESTDIR=/tmp/crossed \
    ../../x.py install -i --stage 1 --host aarch64-apple-darwin --target aarch64-apple-darwin,x86_64-apple-darwin \
    src/librustc library/std
    ```

1. Copy the cross-compiler to the DTK

```
rsync -avz /tmp/crossed/ dtk:crossed/
```

## Compiling on aarch64-apple-darwin (a.k.a. the DTK)

1. Configure the compiler

    ```
    ../../configure \
    --build=x86_64-apple-darwin
    ```

1. Cross-compile the compiler

    ```
    SDKROOT=$(xcrun -sdk macosx11.0 --show-sdk-path) \
    MACOSX_DEPLOYMENT_TARGET=$(xcrun -sdk macosx11.0 --show-sdk-platform-version) \
    DESTDIR=~/crossed \
    ../../x.py install -i --stage 1 --host aarch64-apple-darwin --target aarch64-apple-darwin,x86_64-apple-darwin \
    src/librustc library/std
    ```

# Next steps

At this point, you have a compiler that should be suffient for many goals.

## Build a native binary

On the DTK, you can now use native Rust:

```
% echo 'fn main() { println!("Hello, DTK!") }' > hello.rs

% ~/crossed/usr/local/bin/rustc hello.rs

% ./hello
Hello, DTK!

% file hello
hello: Mach-O 64-bit executable arm64
```

## Installing rustup

Rustup doesn't know that `x86_64` binaries can run on `aarch64`, so we
have to hack the install script so it thinks that the machine is
`x86_64`.

1. Download the installer script

    ```
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh
    ```

1. Apply this patch to force it to think we are x86_64:

    ```diff
    --- /tmp/rustup.sh  2020-07-05 07:49:10.000000000 -0700
    +++ /tmp/rustup-hacked.sh   2020-07-04 17:27:08.000000000 -0700
    @@ -188,6 +188,8 @@
             fi
         fi

    +    _cputype=x86_64
    +
         case "$_ostype" in

             Android)
    ```

1. Install rustup

    ```
    bash /tmp/rustup.sh --default-toolchain none
    ```

1. Create a toolchain and make it the default:

    ```
    rustup toolchain link native ~/crossed/usr/local/
    rustup default native
    ```

## Build the compiler on the DTK using the native toolchain

Re-run the setup steps on the DTK, if you haven't already.

1. `cd silicon/pure-native`

1. Get an x86_64 version of Cargo to use during bootstrapping

    ```
    curl 'https://static.rust-lang.org/dist/2020-06-16/cargo-beta-x86_64-apple-darwin.tar.xz' -o cargo.tar.xz
    tar -xf cargo.tar.xz
    ```

1. Configure the compiler

    ```
    ../../configure \
    --set build.cargo="${PWD}/cargo-beta-x86_64-apple-darwin/cargo/bin/cargo" \
    --set build.rustc="${HOME}/crossed/usr/local/bin/rustc" \
    --set build.rustfmt="${HOME}/crossed/usr/local/bin/rustfmt"
    ```

1. Build and install the compiler

    ```
    DESTDIR=~/native-build \
    ../../x.py install -i --stage 1 --host aarch64-apple-darwin --target aarch64-apple-darwin,x86_64-apple-darwin \
    src/librustc library/std cargo
    ```

# Miscellaneous notes

- `SDKROOT` - used to indicate to `rustc` what SDK whould be
  used. Point this at the `MacOSX11` target inside your `Xcode.app` /
  `Xcode-beta.app`

- `MACOSX_DEPLOYMENT_TARGET` - used to indicate to `rustc` what
  version of macOS to target.

- We build LLVM *3 times* and use incremental builds for faster
  iteration times. This can easily take up **30 GiB** of
  space. Building can take several hours on an 2.9 GHz 6-Core Intel
  Core i9.

- Rough build times
  - 2.9 GHz 6-Core Intel Core i9
    - bootstrap - `Build completed successfully in 0:30:32`
    - cross - `Build completed successfully in 1:04:22`
  - DTK
    - pure-native - `Build completed successfully in 0:39:10`

# Troubleshooting

- Failure while running `x.py`? Add `-vvvvvvv`.

- "archive member 'something.o' with length xxxx is not mach-o or llvm bitcode file 'some-tmp-path/libbacktrace_sys-e02ed1fd10e8b80d.rlib' for architecture arm64"?
  - Find the rlib: `find build -name 'libbacktrace_sys*'`
  - Remove it: `rm build/aarch64-apple-darwin/stage0-std/aarch64-apple-darwin/release/deps/libbacktrace_sys-e02ed1fd10e8b80d.r{meta,lib}`
  - Look at the objects it created in `build/aarch64-apple-darwin/stage0-std/aarch64-apple-darwin/release/build/backtrace-sys-fec595dd32504c90/`
  - Delete those too: `rm -r build/aarch64-apple-darwin/stage0-std/aarch64-apple-darwin/release/build/backtrace-sys-fec595dd32504c90`
  - Rebuild with verbose
