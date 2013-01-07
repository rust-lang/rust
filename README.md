# The Rust Programming Language

This is a compiler for Rust, including standard libraries, tools and
documentation.


## Installation

The Rust compiler currently must be built from a [tarball], unless you
are on Windows, in which case using the [installer][win-exe] is
recommended.

Since the Rust compiler is written in Rust, it must be built by
a precompiled "snapshot" version of itself (made in an earlier state
of development). As such, source builds require a connection to
the Internet, to fetch snapshots, and an OS that can execute the
available snapshot binaries.

Snapshot binaries are currently built and tested on several platforms:

* Windows (7, Server 2008 R2), x86 only
* Linux (various distributions), x86 and x86-64
* OSX 10.6 ("Snow Leopard") or greater, x86 and x86-64

You may find that other platforms work, but these are our "tier 1"
supported build environments that are most likely to work.

> ***Note:*** Windows users should read the detailed
> [getting started][wiki-start] notes on the wiki. Even when using
> the binary installer the Windows build requires a MinGW installation,
> the precise details of which are not discussed here.

To build from source you will also need the following prerequisite
packages:

* g++ 4.4 or clang++ 3.x
* python 2.6 or later (but not 3.x)
* perl 5.0 or later
* gnu make 3.81 or later
* curl

Assuming you're on a relatively modern *nix system and have met the
prerequisites, something along these lines should work.

    $ wget http://static.rust-lang.org/dist/rust-0.5.tar.gz
    $ tar -xzf rust-0.5.tar.gz
    $ cd rust-0.5
    $ ./configure
    $ make && make install

You may need to use `sudo make install` if you do not normally have
permission to modify the destination directory. The install locations
can be adjusted by passing a `--prefix` argument to
`configure`. Various other options are also supported, pass `--help`
for more information on them.

When complete, `make install` will place several programs into
`/usr/local/bin`: `rustc`, the Rust compiler; `rustdoc`, the
API-documentation tool, and `cargo`, the Rust package manager.

[wiki-start]: https://github.com/mozilla/rust/wiki/Note-getting-started-developing-Rust
[tarball]: http://static.rust-lang.org/dist/rust-0.5.tar.gz
[win-exe]: http://static.rust-lang.org/dist/rust-0.5-install.exe


## License

Rust is primarily distributed under the terms of both the MIT license
and the Apache License (Version 2.0), with portions covered by various
BSD-like licenses.

See LICENSE-APACHE, LICENSE-MIT, and COPYRIGHT for details.

## More help

The [tutorial] is a good starting point.

[tutorial]: http://static.rust-lang.org/doc/tutorial.html


========================== Steps for Android Target addtion =======================

1. setup android ndk standalone tool chain with platform=14 option

    Android NDK can be downloaded from http://developer.android.com/tools/sdk/ndk/index.html
    
    example command to setup standalone tool chain:
    
       ~/work/toolchains/android-ndk-r8c/build/tools$ ./make-standalone-toolchain.sh --platform=android-14 --install-dir=/home/ubuntu/work/toolchains/ndk_standalone --ndk-dir=/home/ubuntu/work/toolchains/android-ndk-r8c

2. Download rustc from git repository

    a. git clone  http://github.com/webconv/rust.git
    
    b. cd rust
    
    c. mkdir build
    
    d. cd build

3. Create Makefile using CMake

    cmake ../ -DTargetOsType=android -DTargetCpuType=arm -DToolchain=path_of_standalone_toolchain_dir

4. Build libuv and  llvm

    make libuv

    make llvm 


5. Create Makefile again (the Makefile made in step 3 does not contain information of llvm)

    cmake ../

6. Build Rustc ( make and make install have been separated)

    a.make
    
    b.make install  [ it will copy ARM libraries into /usr/local/lib/rustc/arm-unknown-android/lib
    

7. How to cross compiler
    
    rustc --target arm-unknown-android hello.rs
 

8. How to run on Android

    use adb -e push command to push all arm libs as specified in 6 b
   
    push your binary
   
    set LD_LIBRARY_PATH
   
    run using adb shell  

