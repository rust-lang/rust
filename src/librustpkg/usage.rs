// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

pub fn general() {
    io::println("Usage: rustpkg [options] <cmd> [args..]

Where <cmd> is one of:
    build, clean, do, info, install, list, prefer, test, uninstall, unprefer

Options:

    -h, --help                  Display this message
    --sysroot PATH              Override the system root
    <cmd> -h, <cmd> --help      Display help for <cmd>");
}

pub fn build() {
    io::println("rustpkg build [options..] [package-ID]

Build the given package ID if specified. With no package ID argument,
build the package in the current directory. In that case, the current
directory must be a direct child of an `src` directory in a workspace.

Options:
    -c, --cfg      Pass a cfg flag to the package script
    --no-link      Compile and assemble, but don't link (like -c in rustc)
    --no-trans     Parse and translate, but don't generate any code
    --pretty       Pretty-print the code, but don't generate output
    --parse-only   Parse the code, but don't typecheck or generate code
    -S             Generate assembly code, but don't assemble or link it
    -S --emit-llvm Generate LLVM assembly code
    --emit-llvm    Generate LLVM bitcode
    --linker PATH  Use a linker other than the system linker
    --link-args [ARG..] Extra arguments to pass to the linker
    --opt-level=n  Set the optimization level (0 <= n <= 3)
    -O             Equivalent to --opt-level=2
    --save-temps   Don't delete temporary files
    --target TRIPLE Set the target triple
    --target-cpu CPU Set the target CPU
    -Z FLAG        Enable an experimental rustc feature (see `rustc --help`)");
}

pub fn clean() {
    io::println("rustpkg clean

Remove all build files in the work cache for the package in the current
directory.");
}

pub fn do_cmd() {
    io::println("rustpkg do <cmd>

Runs a command in the package script. You can listen to a command
by tagging a function with the attribute `#[pkg_do(cmd)]`.");
}

pub fn info() {
    io::println("rustpkg [options..] info

Probe the package script in the current directory for information.

Options:
    -j, --json      Output the result as JSON");
}

pub fn list() {
    io::println("rustpkg list

List all installed packages.");
}

pub fn install() {
    io::println("rustpkg install [options..] [package-ID]

Install the given package ID if specified. With no package ID
argument, install the package in the current directory.
In that case, the current directory must be a direct child of a
`src` directory in a workspace.

Examples:
    rustpkg install
    rustpkg install github.com/mozilla/servo
    rustpkg install github.com/mozilla/servo#0.1.2

Options:
    -c, --cfg      Pass a cfg flag to the package script
    --emit-llvm    Generate LLVM bitcode
    --linker PATH  Use a linker other than the system linker
    --link-args [ARG..] Extra arguments to pass to the linker
    --opt-level=n  Set the optimization level (0 <= n <= 3)
    -O             Equivalent to --opt-level=2
    --save-temps   Don't delete temporary files
    --target TRIPLE Set the target triple
    --target-cpu CPU Set the target CPU
    -Z FLAG        Enable an experimental rustc feature (see `rustc --help`)");
}

pub fn uninstall() {
    io::println("rustpkg uninstall <id|name>[@version]

Remove a package by id or name and optionally version. If the package(s)
is/are depended on by another package then they cannot be removed.");
}

pub fn prefer() {
    io::println("rustpkg [options..] prefer <id|name>[@version]

By default all binaries are given a unique name so that multiple versions can
coexist. The prefer command will symlink the uniquely named binary to
the binary directory under its bare name. If version is not supplied, the
latest version of the package will be preferred.

Example:
    export PATH=$PATH:/home/user/.rustpkg/bin
    rustpkg prefer machine@1.2.4
    machine -v
    ==> v1.2.4
    rustpkg prefer machine@0.4.6
    machine -v
    ==> v0.4.6");
}

pub fn unprefer() {
    io::println("rustpkg [options..] unprefer <id|name>[@version]

Remove all symlinks from the store to the binary directory for a package
name and optionally version. If version is not supplied, the latest version
of the package will be unpreferred. See `rustpkg prefer -h` for more
information.");
}

pub fn test() {
    io::println("rustpkg [options..] test

Build all test crates in the current directory with the test flag.
Then, run all the resulting test executables, redirecting the output
and exit code.

Options:
    -c, --cfg      Pass a cfg flag to the package script");
}

pub fn init() {
    io::println("rustpkg init

This will turn the current working directory into a workspace. The first
command you run when starting off a new project.
");
}
