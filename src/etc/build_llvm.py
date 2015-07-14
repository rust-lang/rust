# Copyright 2011-2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os
import subprocess
import multiprocessing

def build_llvm_cmake(src_dir, build_dir, target, ncpu,
                     is_release, is_assert):
    if "msvc" in target:
        cmake_target = "Visual Studio 12"
    elif "windows-gnu" in target:
        cmake_target = "MinGW Makefiles"
    else:
        cmake_target = "Unix Makefiles"
    if is_release:
        cmake_build_type = "-DCMAKE_BUILD_TYPE=Release"
    else:
        cmake_build_type = "-DCMAKE_BUILD_TYPE=Debug"
    if is_assert:
        cmake_assert = "-DLLVM_ENABLE_ASSERTIONS=ON"
    else:
        cmake_assert = "-DLLVM_ENABLE_ASSERTIONS=OFF"
    ret = subprocess.call(["cmake", "-G", cmake_target,
                           cmake_build_type, cmake_assert,
                           "-DLLVM_ENABLE_TERMINFO=OFF",
                           "-DLLVM_ENABLE_ZLIB=OFF",
                           "-DLLVM_ENABLE_FFI=OFF",
                           "-DLLVM_BUILD_DOCS=OFF",
                           src_dir], cwd = build_dir)
    if ret != 0:
        return ret
    if "msvc" in target:
        build_cmd = ["cmake", "--build", "."]
    else:
        build_cmd = ["cmake", "--build", ".", "--", "-j"+str(ncpu)]
    return subprocess.call(build_cmd, cwd = build_dir)

def build_llvm_autotools(src_dir, build_dir, target, ncpu,
                         is_release, is_assert):
    if is_release:
        optimized = "--enable-optimized"
    else:
        optimized = "--disable-optimized"
    if is_assert:
        assertions = "--enable-assertions"
    else:
        assertions = "--disable-assertions"
    ret = subprocess.call([os.path.join(src_dir, "configure"),
                           "--enable-targets=x86,x86_64,arm,aarch64,mips,powerpc",
                           optimized, assertions,
                           "--disable-docs", "--enable-bindings=none",
                           "--disable-terminfo", "--disable-zlib",
                           "--disable-libffi",
                           "--host=" + target, "--target=" + target,
                           "--with-python=/usr/bin/python2.7"],
                          cwd = build_dir)
    if ret != 0:
        return ret
    return subprocess.call(["make", "-j"+str(ncpu)], cwd = build_dir)

def llvm_build_dir(rust_root, target):
    return os.path.join(rust_root, "target", "llvm", target)

def llvm_build_artifacts_dir(rust_root, target, is_release, is_assert):
    build_dir = llvm_build_dir(rust_root, target)
    if "windows" in target:     # cmake puts everything under build_dir
        return build_dir
    else:
        if is_release:
            subdir = "Release"
        else:
            subdir = "Debug"
        if is_assert:
            subdir += "+Asserts"
        return os.path.join(build_dir, subdir)

def build_llvm(rust_root, target, force_rebuild, is_release, is_assert):
    if is_release:
        profile = "Release"
    else:
        profile = "Debug"
    if is_assert:
        profile += "+Asserts"
    print("Building LLVM for target " + target + " profile " + profile + ":")
    src_dir = os.path.join(rust_root, "src", "llvm")
    build_dir = llvm_build_dir(rust_root, target)
    # create build dir
    try:
        os.makedirs(build_dir)
    except OSError:
        if not os.path.isdir(build_dir):
            raise
    # use a stamp file to avoid rebuilding llvm
    build_artifacts_dir = llvm_build_artifacts_dir(rust_root, target,
                                                   is_release, is_assert)
    stamp_file = os.path.join(build_artifacts_dir, "llvm.built.for.rust")
    if os.path.isfile(stamp_file) and not force_rebuild:
        print("Skipped. Use --rebuild-llvm to override.")
        return                  # no need to build llvm here.
    ncpu = multiprocessing.cpu_count()
    # build llvm
    if "windows" in target:
        ret = build_llvm_cmake(src_dir, build_dir, target, ncpu,
                               is_release, is_assert)
    else:
        ret = build_llvm_autotools(src_dir, build_dir, target,
                                   ncpu, is_release, is_assert)
    if ret != 0:
        print("Build failed.")
        exit(ret)
    # make a note so that we don't rebuild llvm
    with open(stamp_file, "w") as f:
        f.write("built")

def get_llvm_bin_dir(rust_root, target, external_llvm_root):
    if external_llvm_root:
        llvm_root = external_llvm_root
    else:
        llvm_root = os.path.join(rust_root, "target", "llvm", target)
    dirs = [".", "Release", "Release+Asserts", "Debug", "Debug+Asserts"]
    for d in dirs:
        bin_dir = os.path.join(llvm_root, d, "bin")
        f = os.path.join(bin_dir, "llc")
        fwin = os.path.join(bin_dir, "llc.exe")
        if os.path.isfile(f) or os.path.isfile(fwin):
            return bin_dir
    print("Path " + llvm_root + " does not contain valid LLVM build.")
    exit(1)
