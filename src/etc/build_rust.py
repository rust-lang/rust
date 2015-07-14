#!/usr/bin/python

from __future__ import print_function

import sys
import os
import shutil
import glob
import subprocess
import argparse
import build_llvm

def scrub(b):
    if sys.version_info >= (3,) and type(b) == bytes:
        return b.decode("ascii")
    else:
        return b

def cmd_out(cmdline):
    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE)
    return scrub(p.communicate()[0].strip())

def scrape_rustc_host_triple():
    output = cmd_out(["rustc", "-Vv"]).split()
    return output[output.index("host:")+1]

def git_commit_hash():
    return cmd_out(["git", "rev-parse", "HEAD"])

def git_short_hash():
    return cmd_out(["git", "rev-parse", "--short", "HEAD"])

def git_short_date():
    return cmd_out(["git", "show", "-s", "--format=%cd", "--date=short"])

def set_release_channel(channel):
    RELEASE = "1.3.0-" + channel
    VERSION = RELEASE + " (" + git_short_hash() + " " + git_short_date() + ")"
    os.environ["CFG_VERSION"] = VERSION
    os.environ["CFG_VER_HASH"] = git_commit_hash()
    os.environ["CFG_VER_DATE"] = git_short_date()
    os.environ["CFG_RELEASE"] = RELEASE
    if channel in ["beta", "stable"]:
        os.environ["CFG_DISABLE_UNSTABLE_FEATURES"] = "1"
        # set the bootstrap key to subvert the feature gating during build
        os.environ["CFG_BOOTSTRAP_KEY"] = "5196bb7834f331542c9875f3059"
        os.environ["RUSTC_BOOTSTRAP_KEY"] = "5196bb7834f331542c9875f3059"
    os.environ["CFG_RELEASE_CHANNEL"] = channel

def set_env_vars(rust_root, target, external_llvm_root):
    if external_llvm_root:
        llvm_root = external_llvm_root
    else:
        llvm_root = build_llvm.llvm_build_dir(rust_root, target)
    if "windows" in target:
        os.environ["CFG_LIBDIR_RELATIVE"] = "bin"
    else:
        os.environ["CFG_LIBDIR_RELATIVE"] = "lib"
    os.environ["CFG_PREFIX"] = "/"
    os.environ["CFG_COMPILER_HOST_TRIPLE"] = target
    os.environ["CARGO_TARGET_DIR"] = os.path.join(rust_root, "target")
    os.environ["CFG_LLVM_ROOT"] = llvm_root

# build the src/driver crate, which is the root package for the compiler
# (including rustdoc)
def build_driver_crate(rust_root, target, is_release, verbose):
    args = ["cargo", "build", "--manifest-path",
            os.path.join(rust_root, "src", "driver", "Cargo.toml")]
    args.extend(["--target", target])
    if is_release:
        args.append("--release")
    if verbose:
        args.append("--verbose")
    ret = subprocess.call(args)
    if ret == 0:
        print("Build succeeded.")
    else:
        print("Build failed.")
        exit(ret)

# build_dir refers to the <rust-root>/target directory
def copy_rust_dist(build_dir, dest_dir, host, targets, is_release):
    if is_release:
        profile = "release"
    else:
        profile = "debug"
    bin_dir = os.path.join(dest_dir, "bin")
    if "windows" in host:
        lib_dir = bin_dir
        rustc = "rustc.exe"
        rustdoc = "rustdoc.exe"
    else:
        lib_dir = os.path.join(dest_dir, "lib")
        rustc = "rustc"
        rustdoc = "rustdoc"
    shutil.rmtree(dest_dir, ignore_errors = True)
    os.makedirs(bin_dir)
    host_build_dir = os.path.join(build_dir, host, profile)
    shutil.copy2(os.path.join(host_build_dir, rustc), bin_dir)
    shutil.copy2(os.path.join(host_build_dir, rustdoc), bin_dir)
    for target in targets:
        target_build_dir = os.path.join(build_dir, target, profile)
        target_lib_dir = os.path.join(lib_dir, "rustlib", target, "lib")
        rt_lib_dir = os.path.join(target_build_dir, "build", "std*", "out")
        os.makedirs(target_lib_dir)
        copy_list = glob.glob(os.path.join(target_build_dir, "deps", "*.*"))
        copy_list.extend(glob.glob(os.path.join(rt_lib_dir, "*compiler-rt*.*")))
        if not "windows" in target:
            copy_list.extend(glob.glob(os.path.join(rt_lib_dir,
                                                    "*morestack*.*")))
        for f in copy_list:
            shutil.copy2(f, target_lib_dir)

# for now we only build libstd (and all dependencies) docs
# docs are built by the stage1 compiler
def build_rust_docs(rust_root, target, verbose):
    print("Building docs:")
    args = ["cargo", "doc", "--target", target, "--manifest-path",
            os.path.join(rust_root, "src", "libstd", "Cargo.toml")]
    if verbose:
        args.append("--verbose")
    ret = subprocess.call(args)
    if ret == 0:
        print("Cargo doc succeeded.")
    else:
        print("Cargo doc failed.")
        exit(ret)
    build_rustbook(rust_root, verbose)

# the rustbook crate is built by the stage1 compiler, as a native exe.
def build_rustbook(rust_root, verbose):
    print("Building The Rust Programming Language book:")
    args = ["cargo", "build", "--manifest-path",
            os.path.join(rust_root, "src", "rustbook", "Cargo.toml")]
    if verbose:
        args.append("--verbose")
    ret = subprocess.call(args)
    if ret != 0:
        print("Building rustbook failed.")
        exit(ret)
    rustbook_exe = os.path.join(rust_root, "target", "debug", "rustbook")
    doc_dest = os.path.join(rust_root, "target", "doc")
    shutil.rmtree(doc_dest, ignore_errors = True)
    os.makedirs(doc_dest)
    book_src = os.path.join(rust_root, "src", "doc", "trpl")
    book_dest = os.path.join(doc_dest, "book")
    ret1 = subprocess.call([rustbook_exe, "build", book_src, book_dest])
    style_src = os.path.join(rust_root, "src", "doc", "style")
    style_dest = os.path.join(doc_dest, "style")
    ret2 = subprocess.call([rustbook_exe, "build", style_src, style_dest])
    if ret1 == 0 and ret2 == 0:
        print("Done.")
    else:
        print("Rustbook failed.")
        exit(1)

def run_test_for_crate(rust_root, crate, target, verbose):
    print("Running " + crate + " tests:")
    args = ["cargo", "test", "--target", target, "--manifest-path",
            os.path.join(rust_root, "src", crate, "Cargo.toml")]
    if verbose:
        args.append("--verbose")
    return subprocess.call(args)

def run_crate_tests(rust_root, target, verbose):
    crates_to_test = ["libcoretest", "liballoc", "libarena", "libcollections",
                      "libcollectionstest", "libflate", "libfmt_macros",
                      "libgetopts", "libgraphviz", "liblog", "librand",
                      "librbml", "libstd", "libterm", "libserialize",
                      "libsyntax", "librustc", "librustc_back",
                      "librustc_bitflags", "librustc_data_structures",
                      "librustc_driver", "librustdoc", "libtest"]
    clean_build_dirs(rust_root, target, "debug")
    for crate in crates_to_test:
        if run_test_for_crate(rust_root, crate, target, verbose) != 0:
            print("Tests in crate " + crate + " failed.")

# the compiletest crate is built by the stage1 compiler, as a native exe
def run_compiletests(rust_root, target, llvm_bin_dir, verbose):
    print("Building compiletest:")
    args = ["cargo", "build", "--manifest-path",
            os.path.join(rust_root, "src", "compiletest", "Cargo.toml")]
    if verbose:
        args.append("--verbose")
    ret = subprocess.call(args)
    if ret == 0:
        print("Done.")
    else:
        print("Building compiletest failed.")
        exit(ret)
    target_dir = os.path.join(rust_root, "target")
    ctest_exe = os.path.join(target_dir, "debug", "compiletest")
    bin_dir = os.path.join(target_dir, "stage2", "bin")
    if "windows" in target:
        lib_dir = bin_dir
    else:
        lib_dir = os.path.join(target_dir, "stage2", "lib")
    rustlib_dir = os.path.join(lib_dir, "rustlib", target, "lib")
    rustc_path = os.path.join(bin_dir, "rustc")
    rustdoc_path = os.path.join(bin_dir, "rustdoc")
    aux_base = os.path.join(rust_root, "src", "test", "auxiliary")
    stage_id = "stage2-" + target
    test_helpers_lib_dir = glob.glob(
        os.path.join(target_dir, "debug", "build", "compiletest*", "out"))
    rust_flags = "--cfg rtopt -O -L " + test_helpers_lib_dir[0]
    tests_to_run = [("run-pass", "run-pass"),
                    ("run-pass", "run-pass-fulldeps"),
                    ("run-pass-valgrind", "run-pass-valgrind"),
                    ("run-fail", "run-fail"),
                    ("run-fail", "run-fail-fulldeps"),
                    ("compile-fail", "compile-fail"),
                    ("compile-fail", "compile-fail-fulldeps"),
                    ("parse-fail", "parse-fail"),
                    ("pretty", "pretty"),
                    ("debuginfo-gdb", "debuginfo")] # TODO: detect gdb/lldb
    args = [ctest_exe, "--compile-lib-path", lib_dir,
            "--run-lib-path", rustlib_dir, "--rustc-path", rustc_path,
            "--rustdoc-path", rustdoc_path, "--llvm-bin-path", llvm_bin_dir,
            "--aux-base", aux_base, "--stage-id", stage_id,
            "--target", target, "--host", host, "--python", sys.executable,
            "--gdb-version", "", "--lldb-version=", "",
            "--android-cross-path", "", "--adb-path", "", "--adb-test-dir", "",
            "--host-rustcflags", rust_flags, "--target-rustcflags", rust_flags,
            "--lldb-python-dir", ""]
    if verbose:
        args.append("--verbose")
    test_logs = os.path.join(rust_root, "target", "test-report")
    shutil.rmtree(test_logs, ignore_errors = True)
    os.makedirs(test_logs)
    for test in tests_to_run:
        src_base = os.path.join(rust_root, "src", "test", test[1])
        build_base = os.path.join(rust_root, "target", "test", test[1])
        shutil.rmtree(build_base, ignore_errors = True)
        os.makedirs(build_base)
        log_file = os.path.join(test_logs, test[1])
        final_args = args + ["--src-base", src_base,
                             "--build-base", build_base,
                             "--mode", test[0], "--logfile", log_file]
        ret = subprocess.call(final_args)
        if ret != 0:
            print("Compiler test " + test[1] + " failed.")

def build_stage1_rust(rust_root, build,
                      external_llvm_root, is_release, verbose):
    print("Building stage1 compiler:")
    set_env_vars(rust_root, build, external_llvm_root)
    build_driver_crate(rust_root, build, is_release, verbose)
    print("Copying stage1 compiler to target/stage1:")
    build_dir = os.path.join(rust_root, "target")
    dest_dir = os.path.join(build_dir, "stage1")
    copy_rust_dist(build_dir, dest_dir, build, [build], is_release)
    print("Done.")

def clean_build_dirs(rust_root, build, profile):
    dir1 = os.path.join(rust_root, "target", profile)
    dir2 = os.path.join(rust_root, "target", build, profile)
    shutil.rmtree(dir1, ignore_errors = True)
    shutil.rmtree(dir2, ignore_errors = True)

# switch to the rustc in the specified sysroot
def switch_rustc(sysroot, host):
    bin_dir = os.path.join(sysroot, "bin")
    if "windows" in host:
        lib_dir = bin_dir
        rustc = "rustc.exe"
        rustdoc = "rustdoc.exe"
    else:
        lib_dir = os.path.join(sysroot, "lib")
        rustc = "rustc"
        rustdoc = "rustdoc"
    os.environ["RUSTC"] = os.path.join(bin_dir, rustc)
    os.environ["RUSTDOC"] = os.path.join(bin_dir, rustdoc)
    rustlib_dir = os.path.join(lib_dir, "rustlib", host, "lib")
    if "windows" in host:
        os.environ["PATH"] += os.pathsep + rustlib_dir
    elif "darwin" in host:
        os.environ["DYLD_LIBRARY_PATH"] = rustlib_dir
    else:
        os.environ["LD_LIBRARY_PATH"] = rustlib_dir

def build_stage2_rust(rust_root, build, host, targets,
                      external_llvm_root, is_release, verbose):
    build_dir = os.path.join(rust_root, "target")
    stage1_dir = os.path.join(build_dir, "stage1")
    switch_rustc(stage1_dir, build)
    for target in targets:
        print("Building stage2 compiler for target " + target + ":")
        set_env_vars(rust_root, target, external_llvm_root)
        build_driver_crate(rust_root, target, is_release, verbose)
    set_env_vars(rust_root, host, llvm_root)
    build_rust_docs(rust_root, host, verbose = verbose)
    print("Copying stage2 compiler to target/stage2:")
    dest_dir = os.path.join(build_dir, "stage2")
    copy_rust_dist(build_dir, dest_dir, host, targets, is_release)
    print("Done.")


# main function

# parse command line arguments
parser = argparse.ArgumentParser(description="Build the Rust compiler.")
parser.add_argument("--verbose", action="store_true", default=False,
                    help="Pass --verbose to Cargo when building Rust.")
parser.add_argument("--host", action="store",
                    help="GNUs ./configure syntax LLVM host triple")
parser.add_argument("--target", action="append",
                    help="GNUs ./configure syntax LLVM target triples")
parser.add_argument("--release-channel", action="store", default="dev",
                    help="The name of the release channel to build.")
parser.add_argument("--enable-debug", action="store_true", default=False,
                    help="Build with debug profile. The default is a\
                    release build. Note this only applies to the\
                    compiler, not LLVM.")
parser.add_argument("--enable-llvm-debug",
                    action="store_true", default=False,
                    help="Build LLVM with debug profile. The default is a\
                    release build with no assertions.")
parser.add_argument("--enable-llvm-assertions",
                    action="store_true", default=False,
                    help="Build LLVM with assertions. Off by default for\
                    non-nightly builds.")
parser.add_argument("--no-bootstrap", action="store_true", default=False,
                    help="Do not bootstrap. Build stage2 compiler only.")
parser.add_argument("--rebuild-llvm", action="store_true", default=False,
                    help="Force rebuilding LLVM.")
parser.add_argument("--llvm-root", action="store",
                    help="Specify external LLVM root.")
parser.add_argument("--run-tests-only", action="store_true", default=False,
                    help="Run library tests only. Don't build the compiler.")
args = parser.parse_args()

# collect some essential infomation from the build environment
rust_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
build = scrape_rustc_host_triple()
if args.host:
    host = args.host
else:
    host = build
targets = [host]
if args.target:
    targets.append([x for x in args.target and x not in targets])

is_release = not args.enable_debug
llvm_release = not args.enable_llvm_debug
verbose = args.verbose
llvm_root = args.llvm_root
is_assert = args.enable_llvm_assertions
release_channel = args.release_channel

if release_channel not in ["dev", "nightly", "beta", "stable"]:
    print("Release channel must be one of 'dev', 'nightly', 'beta', 'stable'")
    exit(1)
elif release_channel == "nightly":
    print("Nightly builds always have LLVM assertions on.")
    is_assert = True
print("Building Rust for release channel " + release_channel + ":")
set_release_channel(release_channel)

# build LLVM
if llvm_root and ((build != host) or targets.len() > 1):
    print("--llvm-root is only allowed for native builds.")
    exit(1)
if not llvm_root:
    force_rebuild = args.rebuild_llvm
    llvm_targets = targets
    if build not in targets:
        llvm_targets.insert(0, build)
    for target in llvm_targets:
        build_llvm.build_llvm(rust_root, target, force_rebuild = force_rebuild,
                              is_release = llvm_release, is_assert = is_assert)

# build rustc and docs
if is_release:
    profile = "release"
else:
    profile = "debug"
if not args.run_tests_only:
    print("Building Rust with " + profile + " profile:")
    if not args.no_bootstrap:
        build_stage1_rust(rust_root, build, llvm_root,
                          is_release = is_release, verbose = verbose)
        clean_build_dirs(rust_root, build, profile)
    build_stage2_rust(rust_root, build, host, targets, llvm_root,
                      is_release = is_release, verbose = verbose)

# we only run tests for native builds
if host == build:
    set_env_vars(rust_root, host, llvm_root)
    stage2_dir = os.path.join(rust_root, "target", "stage2")
    switch_rustc(stage2_dir, host)
    llvm_bin_dir = build_llvm.get_llvm_bin_dir(rust_root, target, llvm_root)
    run_compiletests(rust_root, host, llvm_bin_dir, verbose = verbose)
    run_crate_tests(rust_root, host, verbose = verbose)
