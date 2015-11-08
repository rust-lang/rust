# Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Script parameters:
#     argv[1] = rust component root,
#     argv[2] = gcc component root,
#     argv[3] = target triple
# The first two correspond to the two installable components defined in the setup script.

import sys
import os
import shutil
import subprocess


def find_files(files, path):
    found = []
    for fname in files:
        for dir in path:
            filepath = os.path.normpath(os.path.join(dir, fname))
            if os.path.isfile(filepath):
                found.append(filepath)
                break
        else:
            raise Exception("Could not find '%s' in %s" % (fname, path))
    return found


# rust_root - root directory of the host binaries image
# plat_root - root directory of the target platform tools and libs image
#             (the two get overlayed on top of each other during installation)
# target_triple - triple of the target image being layed out
def make_win_dist(rust_root, plat_root, target_triple):
    # Ask gcc where it keeps its stuff
    gcc_out = subprocess.check_output(["gcc.exe", "-print-search-dirs"])
    bin_path = os.environ["PATH"].split(os.pathsep)
    lib_path = []
    for line in gcc_out.splitlines():
        key, val = line.split(':', 1)
        if key == "programs":
            bin_path.extend(val.lstrip(' =').split(';'))
        elif key == "libraries":
            lib_path.extend(val.lstrip(' =').split(';'))

    target_tools = ["gcc.exe", "ld.exe", "ar.exe", "dlltool.exe"]

    rustc_dlls = ["libstdc++-6.dll"]
    if target_triple.startswith("i686-"):
        rustc_dlls.append("libgcc_s_dw2-1.dll")
    else:
        rustc_dlls.append("libgcc_s_seh-1.dll")

    target_libs = [ # MinGW libs
                    "libgcc.a",
                    "libgcc_eh.a",
                    "libgcc_s.a",
                    "libm.a",
                    "libmingw32.a",
                    "libmingwex.a",
                    "libstdc++.a",
                    "libiconv.a",
                    "libmoldname.a",
                    # Windows import libs
                    "libadvapi32.a",
                    "libbcrypt.a",
                    "libcomctl32.a",
                    "libcomdlg32.a",
                    "libcrypt32.a",
                    "libgdi32.a",
                    "libimagehlp.a",
                    "libiphlpapi.a",
                    "libkernel32.a",
                    "libmsvcrt.a",
                    "libodbc32.a",
                    "libole32.a",
                    "liboleaut32.a",
                    "libopengl32.a",
                    "libpsapi.a",
                    "librpcrt4.a",
                    "libsetupapi.a",
                    "libshell32.a",
                    "libuser32.a",
                    "libuserenv.a",
                    "libuuid.a",
                    "libwinhttp.a",
                    "libwinmm.a",
                    "libwinspool.a",
                    "libws2_32.a",
                    "libwsock32.a",
                    ]

    # Find mingw artifacts we want to bundle
    target_tools = find_files(target_tools, bin_path)
    rustc_dlls = find_files(rustc_dlls, bin_path)
    target_libs = find_files(target_libs, lib_path)

    # Copy runtime dlls next to rustc.exe
    dist_bin_dir = os.path.join(rust_root, "bin")
    for src in rustc_dlls:
        shutil.copy(src, dist_bin_dir)

    # Copy platform tools to platform-specific bin directory
    target_bin_dir = os.path.join(plat_root, "lib", "rustlib", target_triple, "bin")
    if not os.path.exists(target_bin_dir):
        os.makedirs(target_bin_dir)
    for src in target_tools:
        shutil.copy(src, target_bin_dir)

    # Copy platform libs to platform-specific lib directory
    target_lib_dir = os.path.join(plat_root, "lib", "rustlib", target_triple, "lib")
    if not os.path.exists(target_lib_dir):
        os.makedirs(target_lib_dir)
    for src in target_libs:
        shutil.copy(src, target_lib_dir)

if __name__ == "__main__":
    make_win_dist(sys.argv[1], sys.argv[2], sys.argv[3])
