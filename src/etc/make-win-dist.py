# Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import sys, os, shutil, subprocess

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

def make_win_dist(dist_root, target_triple):
    # Ask gcc where it keeps its' stuff
    gcc_out = subprocess.check_output(["gcc.exe", "-print-search-dirs"])
    bin_path = os.environ["PATH"].split(os.pathsep)
    lib_path = []
    for line in gcc_out.splitlines():
        key, val = line.split(':', 1)
        if key == "programs":
            bin_path.extend(val.lstrip(' =').split(';'))
        elif key == "libraries":
            lib_path.extend(val.lstrip(' =').split(';'))

    target_tools = ["gcc.exe", "ld.exe", "ar.exe", "dlltool.exe", "windres.exe"]

    rustc_dlls = ["libstdc++-6.dll"]
    if target_triple.startswith("i686-"):
        rustc_dlls.append("libgcc_s_dw2-1.dll")
    else:
        rustc_dlls.append("libgcc_s_seh-1.dll")

    target_libs = ["crtbegin.o", "crtend.o", "crt2.o", "dllcrt2.o",
                   "libadvapi32.a", "libcrypt32.a", "libgcc.a", "libgcc_eh.a", "libgcc_s.a",
                   "libimagehlp.a", "libiphlpapi.a", "libkernel32.a", "libm.a", "libmingw32.a",
                   "libmingwex.a", "libmsvcrt.a", "libpsapi.a", "libshell32.a", "libstdc++.a",
                   "libuser32.a", "libws2_32.a", "libiconv.a", "libmoldname.a"]

    # Find mingw artifacts we want to bundle
    target_tools = find_files(target_tools, bin_path)
    rustc_dlls = find_files(rustc_dlls, bin_path)
    target_libs = find_files(target_libs, lib_path)

    # Copy runtime dlls next to rustc.exe
    dist_bin_dir = os.path.join(dist_root, "bin")
    for src in rustc_dlls:
        shutil.copy(src, dist_bin_dir)

    # Copy platform tools to platform-specific bin directory
    target_bin_dir = os.path.join(dist_root, "bin", "rustlib", target_triple, "gcc", "bin")
    if not os.path.exists(target_bin_dir):
        os.makedirs(target_bin_dir)
    for src in target_tools:
        shutil.copy(src, target_bin_dir)

    # Copy platform libs to platform-spcific lib directory
    target_lib_dir = os.path.join(dist_root, "bin", "rustlib", target_triple, "gcc", "lib")
    if not os.path.exists(target_lib_dir):
        os.makedirs(target_lib_dir)
    for src in target_libs:
        shutil.copy(src, target_lib_dir)

    # Copy license files
    lic_dir = os.path.join(dist_root, "bin", "third-party")
    if os.path.exists(lic_dir):
        shutil.rmtree(lic_dir) # copytree() won't overwrite existing files
    shutil.copytree(os.path.join(os.path.dirname(__file__), "third-party"), lic_dir)

if __name__=="__main__":
    make_win_dist(sys.argv[1], sys.argv[2])
