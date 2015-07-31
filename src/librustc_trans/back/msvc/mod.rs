// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! MSVC-specific logic for linkers and such.
//!
//! This module contains a cross-platform interface but has a blank unix
//! implementation. The Windows implementation builds on top of Windows native
//! libraries (reading registry keys), so it otherwise wouldn't link on unix.
//!
//! Note that we don't have much special logic for finding the system linker on
//! any other platforms, so it may seem a little odd to single out MSVC to have
//! a good deal of code just to find the linker. Unlike Unix systems, however,
//! the MSVC linker is not in the system PATH by default. It also additionally
//! needs a few environment variables or command line flags to be able to link
//! against system libraries.
//!
//! In order to have a nice smooth experience on Windows, the logic in this file
//! is here to find the MSVC linker and set it up in the default configuration
//! one would need to set up anyway. This means that the Rust compiler can be
//! run not only in the developer shells of MSVC but also the standard cmd.exe
//! shell or MSYS shells.
//!
//! As a high-level note, all logic in this module for looking up various
//! paths/files is copied over from Clang in its MSVCToolChain.cpp file, but
//! comments can also be found below leading through the various code paths.

use std::process::Command;
use session::Session;

#[cfg(windows)]
mod registry;

#[cfg(windows)]
pub fn link_exe_cmd(sess: &Session) -> Command {
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
    use self::registry::{RegistryKey, LOCAL_MACHINE};

    // When finding the link.exe binary the 32-bit version is at the top level
    // but the versions to cross to other architectures are stored in
    // sub-folders. Unknown architectures also just bail out early to return the
    // standard `link.exe` command.
    let extra = match &sess.target.target.arch[..] {
        "x86" => "",
        "x86_64" => "amd64",
        "arm" => "arm",
        _ => return Command::new("link.exe"),
    };

    let vs_install_dir = get_vs_install_dir();

    // First up, we need to find the `link.exe` binary itself, and there's a few
    // locations that we can look. First up is the standard VCINSTALLDIR
    // environment variable which is normally set by the vcvarsall.bat file. If
    // an environment is set up manually by whomever's driving the compiler then
    // we shouldn't muck with that decision and should instead respect that.
    //
    // Next up is looking in PATH itself. Here we look for `cl.exe` and then
    // assume that `link.exe` is next to it if we find it. Note that we look for
    // `cl.exe` because MinGW ships /usr/bin/link.exe which is normally found in
    // PATH but we're not interested in finding that.
    //
    // Finally we read the Windows registry to discover the VS install root.
    // From here we probe for `link.exe` just to make sure that it exists.
    let mut cmd = env::var_os("VCINSTALLDIR").and_then(|dir| {
        let mut p = PathBuf::from(dir);
        p.push("bin");
        p.push(extra);
        p.push("link.exe");
        if fs::metadata(&p).is_ok() {Some(p)} else {None}
    }).or_else(|| {
        env::var_os("PATH").and_then(|path| {
            env::split_paths(&path).find(|path| {
                fs::metadata(&path.join("cl.exe")).is_ok()
            }).map(|p| {
                p.join("link.exe")
            })
        })
    }).or_else(|| {
        vs_install_dir.as_ref().and_then(|p| {
            let mut p = p.join("VC/bin");
            p.push(extra);
            p.push("link.exe");
            if fs::metadata(&p).is_ok() {Some(p)} else {None}
        })
    }).map(|linker| {
        Command::new(linker)
    }).unwrap_or_else(|| {
        Command::new("link.exe")
    });

    // The MSVC linker uses the LIB environment variable as the default lookup
    // path for libraries. This environment variable is normally set up by the
    // VS shells, so we only want to start adding our own pieces if it's not
    // set.
    //
    // If we're adding our own pieces, then we need to add a few primary
    // directories to the default search path for the linker. The first is in
    // the VS install direcotry, the next is the Windows SDK directory, and the
    // last is the possible UCRT installation directory.
    //
    // The UCRT is a recent addition to Visual Studio installs (2015 at the time
    // of this writing), and it's in the normal windows SDK folder, but there
    // apparently aren't registry keys pointing to it. As a result we detect the
    // installation and then add it manually. This logic will probably need to
    // be tweaked over time...
    if env::var_os("LIB").is_none() {
        if let Some(mut vs_install_dir) = vs_install_dir {
            vs_install_dir.push("VC/lib");
            vs_install_dir.push(extra);
            let mut arg = OsString::from("/LIBPATH:");
            arg.push(&vs_install_dir);
            cmd.arg(arg);

            if let Some((ucrt_root, vers)) = ucrt_install_dir(&vs_install_dir) {
                if let Some(arch) = windows_sdk_v8_subdir(sess) {
                    let mut arg = OsString::from("/LIBPATH:");
                    arg.push(ucrt_root.join("Lib").join(vers)
                                      .join("ucrt").join(arch));
                    cmd.arg(arg);
                }
            }
        }
        if let Some(path) = get_windows_sdk_lib_path(sess) {
            let mut arg = OsString::from("/LIBPATH:");
            arg.push(&path);
            cmd.arg(arg);
        }
    }

    return cmd;

    // When looking for the Visual Studio installation directory we look in a
    // number of locations in varying degrees of precedence:
    //
    // 1. The Visual Studio registry keys
    // 2. The Visual Studio Express registry keys
    // 3. A number of somewhat standard environment variables
    //
    // If we find a hit from any of these keys then we strip off the IDE/Tools
    // folders which are typically found at the end.
    //
    // As a final note, when we take a look at the registry keys they're
    // typically found underneath the version of what's installed, but we don't
    // quite know what's installed. As a result we probe all sub-keys of the two
    // keys we're looking at to find out the maximum version of what's installed
    // and we use that root directory.
    fn get_vs_install_dir() -> Option<PathBuf> {
        LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\VisualStudio".as_ref()).or_else(|_| {
            LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\VCExpress".as_ref())
        }).ok().and_then(|key| {
            max_version(&key).and_then(|(_vers, key)| {
                key.query_str("InstallDir").ok()
            })
        }).or_else(|| {
            env::var_os("VS120COMNTOOLS")
        }).or_else(|| {
            env::var_os("VS100COMNTOOLS")
        }).or_else(|| {
            env::var_os("VS90COMNTOOLS")
        }).or_else(|| {
            env::var_os("VS80COMNTOOLS")
        }).map(PathBuf::from).and_then(|mut dir| {
            if dir.ends_with("Common7/IDE") || dir.ends_with("Common7/Tools") {
                dir.pop();
                dir.pop();
                Some(dir)
            } else {
                None
            }
        })
    }

    // Given a registry key, look at all the sub keys and find the one which has
    // the maximal numeric value.
    //
    // Returns the name of the maximal key as well as the opened maximal key.
    fn max_version(key: &RegistryKey) -> Option<(OsString, RegistryKey)> {
        let mut max_vers = 0;
        let mut max_key = None;
        for subkey in key.iter().filter_map(|k| k.ok()) {
            let val = subkey.to_str().and_then(|s| {
                s.trim_left_matches("v").replace(".", "").parse().ok()
            });
            let val = match val {
                Some(s) => s,
                None => continue,
            };
            if val > max_vers {
                if let Ok(k) = key.open(&subkey) {
                    max_vers = val;
                    max_key = Some((subkey, k));
                }
            }
        }
        return max_key
    }

    fn get_windows_sdk_path() -> Option<(PathBuf, usize, Option<OsString>)> {
        let key = r"SOFTWARE\Microsoft\Microsoft SDKs\Windows";
        let key = LOCAL_MACHINE.open(key.as_ref());
        let (n, k) = match key.ok().as_ref().and_then(max_version) {
            Some(p) => p,
            None => return None,
        };
        let mut parts = n.to_str().unwrap().trim_left_matches("v").splitn(2, ".");
        let major = parts.next().unwrap().parse::<usize>().unwrap();
        let _minor = parts.next().unwrap().parse::<usize>().unwrap();
        k.query_str("InstallationFolder").ok().map(|folder| {
            let ver = k.query_str("ProductVersion");
            (PathBuf::from(folder), major, ver.ok())
        })
    }

    fn get_windows_sdk_lib_path(sess: &Session) -> Option<PathBuf> {
        let (mut path, major, ver) = match get_windows_sdk_path() {
            Some(p) => p,
            None => return None,
        };
        path.push("Lib");
        if major <= 7 {
            // In Windows SDK 7.x, x86 libraries are directly in the Lib folder,
            // x64 libraries are inside, and it's not necessary to link against
            // the SDK 7.x when targeting ARM or other architectures.
            let x86 = match &sess.target.target.arch[..] {
                "x86" => true,
                "x86_64" => false,
                _ => return None,
            };
            Some(if x86 {path} else {path.join("x64")})
        } else if major <= 8 {
            // Windows SDK 8.x installs libraries in a folder whose names
            // depend on the version of the OS you're targeting. By default
            // choose the newest, which usually corresponds to the version of
            // the OS you've installed the SDK on.
            let extra = match windows_sdk_v8_subdir(sess) {
                Some(e) => e,
                None => return None,
            };
            ["winv6.3", "win8", "win7"].iter().map(|p| path.join(p)).find(|part| {
                fs::metadata(part).is_ok()
            }).map(|path| {
                path.join("um").join(extra)
            })
        } else if let Some(mut ver) = ver {
            // Windows SDK 10 splits the libraries into architectures the same
            // as Windows SDK 8.x, except for the addition of arm64.
            // Additionally, the SDK 10 is split by Windows 10 build numbers
            // rather than the OS version like the SDK 8.x does.
            let extra = match windows_sdk_v10_subdir(sess) {
                Some(e) => e,
                None => return None,
            };
            // To get the correct directory we need to get the Windows SDK 10
            // version, and so far it looks like the "ProductVersion" of the SDK
            // corresponds to the folder name that the libraries are located in
            // except that the folder contains an extra ".0". For now just
            // append a ".0" to look for find the directory we're in. This logic
            // will likely want to be refactored one day.
            ver.push(".0");
            let p = path.join(ver).join("um").join(extra);
            fs::metadata(&p).ok().map(|_| p)
        } else { None }
    }

    fn windows_sdk_v8_subdir(sess: &Session) -> Option<&'static str> {
        match &sess.target.target.arch[..] {
            "x86" => Some("x86"),
            "x86_64" => Some("x64"),
            "arm" => Some("arm"),
            _ => return None,
        }
    }

    fn windows_sdk_v10_subdir(sess: &Session) -> Option<&'static str> {
        match &sess.target.target.arch[..] {
            "x86" => Some("x86"),
            "x86_64" => Some("x64"),
            "arm" => Some("arm"),
            "aarch64" => Some("arm64"), // FIXME - Check if aarch64 is correct
            _ => return None,
        }
    }

    fn ucrt_install_dir(vs_install_dir: &Path) -> Option<(PathBuf, String)> {
        let is_vs_14 = vs_install_dir.iter().filter_map(|p| p.to_str()).any(|s| {
            s == "Microsoft Visual Studio 14.0"
        });
        if !is_vs_14 {
            return None
        }
        let key = r"SOFTWARE\Microsoft\Windows Kits\Installed Roots";
        let sdk_dir = LOCAL_MACHINE.open(key.as_ref()).and_then(|p| {
            p.query_str("KitsRoot10")
        }).map(PathBuf::from);
        let sdk_dir = match sdk_dir {
            Ok(p) => p,
            Err(..) => return None,
        };
        (move || -> io::Result<_> {
            let mut max = None;
            let mut max_s = None;
            for entry in try!(fs::read_dir(&sdk_dir.join("Lib"))) {
                let entry = try!(entry);
                if let Ok(s) = entry.file_name().into_string() {
                    if let Ok(u) = s.replace(".", "").parse::<usize>() {
                        if Some(u) > max {
                            max = Some(u);
                            max_s = Some(s);
                        }
                    }
                }
            }
            Ok(max_s.map(|m| (sdk_dir, m)))
        })().ok().and_then(|x| x)
    }
}

#[cfg(not(windows))]
pub fn link_exe_cmd(_sess: &Session) -> Command {
    Command::new("link.exe")
}
