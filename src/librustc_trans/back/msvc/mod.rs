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
//! paths/files is based on Microsoft's logic in their vcvars bat files, but
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
    use std::path::{Path, PathBuf};
    use self::registry::{LOCAL_MACHINE};

    let arch = &sess.target.target.arch;
    let (binsub, libsub, vclibsub) =
        match (bin_subdir(arch), lib_subdir(arch), vc_lib_subdir(arch)) {
        (Some(x), Some(y), Some(z)) => (x, y, z),
        _ => return Command::new("link.exe"),
    };

    // First we need to figure out whether the environment is already correctly
    // configured by vcvars. We do this by looking at the environment variable
    // `VCINSTALLDIR` which is always set by vcvars, and unlikely to be set
    // otherwise. If it is defined, then we derive the path to `link.exe` from
    // that and trust that everything else is configured correctly.
    //
    // If `VCINSTALLDIR` wasn't defined (or we couldn't find the linker where it
    // claimed it should be), then we resort to finding everything ourselves.
    // First we find where the latest version of MSVC is installed and what
    // version it is. Then based on the version we find the appropriate SDKs.
    //
    // For MSVC 14 (VS 2015) we look for the Win10 SDK and failing that we look
    // for the Win8.1 SDK. We also look for the Universal CRT.
    //
    // For MSVC 12 (VS 2013) we look for the Win8.1 SDK.
    //
    // For MSVC 11 (VS 2012) we look for the Win8 SDK.
    //
    // For all other versions the user has to execute the appropriate vcvars bat
    // file themselves to configure the environment.
    //
    // If despite our best efforts we are still unable to find MSVC then we just
    // blindly call `link.exe` and hope for the best.
    return env::var_os("VCINSTALLDIR").and_then(|dir| {
        debug!("Environment already configured by user. Assuming it works.");
        let mut p = PathBuf::from(dir);
        p.push("bin");
        p.push(binsub);
        p.push("link.exe");
        if !p.is_file() { return None }
        Some(Command::new(p))
    }).or_else(|| {
        get_vc_dir().and_then(|(ver, vcdir)| {
            debug!("Found VC installation directory {:?}", vcdir);
            let mut linker = vcdir.clone();
            linker.push("bin");
            linker.push(binsub);
            linker.push("link.exe");
            if !linker.is_file() { return None }
            let mut cmd = Command::new(linker);
            add_lib(&mut cmd, &vcdir.join("lib").join(vclibsub));
            if ver == "14.0" {
                if let Some(dir) = get_ucrt_dir() {
                    debug!("Found Universal CRT {:?}", dir);
                    add_lib(&mut cmd, &dir.join("ucrt").join(libsub));
                }
                if let Some(dir) = get_sdk10_dir() {
                    debug!("Found Win10 SDK {:?}", dir);
                    add_lib(&mut cmd, &dir.join("um").join(libsub));
                } else if let Some(dir) = get_sdk81_dir() {
                    debug!("Found Win8.1 SDK {:?}", dir);
                    add_lib(&mut cmd, &dir.join("um").join(libsub));
                }
            } else if ver == "12.0" {
                if let Some(dir) = get_sdk81_dir() {
                    debug!("Found Win8.1 SDK {:?}", dir);
                    add_lib(&mut cmd, &dir.join("um").join(libsub));
                }
            } else { // ver == "11.0"
                if let Some(dir) = get_sdk8_dir() {
                    debug!("Found Win8 SDK {:?}", dir);
                    add_lib(&mut cmd, &dir.join("um").join(libsub));
                }
            }
            Some(cmd)
        })
    }).unwrap_or_else(|| {
        debug!("Failed to locate linker.");
        Command::new("link.exe")
    });

    // A convenience function to make the above code simpler
    fn add_lib(cmd: &mut Command, lib: &Path) {
        let mut arg: OsString = "/LIBPATH:".into();
        arg.push(lib);
        cmd.arg(arg);
    }

    // To find MSVC we look in a specific registry key for the newest of the
    // three versions that we support.
    fn get_vc_dir() -> Option<(&'static str, PathBuf)> {
        LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\VisualStudio\SxS\VC7".as_ref())
        .ok().and_then(|key| {
            ["14.0", "12.0", "11.0"].iter().filter_map(|ver| {
                key.query_str(ver).ok().map(|p| (*ver, p.into()))
            }).next()
        })
    }

    // To find the Universal CRT we look in a specific registry key for where
    // all the Universal CRTs are located and then sort them asciibetically to
    // find the newest version. While this sort of sorting isn't ideal,  it is
    // what vcvars does so that's good enough for us.
    fn get_ucrt_dir() -> Option<PathBuf> {
        LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\Windows Kits\Installed Roots".as_ref())
        .ok().and_then(|key| {
            key.query_str("KitsRoot10").ok()
        }).and_then(|root| {
            fs::read_dir(Path::new(&root).join("Lib")).ok()
        }).and_then(|readdir| {
            let mut dirs: Vec<_> = readdir.filter_map(|dir| {
                dir.ok()
            }).map(|dir| {
                dir.path()
            }).filter(|dir| {
                dir.components().last().and_then(|c| {
                    c.as_os_str().to_str()
                }).map(|c| c.starts_with("10.")).unwrap_or(false)
            }).collect();
            dirs.sort();
            dirs.pop()
        })
    }

    // Vcvars finds the correct version of the Windows 10 SDK by looking
    // for the include um/Windows.h because sometimes a given version will
    // only have UCRT bits without the rest of the SDK. Since we only care about
    // libraries and not includes, we just look for the folder `um` in the lib
    // section. Like we do for the Universal CRT, we sort the possibilities
    // asciibetically to find the newest one as that is what vcvars does.
    fn get_sdk10_dir() -> Option<PathBuf> {
        LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\Microsoft SDKs\Windows\v10.0".as_ref())
        .ok().and_then(|key| {
            key.query_str("InstallationFolder").ok()
        }).and_then(|root| {
            fs::read_dir(Path::new(&root).join("lib")).ok()
        }).and_then(|readdir| {
            let mut dirs: Vec<_> = readdir.filter_map(|dir| dir.ok())
                .map(|dir| dir.path()).collect();
            dirs.sort();
            dirs.into_iter().rev().filter(|dir| {
                dir.join("um").is_dir()
            }).next()
        })
    }

    // Interestingly there are several subdirectories, `win7` `win8` and
    // `winv6.3`. Vcvars seems to only care about `winv6.3` though, so the same
    // applies to us. Note that if we were targetting kernel mode drivers
    // instead of user mode applications, we would care.
    fn get_sdk81_dir() -> Option<PathBuf> {
        LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\Microsoft SDKs\Windows\v8.1".as_ref())
        .ok().and_then(|key| {
            key.query_str("InstallationFolder").ok()
        }).map(|root| {
            Path::new(&root).join("lib").join("winv6.3")
        })
    }

    fn get_sdk8_dir() -> Option<PathBuf> {
        LOCAL_MACHINE.open(r"SOFTWARE\Microsoft\Microsoft SDKs\Windows\v8.0".as_ref())
        .ok().and_then(|key| {
            key.query_str("InstallationFolder").ok()
        }).map(|root| {
            Path::new(&root).join("lib").join("win8")
        })
    }

    // When choosing the linker toolchain to use, we have to choose the one
    // which matches the host architecture. Otherwise we end up in situations
    // where someone on 32-bit Windows is trying to cross compile to 64-bit and
    // it tries to invoke the native 64-bit linker which won't work.
    //
    // FIXME - This currently functions based on the host architecture of rustc
    // itself but it should instead detect the bitness of the OS itself.
    //
    // FIXME - Figure out what happens when the host architecture is arm.
    //
    // FIXME - Some versions of MSVC may not come with all these toolchains.
    // Consider returning an array of toolchains and trying them one at a time
    // until the linker is found.
    fn bin_subdir(arch: &str) -> Option<&'static str> {
        if cfg!(target_arch = "x86_64") {
            match arch {
                "x86" => Some("amd64_x86"),
                "x86_64" => Some("amd64"),
                "arm" => Some("amd64_arm"),
                _ => None,
            }
        } else if cfg!(target_arch = "x86") {
            match arch {
                "x86" => Some(""),
                "x86_64" => Some("x86_amd64"),
                "arm" => Some("x86_arm"),
                _ => None,
            }
        } else { None }
    }
    fn lib_subdir(arch: &str) -> Option<&'static str> {
        match arch {
            "x86" => Some("x86"),
            "x86_64" => Some("x64"),
            "arm" => Some("arm"),
            _ => None,
        }
    }
    // MSVC's x86 libraries are not in a subfolder
    fn vc_lib_subdir(arch: &str) -> Option<&'static str> {
        match arch {
            "x86" => Some(""),
            "x86_64" => Some("amd64"),
            "arm" => Some("arm"),
            _ => None,
        }
    }
}

// If we're not on Windows, then there's no registry to search through and MSVC
// wouldn't be able to run, so we just call `link.exe` and hope for the best.
#[cfg(not(windows))]
pub fn link_exe_cmd(_sess: &Session) -> Command {
    Command::new("link.exe")
}
