// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session;
use metadata::cstore;
use metadata::filesearch;

use core::hashmap::HashSet;
use core::os;
use core::uint;
use core::util;
use core::vec;

fn not_win32(os: session::os) -> bool {
  os != session::os_win32
}

pub fn get_rpath_flags(sess: session::Session, out_filename: &Path)
                    -> ~[~str] {
    let os = sess.targ_cfg.os;

    // No rpath on windows
    if os == session::os_win32 {
        return ~[];
    }

    debug!("preparing the RPATH!");

    let sysroot = sess.filesearch.sysroot();
    let output = out_filename;
    let libs = cstore::get_used_crate_files(sess.cstore);
    // We don't currently rpath extern libraries, but we know
    // where rustrt is and we know every rust program needs it
    let libs = vec::append_one(libs, get_sysroot_absolute_rt_lib(sess));

    let rpaths = get_rpaths(os, sysroot, output, libs,
                            sess.opts.target_triple);
    rpaths_to_flags(rpaths)
}

fn get_sysroot_absolute_rt_lib(sess: session::Session) -> Path {
    let r = filesearch::relative_target_lib_path(sess.opts.target_triple);
    sess.filesearch.sysroot().push_rel(&r).push(os::dll_filename("rustrt"))
}

pub fn rpaths_to_flags(rpaths: &[Path]) -> ~[~str] {
    vec::map(rpaths, |rpath| fmt!("-Wl,-rpath,%s",rpath.to_str()))
}

fn get_rpaths(os: session::os,
              sysroot: &Path,
              output: &Path,
              libs: &[Path],
              target_triple: &str) -> ~[Path] {
    debug!("sysroot: %s", sysroot.to_str());
    debug!("output: %s", output.to_str());
    debug!("libs:");
    for libs.iter().advance |libpath| {
        debug!("    %s", libpath.to_str());
    }
    debug!("target_triple: %s", target_triple);

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rel_rpaths = get_rpaths_relative_to_output(os, output, libs);

    // Make backup absolute paths to the libraries. Binaries can
    // be moved as long as the crates they link against don't move.
    let abs_rpaths = get_absolute_rpaths(libs);

    // And a final backup rpath to the global library location.
    let fallback_rpaths = ~[get_install_prefix_rpath(target_triple)];

    fn log_rpaths(desc: &str, rpaths: &[Path]) {
        debug!("%s rpaths:", desc);
        for rpaths.iter().advance |rpath| {
            debug!("    %s", rpath.to_str());
        }
    }

    log_rpaths("relative", rel_rpaths);
    log_rpaths("absolute", abs_rpaths);
    log_rpaths("fallback", fallback_rpaths);

    let mut rpaths = rel_rpaths;
    rpaths.push_all(abs_rpaths);
    rpaths.push_all(fallback_rpaths);

    // Remove duplicates
    let rpaths = minimize_rpaths(rpaths);
    return rpaths;
}

fn get_rpaths_relative_to_output(os: session::os,
                                 output: &Path,
                                 libs: &[Path]) -> ~[Path] {
    vec::map(libs, |a| {
        get_rpath_relative_to_output(os, output, a)
    })
}

pub fn get_rpath_relative_to_output(os: session::os,
                                    output: &Path,
                                    lib: &Path)
                                 -> Path {
    use core::os;

    assert!(not_win32(os));

    // Mac doesn't appear to support $ORIGIN
    let prefix = match os {
        session::os_android | session::os_linux | session::os_freebsd
                          => "$ORIGIN",
        session::os_macos => "@executable_path",
        session::os_win32 => util::unreachable()
    };

    Path(prefix).push_rel(&get_relative_to(&os::make_absolute(output),
                                           &os::make_absolute(lib)))
}

// Find the relative path from one file to another
pub fn get_relative_to(abs1: &Path, abs2: &Path) -> Path {
    assert!(abs1.is_absolute);
    assert!(abs2.is_absolute);
    let abs1 = abs1.normalize();
    let abs2 = abs2.normalize();
    debug!("finding relative path from %s to %s",
           abs1.to_str(), abs2.to_str());
    let split1: &[~str] = abs1.components;
    let split2: &[~str] = abs2.components;
    let len1 = split1.len();
    let len2 = split2.len();
    assert!(len1 > 0);
    assert!(len2 > 0);

    let max_common_path = uint::min(len1, len2) - 1;
    let mut start_idx = 0;
    while start_idx < max_common_path
        && split1[start_idx] == split2[start_idx] {
        start_idx += 1;
    }

    let mut path = ~[];
    for uint::range(start_idx, len1 - 1) |_i| { path.push(~".."); };

    path.push_all(split2.slice(start_idx, len2 - 1));

    return if !path.is_empty() {
        Path("").push_many(path)
    } else {
        Path(".")
    }
}

fn get_absolute_rpaths(libs: &[Path]) -> ~[Path] {
    vec::map(libs, |a| get_absolute_rpath(a) )
}

pub fn get_absolute_rpath(lib: &Path) -> Path {
    os::make_absolute(lib).dir_path()
}

pub fn get_install_prefix_rpath(target_triple: &str) -> Path {
    let install_prefix = env!("CFG_PREFIX");

    if install_prefix == "" {
        fail!("rustc compiled without CFG_PREFIX environment variable");
    }

    let tlib = filesearch::relative_target_lib_path(target_triple);
    os::make_absolute(&Path(install_prefix).push_rel(&tlib))
}

pub fn minimize_rpaths(rpaths: &[Path]) -> ~[Path] {
    let mut set = HashSet::new();
    let mut minimized = ~[];
    for rpaths.iter().advance |rpath| {
        if set.insert(rpath.to_str()) {
            minimized.push(copy *rpath);
        }
    }
    minimized
}

#[cfg(unix, test)]
mod test {
    use core::prelude::*;
    use core::os;

    // FIXME(#2119): the outer attribute should be #[cfg(unix, test)], then
    // these redundant #[cfg(test)] blocks can be removed
    #[cfg(test)]
    #[cfg(test)]
    use back::rpath::{get_absolute_rpath, get_install_prefix_rpath};
    use back::rpath::{get_relative_to, get_rpath_relative_to_output};
    use back::rpath::{minimize_rpaths, rpaths_to_flags};
    use driver::session;

    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags([Path("path1"),
                                     Path("path2")]);
        assert_eq!(flags, ~[~"-Wl,-rpath,path1", ~"-Wl,-rpath,path2"]);
    }

    #[test]
    fn test_prefix_rpath() {
        let res = get_install_prefix_rpath("triple");
        let d = Path(env!("CFG_PREFIX"))
            .push_rel(&Path("lib/rustc/triple/lib"));
        debug!("test_prefix_path: %s vs. %s",
               res.to_str(),
               d.to_str());
        assert!(res.to_str().ends_with(d.to_str()));
    }

    #[test]
    fn test_prefix_rpath_abs() {
        let res = get_install_prefix_rpath("triple");
        assert!(res.is_absolute);
    }

    #[test]
    fn test_minimize1() {
        let res = minimize_rpaths([Path("rpath1"),
                                   Path("rpath2"),
                                   Path("rpath1")]);
        assert_eq!(res, ~[Path("rpath1"), Path("rpath2")]);
    }

    #[test]
    fn test_minimize2() {
        let res = minimize_rpaths([Path("1a"), Path("2"), Path("2"),
                                   Path("1a"), Path("4a"),Path("1a"),
                                   Path("2"), Path("3"), Path("4a"),
                                   Path("3")]);
        assert_eq!(res, ~[Path("1a"), Path("2"), Path("4a"), Path("3")]);
    }

    #[test]
    fn test_relative_to1() {
        let p1 = Path("/usr/bin/rustc");
        let p2 = Path("/usr/lib/mylib");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path("../lib"));
    }

    #[test]
    fn test_relative_to2() {
        let p1 = Path("/usr/bin/rustc");
        let p2 = Path("/usr/bin/../lib/mylib");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path("../lib"));
    }

    #[test]
    fn test_relative_to3() {
        let p1 = Path("/usr/bin/whatever/rustc");
        let p2 = Path("/usr/lib/whatever/mylib");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path("../../lib/whatever"));
    }

    #[test]
    fn test_relative_to4() {
        let p1 = Path("/usr/bin/whatever/../rustc");
        let p2 = Path("/usr/lib/whatever/mylib");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path("../lib/whatever"));
    }

    #[test]
    fn test_relative_to5() {
        let p1 = Path("/usr/bin/whatever/../rustc");
        let p2 = Path("/usr/lib/whatever/../mylib");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path("../lib"));
    }

    #[test]
    fn test_relative_to6() {
        let p1 = Path("/1");
        let p2 = Path("/2/3");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path("2"));
    }

    #[test]
    fn test_relative_to7() {
        let p1 = Path("/1/2");
        let p2 = Path("/3");
        let res = get_relative_to(&p1, &p2);
        assert_eq!(res, Path(".."));
    }

    #[test]
    fn test_relative_to8() {
        let p1 = Path("/home/brian/Dev/rust/build/").push_rel(
            &Path("stage2/lib/rustc/i686-unknown-linux-gnu/lib/librustc.so"));
        let p2 = Path("/home/brian/Dev/rust/build/stage2/bin/..").push_rel(
            &Path("lib/rustc/i686-unknown-linux-gnu/lib/libstd.so"));
        let res = get_relative_to(&p1, &p2);
        debug!("test_relative_tu8: %s vs. %s",
               res.to_str(),
               Path(".").to_str());
        assert_eq!(res, Path("."));
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "andorid")]
    fn test_rpath_relative() {
      let o = session::os_linux;
      let res = get_rpath_relative_to_output(o,
            &Path("bin/rustc"), &Path("lib/libstd.so"));
      assert_eq!(res.to_str(), ~"$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "freebsd")]
    fn test_rpath_relative() {
        let o = session::os_freebsd;
        let res = get_rpath_relative_to_output(o,
            &Path("bin/rustc"), &Path("lib/libstd.so"));
        assert_eq!(res.to_str(), ~"$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_rpath_relative() {
        // this is why refinements would be nice
        let o = session::os_macos;
        let res = get_rpath_relative_to_output(o,
                                               &Path("bin/rustc"),
                                               &Path("lib/libstd.so"));
        assert_eq!(res.to_str(), ~"@executable_path/../lib");
    }

    #[test]
    fn test_get_absolute_rpath() {
        let res = get_absolute_rpath(&Path("lib/libstd.so"));
        debug!("test_get_absolute_rpath: %s vs. %s",
               res.to_str(),
               os::make_absolute(&Path("lib")).to_str());

        assert_eq!(res, os::make_absolute(&Path("lib")));
    }
}
