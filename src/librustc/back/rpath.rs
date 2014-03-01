// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session;
use metadata::cstore;
use metadata::filesearch;

use collections::HashSet;
use std::{os, vec};
use syntax::abi;

fn not_win32(os: abi::Os) -> bool {
  os != abi::OsWin32
}

pub fn get_rpath_flags(sess: session::Session, out_filename: &Path) -> ~[~str] {
    let os = sess.targ_cfg.os;

    // No rpath on windows
    if os == abi::OsWin32 {
        return ~[];
    }

    let mut flags = ~[];

    if sess.targ_cfg.os == abi::OsFreebsd {
        flags.push_all([~"-Wl,-rpath,/usr/local/lib/gcc46",
                        ~"-Wl,-rpath,/usr/local/lib/gcc44",
                        ~"-Wl,-z,origin"]);
    }

    debug!("preparing the RPATH!");

    let sysroot = sess.filesearch.sysroot;
    let output = out_filename;
    let libs = sess.cstore.get_used_crates(cstore::RequireDynamic);
    let libs = libs.move_iter().filter_map(|(_, l)| l.map(|p| p.clone())).collect();
    // We don't currently rpath extern libraries, but we know
    // where rustrt is and we know every rust program needs it
    let libs = vec::append_one(libs, get_sysroot_absolute_rt_lib(sess));

    let rpaths = get_rpaths(os, sysroot, output, libs,
                            sess.opts.target_triple);
    flags.push_all(rpaths_to_flags(rpaths));
    flags
}

fn get_sysroot_absolute_rt_lib(sess: session::Session) -> Path {
    let r = filesearch::relative_target_lib_path(sess.opts.target_triple);
    let mut p = sess.filesearch.sysroot.join(&r);
    p.push(os::dll_filename("rustrt"));
    p
}

pub fn rpaths_to_flags(rpaths: &[~str]) -> ~[~str] {
    let mut ret = ~[];
    for rpath in rpaths.iter() {
        ret.push("-Wl,-rpath," + *rpath);
    }
    return ret;
}

fn get_rpaths(os: abi::Os,
              sysroot: &Path,
              output: &Path,
              libs: &[Path],
              target_triple: &str) -> ~[~str] {
    debug!("sysroot: {}", sysroot.display());
    debug!("output: {}", output.display());
    debug!("libs:");
    for libpath in libs.iter() {
        debug!("    {}", libpath.display());
    }
    debug!("target_triple: {}", target_triple);

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rel_rpaths = get_rpaths_relative_to_output(os, output, libs);

    // Make backup absolute paths to the libraries. Binaries can
    // be moved as long as the crates they link against don't move.
    let abs_rpaths = get_absolute_rpaths(libs);

    // And a final backup rpath to the global library location.
    let fallback_rpaths = ~[get_install_prefix_rpath(target_triple)];

    fn log_rpaths(desc: &str, rpaths: &[~str]) {
        debug!("{} rpaths:", desc);
        for rpath in rpaths.iter() {
            debug!("    {}", *rpath);
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

fn get_rpaths_relative_to_output(os: abi::Os,
                                 output: &Path,
                                 libs: &[Path]) -> ~[~str] {
    libs.iter().map(|a| get_rpath_relative_to_output(os, output, a)).collect()
}

pub fn get_rpath_relative_to_output(os: abi::Os,
                                    output: &Path,
                                    lib: &Path)
                                 -> ~str {
    use std::os;

    assert!(not_win32(os));

    // Mac doesn't appear to support $ORIGIN
    let prefix = match os {
        abi::OsAndroid | abi::OsLinux | abi::OsFreebsd
                          => "$ORIGIN",
        abi::OsMacos => "@loader_path",
        abi::OsWin32 => unreachable!()
    };

    let mut lib = os::make_absolute(lib);
    lib.pop();
    let mut output = os::make_absolute(output);
    output.pop();
    let relative = lib.path_relative_from(&output);
    let relative = relative.expect("could not create rpath relative to output");
    // FIXME (#9639): This needs to handle non-utf8 paths
    prefix+"/"+relative.as_str().expect("non-utf8 component in path")
}

fn get_absolute_rpaths(libs: &[Path]) -> ~[~str] {
    libs.iter().map(|a| get_absolute_rpath(a)).collect()
}

pub fn get_absolute_rpath(lib: &Path) -> ~str {
    let mut p = os::make_absolute(lib);
    p.pop();
    // FIXME (#9639): This needs to handle non-utf8 paths
    p.as_str().expect("non-utf8 component in rpath").to_owned()
}

pub fn get_install_prefix_rpath(target_triple: &str) -> ~str {
    let install_prefix = env!("CFG_PREFIX");

    let tlib = filesearch::relative_target_lib_path(target_triple);
    let mut path = Path::new(install_prefix);
    path.push(&tlib);
    let path = os::make_absolute(&path);
    // FIXME (#9639): This needs to handle non-utf8 paths
    path.as_str().expect("non-utf8 component in rpath").to_owned()
}

pub fn minimize_rpaths(rpaths: &[~str]) -> ~[~str] {
    let mut set = HashSet::new();
    let mut minimized = ~[];
    for rpath in rpaths.iter() {
        if set.insert(rpath.as_slice()) {
            minimized.push(rpath.clone());
        }
    }
    minimized
}

#[cfg(unix, test)]
mod test {
    use std::os;

    use back::rpath::{get_absolute_rpath, get_install_prefix_rpath};
    use back::rpath::{minimize_rpaths, rpaths_to_flags, get_rpath_relative_to_output};
    use syntax::abi;
    use metadata::filesearch;

    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags([~"path1", ~"path2"]);
        assert_eq!(flags, ~[~"-Wl,-rpath,path1", ~"-Wl,-rpath,path2"]);
    }

    #[test]
    fn test_prefix_rpath() {
        let res = get_install_prefix_rpath("triple");
        let mut d = Path::new(env!("CFG_PREFIX"));
        d.push("lib");
        d.push(filesearch::rustlibdir());
        d.push("triple/lib");
        debug!("test_prefix_path: {} vs. {}",
               res,
               d.display());
        assert!(res.as_bytes().ends_with(d.as_vec()));
    }

    #[test]
    fn test_prefix_rpath_abs() {
        let res = get_install_prefix_rpath("triple");
        assert!(Path::new(res).is_absolute());
    }

    #[test]
    fn test_minimize1() {
        let res = minimize_rpaths([~"rpath1", ~"rpath2", ~"rpath1"]);
        assert!(res.as_slice() == [~"rpath1", ~"rpath2"]);
    }

    #[test]
    fn test_minimize2() {
        let res = minimize_rpaths([~"1a", ~"2",  ~"2",
                                   ~"1a", ~"4a", ~"1a",
                                   ~"2",  ~"3",  ~"4a",
                                   ~"3"]);
        assert!(res.as_slice() == [~"1a", ~"2", ~"4a", ~"3"]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn test_rpath_relative() {
      let o = abi::OsLinux;
      let res = get_rpath_relative_to_output(o,
            &Path::new("bin/rustc"), &Path::new("lib/libstd.so"));
      assert_eq!(res.as_slice(), "$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "freebsd")]
    fn test_rpath_relative() {
        let o = abi::OsFreebsd;
        let res = get_rpath_relative_to_output(o,
            &Path::new("bin/rustc"), &Path::new("lib/libstd.so"));
        assert_eq!(res.as_slice(), "$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_rpath_relative() {
        let o = abi::OsMacos;
        let res = get_rpath_relative_to_output(o,
                                               &Path::new("bin/rustc"),
                                               &Path::new("lib/libstd.so"));
        assert_eq!(res.as_slice(), "@loader_path/../lib");
    }

    #[test]
    fn test_get_absolute_rpath() {
        let res = get_absolute_rpath(&Path::new("lib/libstd.so"));
        let lib = os::make_absolute(&Path::new("lib"));
        debug!("test_get_absolute_rpath: {} vs. {}",
               res.to_str(), lib.display());

        // FIXME (#9639): This needs to handle non-utf8 paths
        assert_eq!(res.as_slice(), lib.as_str().expect("non-utf8 component in path"));
    }
}
