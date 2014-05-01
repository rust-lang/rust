// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::Session;
use metadata::cstore;
use metadata::filesearch;
use util::fs;

use collections::HashSet;
use std::os;
use syntax::abi;

fn not_win32(os: abi::Os) -> bool {
  os != abi::OsWin32
}

pub fn get_rpath_flags(sess: &Session, out_filename: &Path) -> Vec<~str> {
    let os = sess.targ_cfg.os;

    // No rpath on windows
    if os == abi::OsWin32 {
        return Vec::new();
    }

    let mut flags = Vec::new();

    if sess.targ_cfg.os == abi::OsFreebsd {
        flags.push_all(["-Wl,-rpath,/usr/local/lib/gcc46".to_owned(),
                        "-Wl,-rpath,/usr/local/lib/gcc44".to_owned(),
                        "-Wl,-z,origin".to_owned()]);
    }

    debug!("preparing the RPATH!");

    let sysroot = sess.sysroot();
    let output = out_filename;
    let libs = sess.cstore.get_used_crates(cstore::RequireDynamic);
    let libs = libs.move_iter().filter_map(|(_, l)| {
        l.map(|p| p.clone())
    }).collect::<~[_]>();

    let rpaths = get_rpaths(os, sysroot, output, libs,
                            sess.opts.target_triple);
    flags.push_all(rpaths_to_flags(rpaths.as_slice()).as_slice());
    flags
}

pub fn rpaths_to_flags(rpaths: &[~str]) -> Vec<~str> {
    let mut ret = Vec::new();
    for rpath in rpaths.iter() {
        ret.push("-Wl,-rpath," + *rpath);
    }
    return ret;
}

fn get_rpaths(os: abi::Os,
              sysroot: &Path,
              output: &Path,
              libs: &[Path],
              target_triple: &str) -> Vec<~str> {
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

    // And a final backup rpath to the global library location.
    let fallback_rpaths = vec!(get_install_prefix_rpath(sysroot, target_triple));

    fn log_rpaths(desc: &str, rpaths: &[~str]) {
        debug!("{} rpaths:", desc);
        for rpath in rpaths.iter() {
            debug!("    {}", *rpath);
        }
    }

    log_rpaths("relative", rel_rpaths.as_slice());
    log_rpaths("fallback", fallback_rpaths.as_slice());

    let mut rpaths = rel_rpaths;
    rpaths.push_all(fallback_rpaths.as_slice());

    // Remove duplicates
    let rpaths = minimize_rpaths(rpaths.as_slice());
    return rpaths;
}

fn get_rpaths_relative_to_output(os: abi::Os,
                                 output: &Path,
                                 libs: &[Path]) -> Vec<~str> {
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

    let mut lib = fs::realpath(&os::make_absolute(lib)).unwrap();
    lib.pop();
    let mut output = fs::realpath(&os::make_absolute(output)).unwrap();
    output.pop();
    let relative = lib.path_relative_from(&output);
    let relative = relative.expect("could not create rpath relative to output");
    // FIXME (#9639): This needs to handle non-utf8 paths
    prefix+"/"+relative.as_str().expect("non-utf8 component in path")
}

pub fn get_install_prefix_rpath(sysroot: &Path, target_triple: &str) -> ~str {
    let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");

    let tlib = filesearch::relative_target_lib_path(sysroot, target_triple);
    let mut path = Path::new(install_prefix);
    path.push(&tlib);
    let path = os::make_absolute(&path);
    // FIXME (#9639): This needs to handle non-utf8 paths
    path.as_str().expect("non-utf8 component in rpath").to_owned()
}

pub fn minimize_rpaths(rpaths: &[~str]) -> Vec<~str> {
    let mut set = HashSet::new();
    let mut minimized = Vec::new();
    for rpath in rpaths.iter() {
        if set.insert(rpath.as_slice()) {
            minimized.push(rpath.clone());
        }
    }
    minimized
}

#[cfg(unix, test)]
mod test {
    use back::rpath::get_install_prefix_rpath;
    use back::rpath::{minimize_rpaths, rpaths_to_flags, get_rpath_relative_to_output};
    use syntax::abi;
    use metadata::filesearch;

    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags(["path1".to_owned(), "path2".to_owned()]);
        assert_eq!(flags, vec!("-Wl,-rpath,path1".to_owned(), "-Wl,-rpath,path2".to_owned()));
    }

    #[test]
    fn test_prefix_rpath() {
        let sysroot = filesearch::get_or_default_sysroot();
        let res = get_install_prefix_rpath(&sysroot, "triple");
        let mut d = Path::new((option_env!("CFG_PREFIX")).expect("CFG_PREFIX"));
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
        let sysroot = filesearch::get_or_default_sysroot();
        let res = get_install_prefix_rpath(&sysroot, "triple");
        assert!(Path::new(res).is_absolute());
    }

    #[test]
    fn test_minimize1() {
        let res = minimize_rpaths(["rpath1".to_owned(), "rpath2".to_owned(), "rpath1".to_owned()]);
        assert!(res.as_slice() == ["rpath1".to_owned(), "rpath2".to_owned()]);
    }

    #[test]
    fn test_minimize2() {
        let res = minimize_rpaths(["1a".to_owned(), "2".to_owned(),  "2".to_owned(),
                                   "1a".to_owned(), "4a".to_owned(), "1a".to_owned(),
                                   "2".to_owned(),  "3".to_owned(),  "4a".to_owned(),
                                   "3".to_owned()]);
        assert!(res.as_slice() == ["1a".to_owned(), "2".to_owned(), "4a".to_owned(),
                                   "3".to_owned()]);
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
}
