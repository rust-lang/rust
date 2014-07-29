// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::collections::HashSet;
use std::os;
use std::io::IoError;
use syntax::abi;
use syntax::ast;

pub struct RPathConfig<'a> {
    pub os: abi::Os,
    pub used_crates: Vec<(ast::CrateNum, Option<Path>)>,
    pub out_filename: Path,
    pub get_install_prefix_lib_path: ||:'a -> Path,
    pub realpath: |&Path|:'a -> Result<Path, IoError>
}

pub fn get_rpath_flags(config: RPathConfig) -> Vec<String> {

    // No rpath on windows
    if config.os == abi::OsWin32 {
        return Vec::new();
    }

    let mut flags = Vec::new();

    if config.os == abi::OsFreebsd {
        flags.push_all(["-Wl,-rpath,/usr/local/lib/gcc46".to_string(),
                        "-Wl,-rpath,/usr/local/lib/gcc44".to_string(),
                        "-Wl,-z,origin".to_string()]);
    }
    else if config.os == abi::OsDragonfly {
        flags.push_all(["-Wl,-rpath,/usr/lib/gcc47".to_string(),
                        "-Wl,-rpath,/usr/lib/gcc44".to_string(),
                        "-Wl,-z,origin".to_string()]);
    }


    debug!("preparing the RPATH!");

    let libs = config.used_crates.clone();
    let libs = libs.move_iter().filter_map(|(_, l)| {
        l.map(|p| p.clone())
    }).collect::<Vec<_>>();

    let rpaths = get_rpaths(config, libs.as_slice());
    flags.push_all(rpaths_to_flags(rpaths.as_slice()).as_slice());
    flags
}

fn rpaths_to_flags(rpaths: &[String]) -> Vec<String> {
    let mut ret = Vec::new();
    for rpath in rpaths.iter() {
        ret.push(format!("-Wl,-rpath,{}", (*rpath).as_slice()));
    }
    return ret;
}

fn get_rpaths(mut config: RPathConfig,
              libs: &[Path]) -> Vec<String> {
    debug!("output: {}", config.out_filename.display());
    debug!("libs:");
    for libpath in libs.iter() {
        debug!("    {}", libpath.display());
    }

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rel_rpaths = get_rpaths_relative_to_output(&mut config, libs);

    // And a final backup rpath to the global library location.
    let fallback_rpaths = vec!(get_install_prefix_rpath(config));

    fn log_rpaths(desc: &str, rpaths: &[String]) {
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

fn get_rpaths_relative_to_output(config: &mut RPathConfig,
                                 libs: &[Path]) -> Vec<String> {
    libs.iter().map(|a| get_rpath_relative_to_output(config, a)).collect()
}

fn get_rpath_relative_to_output(config: &mut RPathConfig,
                                lib: &Path) -> String {
    use std::os;

    assert!(config.os != abi::OsWin32);

    // Mac doesn't appear to support $ORIGIN
    let prefix = match config.os {
        abi::OsAndroid | abi::OsLinux | abi::OsFreebsd | abi::OsDragonfly
                          => "$ORIGIN",
        abi::OsMacos => "@loader_path",
        abi::OsWin32 | abi::OsiOS => unreachable!()
    };

    let mut lib = (config.realpath)(&os::make_absolute(lib)).unwrap();
    lib.pop();
    let mut output = (config.realpath)(&os::make_absolute(&config.out_filename)).unwrap();
    output.pop();
    let relative = lib.path_relative_from(&output);
    let relative = relative.expect("could not create rpath relative to output");
    // FIXME (#9639): This needs to handle non-utf8 paths
    format!("{}/{}",
            prefix,
            relative.as_str().expect("non-utf8 component in path"))
}

fn get_install_prefix_rpath(config: RPathConfig) -> String {
    let path = (config.get_install_prefix_lib_path)();
    let path = os::make_absolute(&path);
    // FIXME (#9639): This needs to handle non-utf8 paths
    path.as_str().expect("non-utf8 component in rpath").to_string()
}

fn minimize_rpaths(rpaths: &[String]) -> Vec<String> {
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
    use super::{RPathConfig};
    use super::{minimize_rpaths, rpaths_to_flags, get_rpath_relative_to_output};
    use syntax::abi;

    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags([
            "path1".to_string(),
            "path2".to_string()
        ]);
        assert_eq!(flags,
                   vec!("-Wl,-rpath,path1".to_string(),
                        "-Wl,-rpath,path2".to_string()));
    }

    #[test]
    fn test_minimize1() {
        let res = minimize_rpaths([
            "rpath1".to_string(),
            "rpath2".to_string(),
            "rpath1".to_string()
        ]);
        assert!(res.as_slice() == [
            "rpath1".to_string(),
            "rpath2".to_string()
        ]);
    }

    #[test]
    fn test_minimize2() {
        let res = minimize_rpaths([
            "1a".to_string(),
            "2".to_string(),
            "2".to_string(),
            "1a".to_string(),
            "4a".to_string(),
            "1a".to_string(),
            "2".to_string(),
            "3".to_string(),
            "4a".to_string(),
            "3".to_string()
        ]);
        assert!(res.as_slice() == [
            "1a".to_string(),
            "2".to_string(),
            "4a".to_string(),
            "3".to_string()
        ]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    fn test_rpath_relative() {
        let config = &mut RPathConfig {
            os: abi::OsLinux,
            used_crates: Vec::new(),
            out_filename: Path::new("bin/rustc"),
            get_install_prefix_lib_path: || fail!(),
            realpath: |p| Ok(p.clone())
        };
        let res = get_rpath_relative_to_output(config, &Path::new("lib/libstd.so"));
        assert_eq!(res.as_slice(), "$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "freebsd")]
    fn test_rpath_relative() {
        let config = &mut RPathConfig {
            os: abi::OsFreebsd,
            used_crates: Vec::new(),
            out_filename: Path::new("bin/rustc"),
            get_install_prefix_lib_path: || fail!(),
            realpath: |p| Ok(p.clone())
        };
        let res = get_rpath_relative_to_output(config, &Path::new("lib/libstd.so"));
        assert_eq!(res.as_slice(), "$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "dragonfly")]
    fn test_rpath_relative() {
        let config = &mut RPathConfig {
            os: abi::OsDragonfly,
            used_crates: Vec::new(),
            out_filename: Path::new("bin/rustc"),
            get_install_prefix_lib_path: || fail!(),
            realpath: |p| Ok(p.clone())
        };
        let res = get_rpath_relative_to_output(config, &Path::new("lib/libstd.so"));
        assert_eq!(res.as_slice(), "$ORIGIN/../lib");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_rpath_relative() {
        let config = &mut RPathConfig {
            os: abi::OsMacos,
            used_crates: Vec::new(),
            out_filename: Path::new("bin/rustc"),
            get_install_prefix_lib_path: || fail!(),
            realpath: |p| Ok(p.clone())
        };
        let res = get_rpath_relative_to_output(config, &Path::new("lib/libstd.so"));
        assert_eq!(res.as_slice(), "@loader_path/../lib");
    }
}
