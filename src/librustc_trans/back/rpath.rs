// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::fs;

use rustc::hir::def_id::CrateNum;
use rustc::middle::cstore::LibSource;

pub struct RPathConfig<'a> {
    pub used_crates: Vec<(CrateNum, LibSource)>,
    pub out_filename: PathBuf,
    pub is_like_osx: bool,
    pub has_rpath: bool,
    pub linker_is_gnu: bool,
    pub get_install_prefix_lib_path: &'a mut FnMut() -> PathBuf,
}

pub fn get_rpath_flags(config: &mut RPathConfig) -> Vec<String> {
    // No rpath on windows
    if !config.has_rpath {
        return Vec::new();
    }

    let mut flags = Vec::new();

    debug!("preparing the RPATH!");

    let libs = config.used_crates.clone();
    let libs = libs.into_iter().filter_map(|(_, l)| l.option()).collect::<Vec<_>>();
    let rpaths = get_rpaths(config, &libs[..]);
    flags.extend_from_slice(&rpaths_to_flags(&rpaths[..]));

    // Use DT_RUNPATH instead of DT_RPATH if available
    if config.linker_is_gnu {
        flags.push("-Wl,--enable-new-dtags".to_string());
    }

    flags
}

fn rpaths_to_flags(rpaths: &[String]) -> Vec<String> {
    let mut ret = Vec::new();
    for rpath in rpaths {
        if rpath.contains(',') {
            ret.push("-Wl,-rpath".into());
            ret.push("-Xlinker".into());
            ret.push(rpath.clone());
        } else {
            ret.push(format!("-Wl,-rpath,{}", &(*rpath)));
        }
    }
    return ret;
}

fn get_rpaths(config: &mut RPathConfig, libs: &[PathBuf]) -> Vec<String> {
    debug!("output: {:?}", config.out_filename.display());
    debug!("libs:");
    for libpath in libs {
        debug!("    {:?}", libpath.display());
    }

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rel_rpaths = get_rpaths_relative_to_output(config, libs);

    // And a final backup rpath to the global library location.
    let fallback_rpaths = vec![get_install_prefix_rpath(config)];

    fn log_rpaths(desc: &str, rpaths: &[String]) {
        debug!("{} rpaths:", desc);
        for rpath in rpaths {
            debug!("    {}", *rpath);
        }
    }

    log_rpaths("relative", &rel_rpaths[..]);
    log_rpaths("fallback", &fallback_rpaths[..]);

    let mut rpaths = rel_rpaths;
    rpaths.extend_from_slice(&fallback_rpaths[..]);

    // Remove duplicates
    let rpaths = minimize_rpaths(&rpaths[..]);
    return rpaths;
}

fn get_rpaths_relative_to_output(config: &mut RPathConfig,
                                 libs: &[PathBuf]) -> Vec<String> {
    libs.iter().map(|a| get_rpath_relative_to_output(config, a)).collect()
}

fn get_rpath_relative_to_output(config: &mut RPathConfig, lib: &Path) -> String {
    // Mac doesn't appear to support $ORIGIN
    let prefix = if config.is_like_osx {
        "@loader_path"
    } else {
        "$ORIGIN"
    };

    let cwd = env::current_dir().unwrap();
    let mut lib = fs::canonicalize(&cwd.join(lib)).unwrap_or(cwd.join(lib));
    lib.pop();
    let mut output = cwd.join(&config.out_filename);
    output.pop();
    let output = fs::canonicalize(&output).unwrap_or(output);
    let relative = path_relative_from(&lib, &output)
        .expect(&format!("couldn't create relative path from {:?} to {:?}", output, lib));
    // FIXME (#9639): This needs to handle non-utf8 paths
    format!("{}/{}", prefix,
            relative.to_str().expect("non-utf8 component in path"))
}

// This routine is adapted from the *old* Path's `path_relative_from`
// function, which works differently from the new `relative_from` function.
// In particular, this handles the case on unix where both paths are
// absolute but with only the root as the common directory.
fn path_relative_from(path: &Path, base: &Path) -> Option<PathBuf> {
    use std::path::Component;

    if path.is_absolute() != base.is_absolute() {
        if path.is_absolute() {
            Some(PathBuf::from(path))
        } else {
            None
        }
    } else {
        let mut ita = path.components();
        let mut itb = base.components();
        let mut comps: Vec<Component> = vec![];
        loop {
            match (ita.next(), itb.next()) {
                (None, None) => break,
                (Some(a), None) => {
                    comps.push(a);
                    comps.extend(ita.by_ref());
                    break;
                }
                (None, _) => comps.push(Component::ParentDir),
                (Some(a), Some(b)) if comps.is_empty() && a == b => (),
                (Some(a), Some(b)) if b == Component::CurDir => comps.push(a),
                (Some(_), Some(b)) if b == Component::ParentDir => return None,
                (Some(a), Some(_)) => {
                    comps.push(Component::ParentDir);
                    for _ in itb {
                        comps.push(Component::ParentDir);
                    }
                    comps.push(a);
                    comps.extend(ita.by_ref());
                    break;
                }
            }
        }
        Some(comps.iter().map(|c| c.as_os_str()).collect())
    }
}


fn get_install_prefix_rpath(config: &mut RPathConfig) -> String {
    let path = (config.get_install_prefix_lib_path)();
    let path = env::current_dir().unwrap().join(&path);
    // FIXME (#9639): This needs to handle non-utf8 paths
    path.to_str().expect("non-utf8 component in rpath").to_string()
}

fn minimize_rpaths(rpaths: &[String]) -> Vec<String> {
    let mut set = HashSet::new();
    let mut minimized = Vec::new();
    for rpath in rpaths {
        if set.insert(&rpath[..]) {
            minimized.push(rpath.clone());
        }
    }
    minimized
}

#[cfg(all(unix, test))]
mod tests {
    use super::{RPathConfig};
    use super::{minimize_rpaths, rpaths_to_flags, get_rpath_relative_to_output};
    use std::path::{Path, PathBuf};

    #[test]
    fn test_rpaths_to_flags() {
        let flags = rpaths_to_flags(&[
            "path1".to_string(),
            "path2".to_string()
        ]);
        assert_eq!(flags,
                   ["-Wl,-rpath,path1",
                    "-Wl,-rpath,path2"]);
    }

    #[test]
    fn test_minimize1() {
        let res = minimize_rpaths(&[
            "rpath1".to_string(),
            "rpath2".to_string(),
            "rpath1".to_string()
        ]);
        assert!(res == [
            "rpath1",
            "rpath2",
        ]);
    }

    #[test]
    fn test_minimize2() {
        let res = minimize_rpaths(&[
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
        assert!(res == [
            "1a",
            "2",
            "4a",
            "3",
        ]);
    }

    #[test]
    fn test_rpath_relative() {
        if cfg!(target_os = "macos") {
            let config = &mut RPathConfig {
                used_crates: Vec::new(),
                has_rpath: true,
                is_like_osx: true,
                linker_is_gnu: false,
                out_filename: PathBuf::from("bin/rustc"),
                get_install_prefix_lib_path: &mut || panic!(),
            };
            let res = get_rpath_relative_to_output(config,
                                                   Path::new("lib/libstd.so"));
            assert_eq!(res, "@loader_path/../lib");
        } else {
            let config = &mut RPathConfig {
                used_crates: Vec::new(),
                out_filename: PathBuf::from("bin/rustc"),
                get_install_prefix_lib_path: &mut || panic!(),
                has_rpath: true,
                is_like_osx: false,
                linker_is_gnu: true,
            };
            let res = get_rpath_relative_to_output(config,
                                                   Path::new("lib/libstd.so"));
            assert_eq!(res, "$ORIGIN/../lib");
        }
    }

    #[test]
    fn test_xlinker() {
        let args = rpaths_to_flags(&[
            "a/normal/path".to_string(),
            "a,comma,path".to_string()
        ]);

        assert_eq!(args, vec![
            "-Wl,-rpath,a/normal/path".to_string(),
            "-Wl,-rpath".to_string(),
            "-Xlinker".to_string(),
            "a,comma,path".to_string()
        ]);
    }
}
