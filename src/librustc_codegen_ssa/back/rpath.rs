use rustc_data_structures::fx::FxHashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::fs;

use rustc::hir::def_id::CrateNum;
use rustc::middle::cstore::LibSource;

pub struct RPathConfig<'a> {
    pub used_crates: &'a [(CrateNum, LibSource)],
    pub out_filename: PathBuf,
    pub is_like_osx: bool,
    pub has_rpath: bool,
    pub linker_is_gnu: bool,
    pub get_install_prefix_lib_path: &'a mut dyn FnMut() -> PathBuf,
}

pub fn get_rpath_flags(config: &mut RPathConfig<'_>) -> Vec<String> {
    // No rpath on windows
    if !config.has_rpath {
        return Vec::new();
    }

    debug!("preparing the RPATH!");

    let libs = config.used_crates.clone();
    let libs = libs.iter().filter_map(|&(_, ref l)| l.option()).collect::<Vec<_>>();
    let rpaths = get_rpaths(config, &libs);
    let mut flags = rpaths_to_flags(&rpaths);

    // Use DT_RUNPATH instead of DT_RPATH if available
    if config.linker_is_gnu {
        flags.push("-Wl,--enable-new-dtags".to_owned());
    }

    flags
}

fn rpaths_to_flags(rpaths: &[String]) -> Vec<String> {
    let mut ret = Vec::with_capacity(rpaths.len()); // the minimum needed capacity

    for rpath in rpaths {
        if rpath.contains(',') {
            ret.push("-Wl,-rpath".into());
            ret.push("-Xlinker".into());
            ret.push(rpath.clone());
        } else {
            ret.push(format!("-Wl,-rpath,{}", &(*rpath)));
        }
    }

    ret
}

fn get_rpaths(config: &mut RPathConfig<'_>, libs: &[PathBuf]) -> Vec<String> {
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

    log_rpaths("relative", &rel_rpaths);
    log_rpaths("fallback", &fallback_rpaths);

    let mut rpaths = rel_rpaths;
    rpaths.extend_from_slice(&fallback_rpaths);

    // Remove duplicates
    let rpaths = minimize_rpaths(&rpaths);

    rpaths
}

fn get_rpaths_relative_to_output(config: &mut RPathConfig<'_>,
                                 libs: &[PathBuf]) -> Vec<String> {
    libs.iter().map(|a| get_rpath_relative_to_output(config, a)).collect()
}

fn get_rpath_relative_to_output(config: &mut RPathConfig<'_>, lib: &Path) -> String {
    // Mac doesn't appear to support $ORIGIN
    let prefix = if config.is_like_osx {
        "@loader_path"
    } else {
        "$ORIGIN"
    };

    let cwd = env::current_dir().unwrap();
    let mut lib = fs::canonicalize(&cwd.join(lib)).unwrap_or_else(|_| cwd.join(lib));
    lib.pop(); // strip filename
    let mut output = cwd.join(&config.out_filename);
    output.pop(); // strip filename
    let output = fs::canonicalize(&output).unwrap_or(output);
    let relative = path_relative_from(&lib, &output).unwrap_or_else(||
        panic!("couldn't create relative path from {:?} to {:?}", output, lib));
    // FIXME (#9639): This needs to handle non-utf8 paths
    format!("{}/{}", prefix, relative.to_str().expect("non-utf8 component in path"))
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
        let mut comps: Vec<Component<'_>> = vec![];
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
                    comps.extend(itb.map(|_| Component::ParentDir));
                    comps.push(a);
                    comps.extend(ita.by_ref());
                    break;
                }
            }
        }
        Some(comps.iter().map(|c| c.as_os_str()).collect())
    }
}


fn get_install_prefix_rpath(config: &mut RPathConfig<'_>) -> String {
    let path = (config.get_install_prefix_lib_path)();
    let path = env::current_dir().unwrap().join(&path);
    // FIXME (#9639): This needs to handle non-utf8 paths
    path.to_str().expect("non-utf8 component in rpath").to_owned()
}

fn minimize_rpaths(rpaths: &[String]) -> Vec<String> {
    let mut set = FxHashSet::default();
    let mut minimized = Vec::new();
    for rpath in rpaths {
        if set.insert(rpath) {
            minimized.push(rpath.clone());
        }
    }
    minimized
}

#[cfg(all(unix, test))]
mod tests;
