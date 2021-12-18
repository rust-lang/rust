use pathdiff::diff_paths;
use rustc_data_structures::fx::FxHashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub struct RPathConfig<'a> {
    pub libs: &'a [&'a Path],
    pub out_filename: PathBuf,
    pub is_like_osx: bool,
    pub has_rpath: bool,
    pub linker_is_gnu: bool,
}

pub fn get_rpath_flags(config: &mut RPathConfig<'_>) -> Vec<String> {
    // No rpath on windows
    if !config.has_rpath {
        return Vec::new();
    }

    debug!("preparing the RPATH!");

    let rpaths = get_rpaths(config);
    let mut flags = rpaths_to_flags(&rpaths);

    if config.linker_is_gnu {
        // Use DT_RUNPATH instead of DT_RPATH if available
        flags.push("-Wl,--enable-new-dtags".to_owned());

        // Set DF_ORIGIN for substitute $ORIGIN
        flags.push("-Wl,-z,origin".to_owned());
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

fn get_rpaths(config: &mut RPathConfig<'_>) -> Vec<String> {
    debug!("output: {:?}", config.out_filename.display());
    debug!("libs:");
    for libpath in config.libs {
        debug!("    {:?}", libpath.display());
    }

    // Use relative paths to the libraries. Binaries can be moved
    // as long as they maintain the relative relationship to the
    // crates they depend on.
    let rpaths = get_rpaths_relative_to_output(config);

    debug!("rpaths:");
    for rpath in &rpaths {
        debug!("    {}", rpath);
    }

    // Remove duplicates
    minimize_rpaths(&rpaths)
}

fn get_rpaths_relative_to_output(config: &mut RPathConfig<'_>) -> Vec<String> {
    config.libs.iter().map(|a| get_rpath_relative_to_output(config, a)).collect()
}

fn get_rpath_relative_to_output(config: &mut RPathConfig<'_>, lib: &Path) -> String {
    // Mac doesn't appear to support $ORIGIN
    let prefix = if config.is_like_osx { "@loader_path" } else { "$ORIGIN" };

    let cwd = env::current_dir().unwrap();
    let mut lib = fs::canonicalize(&cwd.join(lib)).unwrap_or_else(|_| cwd.join(lib));
    lib.pop(); // strip filename
    let mut output = cwd.join(&config.out_filename);
    output.pop(); // strip filename
    let output = fs::canonicalize(&output).unwrap_or(output);
    let relative = path_relative_from(&lib, &output)
        .unwrap_or_else(|| panic!("couldn't create relative path from {:?} to {:?}", output, lib));
    // FIXME (#9639): This needs to handle non-utf8 paths
    format!("{}/{}", prefix, relative.to_str().expect("non-utf8 component in path"))
}

// This routine is adapted from the *old* Path's `path_relative_from`
// function, which works differently from the new `relative_from` function.
// In particular, this handles the case on unix where both paths are
// absolute but with only the root as the common directory.
fn path_relative_from(path: &Path, base: &Path) -> Option<PathBuf> {
    diff_paths(path, base)
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
