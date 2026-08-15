use std::ffi::OsString;
use std::path::{Path, PathBuf};

use pathdiff::diff_paths;
use rustc_data_structures::fx::FxHashSet;
use rustc_fs_util::try_canonicalize;
use tracing::debug;

pub(super) struct RPathConfig<'a> {
    pub libs: &'a [&'a Path],
    pub out_filename: PathBuf,
    pub is_like_darwin: bool,
    pub linker_is_gnu: bool,
}

pub(super) fn get_rpath_linker_args(config: &RPathConfig<'_>) -> Vec<OsString> {
    debug!("preparing the RPATH!");

    let rpaths = get_rpaths(config);
    let mut args = Vec::with_capacity(rpaths.len() * 2); // the minimum needed capacity

    for rpath in rpaths {
        args.push("-rpath".into());
        args.push(rpath);
    }

    if config.linker_is_gnu {
        // Use DT_RUNPATH instead of DT_RPATH if available
        args.push("--enable-new-dtags".into());

        // Set DF_ORIGIN for substitute $ORIGIN
        args.push("-z".into());
        args.push("origin".into());
    }

    args
}

fn get_rpaths(config: &RPathConfig<'_>) -> Vec<OsString> {
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
        debug!("    {:?}", rpath);
    }

    // Remove duplicates
    minimize_rpaths(&rpaths)
}

fn get_rpaths_relative_to_output(config: &RPathConfig<'_>) -> Vec<OsString> {
    config.libs.iter().map(|a| get_rpath_relative_to_output(config, a)).collect()
}

fn get_rpath_relative_to_output(config: &RPathConfig<'_>, lib: &Path) -> OsString {
    // Mac doesn't appear to support $ORIGIN
    let prefix = if config.is_like_darwin { "@loader_path" } else { "$ORIGIN" };

    // Strip filenames
    let lib = lib.parent().unwrap();
    let output = config.out_filename.parent().unwrap();

    // If output or lib is empty, just assume it locates in current path
    let lib = if lib == Path::new("") { Path::new(".") } else { lib };
    let output = if output == Path::new("") { Path::new(".") } else { output };

    let lib = try_canonicalize(lib).unwrap();
    let output = try_canonicalize(output).unwrap();
    let relative = path_relative_from(&lib, &output)
        .unwrap_or_else(|| panic!("couldn't create relative path from {output:?} to {lib:?}"));

    let mut rpath = OsString::from(prefix);
    rpath.push("/");
    rpath.push(relative);
    rpath
}

// This routine is adapted from the *old* Path's `path_relative_from`
// function, which works differently from the new `relative_from` function.
// In particular, this handles the case on unix where both paths are
// absolute but with only the root as the common directory.
fn path_relative_from(path: &Path, base: &Path) -> Option<PathBuf> {
    diff_paths(path, base)
}

fn minimize_rpaths(rpaths: &[OsString]) -> Vec<OsString> {
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
