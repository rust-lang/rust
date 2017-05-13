//! A build dependency for running `cmake` to build a native library
//!
//! This crate provides some necessary boilerplate and shim support for running
//! the system `cmake` command to build a native library. It will add
//! appropriate cflags for building code to link into Rust, handle cross
//! compilation, and use the necessary generator for the platform being
//! targeted.
//!
//! The builder-style configuration allows for various variables and such to be
//! passed down into the build as well.
//!
//! ## Installation
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [build-dependencies]
//! cmake = "0.1"
//! ```
//!
//! ## Examples
//!
//! ```no_run
//! use cmake;
//!
//! // Builds the project in the directory located in `libfoo`, installing it
//! // into $OUT_DIR
//! let dst = cmake::build("libfoo");
//!
//! println!("cargo:rustc-link-search=native={}", dst.display());
//! println!("cargo:rustc-link-lib=static=foo");
//! ```
//!
//! ```no_run
//! use cmake::Config;
//!
//! let dst = Config::new("libfoo")
//!                  .define("FOO", "BAR")
//!                  .cflag("-foo")
//!                  .build();
//! println!("cargo:rustc-link-search=native={}", dst.display());
//! println!("cargo:rustc-link-lib=static=foo");
//! ```

#![deny(missing_docs)]

extern crate gcc;

use std::env;
use std::ffi::{OsString, OsStr};
use std::fs::{self, File};
use std::io::ErrorKind;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(windows)]
mod registry;

/// Builder style configuration for a pending CMake build.
pub struct Config {
    path: PathBuf,
    generator: Option<OsString>,
    cflags: OsString,
    cxxflags: OsString,
    defines: Vec<(OsString, OsString)>,
    deps: Vec<String>,
    target: Option<String>,
    host: Option<String>,
    out_dir: Option<PathBuf>,
    profile: Option<String>,
    build_args: Vec<OsString>,
    cmake_target: Option<String>,
}

/// Builds the native library rooted at `path` with the default cmake options.
/// This will return the directory in which the library was installed.
///
/// # Examples
///
/// ```no_run
/// use cmake;
///
/// // Builds the project in the directory located in `libfoo`, installing it
/// // into $OUT_DIR
/// let dst = cmake::build("libfoo");
///
/// println!("cargo:rustc-link-search=native={}", dst.display());
/// println!("cargo:rustc-link-lib=static=foo");
/// ```
///
pub fn build<P: AsRef<Path>>(path: P) -> PathBuf {
    Config::new(path.as_ref()).build()
}

impl Config {
    /// Creates a new blank set of configuration to build the project specified
    /// at the path `path`.
    pub fn new<P: AsRef<Path>>(path: P) -> Config {
        Config {
            path: env::current_dir().unwrap().join(path),
            generator: None,
            cflags: OsString::new(),
            cxxflags: OsString::new(),
            defines: Vec::new(),
            deps: Vec::new(),
            profile: None,
            out_dir: None,
            target: None,
            host: None,
            build_args: Vec::new(),
            cmake_target: None,
        }
    }

    /// Sets the build-tool generator (`-G`) for this compilation.
    pub fn generator<T: AsRef<OsStr>>(&mut self, generator: T) -> &mut Config {
        self.generator = Some(generator.as_ref().to_owned());
        self
    }

    /// Adds a custom flag to pass down to the C compiler, supplementing those
    /// that this library already passes.
    pub fn cflag<P: AsRef<OsStr>>(&mut self, flag: P) -> &mut Config {
        self.cflags.push(" ");
        self.cflags.push(flag.as_ref());
        self
    }

    /// Adds a custom flag to pass down to the C++ compiler, supplementing those
    /// that this library already passes.
    pub fn cxxflag<P: AsRef<OsStr>>(&mut self, flag: P) -> &mut Config {
        self.cxxflags.push(" ");
        self.cxxflags.push(flag.as_ref());
        self
    }

    /// Adds a new `-D` flag to pass to cmake during the generation step.
    pub fn define<K, V>(&mut self, k: K, v: V) -> &mut Config
        where K: AsRef<OsStr>, V: AsRef<OsStr>
    {
        self.defines.push((k.as_ref().to_owned(), v.as_ref().to_owned()));
        self
    }

    /// Registers a dependency for this compilation on the native library built
    /// by Cargo previously.
    ///
    /// This registration will modify the `CMAKE_PREFIX_PATH` environment
    /// variable for the build system generation step.
    pub fn register_dep(&mut self, dep: &str) -> &mut Config {
        self.deps.push(dep.to_string());
        self
    }

    /// Sets the target triple for this compilation.
    ///
    /// This is automatically scraped from `$TARGET` which is set for Cargo
    /// build scripts so it's not necessary to call this from a build script.
    pub fn target(&mut self, target: &str) -> &mut Config {
        self.target = Some(target.to_string());
        self
    }

    /// Sets the host triple for this compilation.
    ///
    /// This is automatically scraped from `$HOST` which is set for Cargo
    /// build scripts so it's not necessary to call this from a build script.
    pub fn host(&mut self, host: &str) -> &mut Config {
        self.host = Some(host.to_string());
        self
    }

    /// Sets the output directory for this compilation.
    ///
    /// This is automatically scraped from `$OUT_DIR` which is set for Cargo
    /// build scripts so it's not necessary to call this from a build script.
    pub fn out_dir<P: AsRef<Path>>(&mut self, out: P) -> &mut Config {
        self.out_dir = Some(out.as_ref().to_path_buf());
        self
    }

    /// Sets the profile for this compilation.
    ///
    /// This is automatically scraped from `$PROFILE` which is set for Cargo
    /// build scripts so it's not necessary to call this from a build script.
    pub fn profile(&mut self, profile: &str) -> &mut Config {
        self.profile = Some(profile.to_string());
        self
    }

    /// Add an argument to the final `cmake` build step
    pub fn build_arg<A: AsRef<OsStr>>(&mut self, arg: A) -> &mut Config {
        self.build_args.push(arg.as_ref().to_owned());
        self
    }

    /// Sets the build target for the final `cmake` build step, this will
    /// default to "install" if not specified.
    pub fn build_target(&mut self, target: &str) -> &mut Config {
        self.cmake_target = Some(target.to_string());
        self
    }

    /// Run this configuration, compiling the library with all the configured
    /// options.
    ///
    /// This will run both the build system generator command as well as the
    /// command to build the library.
    pub fn build(&mut self) -> PathBuf {
        let target = self.target.clone().unwrap_or_else(|| {
            getenv_unwrap("TARGET")
        });
        let host = self.host.clone().unwrap_or_else(|| {
            getenv_unwrap("HOST")
        });
        let msvc = target.contains("msvc");
        let c_compiler = gcc::Config::new().cargo_metadata(false)
                                         .opt_level(0)
                                         .debug(false)
                                         .target(&target)
                                         .host(&host)
                                         .get_compiler();
        let cxx_compiler = gcc::Config::new().cargo_metadata(false)
                                         .cpp(true)
                                         .opt_level(0)
                                         .debug(false)
                                         .target(&target)
                                         .host(&host)
                                         .get_compiler();

        let dst = self.out_dir.clone().unwrap_or_else(|| {
            PathBuf::from(getenv_unwrap("OUT_DIR"))
        });
        let build = dst.join("build");
        self.maybe_clear(&build);
        let _ = fs::create_dir(&build);

        // Add all our dependencies to our cmake paths
        let mut cmake_prefix_path = Vec::new();
        for dep in &self.deps {
            if let Some(root) = env::var_os(&format!("DEP_{}_ROOT", dep)) {
                cmake_prefix_path.push(PathBuf::from(root));
            }
        }
        let system_prefix = env::var_os("CMAKE_PREFIX_PATH")
                                .unwrap_or(OsString::new());
        cmake_prefix_path.extend(env::split_paths(&system_prefix)
                                     .map(|s| s.to_owned()));
        let cmake_prefix_path = env::join_paths(&cmake_prefix_path).unwrap();

        // Build up the first cmake command to build the build system.
        let mut cmd = Command::new("cmake");
        cmd.arg(&self.path)
           .current_dir(&build);
        if target.contains("windows-gnu") {
            if host.contains("windows") {
                // On MinGW we need to coerce cmake to not generate a visual
                // studio build system but instead use makefiles that MinGW can
                // use to build.
                if self.generator.is_none() {
                    cmd.arg("-G").arg("MSYS Makefiles");
                }
            } else {
                // If we're cross compiling onto windows, then set some
                // variables which will hopefully get things to succeed. Some
                // systems may need the `windres` or `dlltool` variables set, so
                // set them if possible.
                if !self.defined("CMAKE_SYSTEM_NAME") {
                    cmd.arg("-DCMAKE_SYSTEM_NAME=Windows");
                }
                if !self.defined("CMAKE_RC_COMPILER") {
                    let exe = find_exe(c_compiler.path());
                    if let Some(name) = exe.file_name().unwrap().to_str() {
                        let name = name.replace("gcc", "windres");
                        let windres = exe.with_file_name(name);
                        if windres.is_file() {
                            let mut arg = OsString::from("-DCMAKE_RC_COMPILER=");
                            arg.push(&windres);
                            cmd.arg(arg);
                        }
                    }
                }
            }
        } else if msvc {
            // If we're on MSVC we need to be sure to use the right generator or
            // otherwise we won't get 32/64 bit correct automatically.
            if self.generator.is_none() {
                cmd.arg("-G").arg(self.visual_studio_generator(&target));
            }
        }
        if let Some(ref generator) = self.generator {
            cmd.arg("-G").arg(generator);
        }
        let profile = self.profile.clone().unwrap_or_else(|| {
            match &getenv_unwrap("PROFILE")[..] {
                "bench" | "release" => "Release",
                // currently we need to always use the same CRT for MSVC
                _ if msvc => "Release",
                _ => "Debug",
            }.to_string()
        });
        for &(ref k, ref v) in &self.defines {
            let mut os = OsString::from("-D");
            os.push(k);
            os.push("=");
            os.push(v);
            cmd.arg(os);
        }

        if !self.defined("CMAKE_INSTALL_PREFIX") {
            let mut dstflag = OsString::from("-DCMAKE_INSTALL_PREFIX=");
            dstflag.push(&dst);
            cmd.arg(dstflag);
        }

        {
            let mut set_compiler = |kind: &str,
                                    compiler: &gcc::Tool,
                                    extra: &OsString| {
                let flag_var = format!("CMAKE_{}_FLAGS", kind);
                let tool_var = format!("CMAKE_{}_COMPILER", kind);
                if !self.defined(&flag_var) {
                    let mut flagsflag = OsString::from("-D");
                    flagsflag.push(&flag_var);
                    flagsflag.push("=");
                    flagsflag.push(extra);
                    for arg in compiler.args() {
                        flagsflag.push(" ");
                        flagsflag.push(arg);
                    }
                    cmd.arg(flagsflag);
                }

                // Apparently cmake likes to have an absolute path to the
                // compiler as otherwise it sometimes thinks that this variable
                // changed as it thinks the found compiler, /usr/bin/cc,
                // differs from the specified compiler, cc. Not entirely sure
                // what's up, but at least this means cmake doesn't get
                // confused?
                //
                // Also don't specify this on Windows as it's not needed for
                // MSVC and for MinGW it doesn't really vary.
                if !self.defined("CMAKE_TOOLCHAIN_FILE")
                   && !self.defined(&tool_var)
                   && env::consts::FAMILY != "windows" {
                    let mut ccompiler = OsString::from("-D");
                    ccompiler.push(&tool_var);
                    ccompiler.push("=");
                    ccompiler.push(find_exe(compiler.path()));
                    cmd.arg(ccompiler);
                }
            };

            set_compiler("C", &c_compiler, &self.cflags);
            set_compiler("CXX", &cxx_compiler, &self.cxxflags);
        }

        if !self.defined("CMAKE_BUILD_TYPE") {
            cmd.arg(&format!("-DCMAKE_BUILD_TYPE={}", profile));
        }

        if !self.defined("CMAKE_TOOLCHAIN_FILE") {
            if let Ok(s) = env::var("CMAKE_TOOLCHAIN_FILE") {
                cmd.arg(&format!("-DCMAKE_TOOLCHAIN_FILE={}", s));
            }
        }

        run(cmd.env("CMAKE_PREFIX_PATH", cmake_prefix_path), "cmake");

        let mut parallel_args = Vec::new();
        if fs::metadata(&dst.join("build/Makefile")).is_ok() {
            if let Ok(s) = env::var("NUM_JOBS") {
                parallel_args.push(format!("-j{}", s));
            }
        }

        // And build!
        let target = self.cmake_target.clone().unwrap_or("install".to_string());
        run(Command::new("cmake")
                    .arg("--build").arg(".")
                    .arg("--target").arg(target)
                    .arg("--config").arg(profile)
                    .arg("--").args(&self.build_args)
                    .args(&parallel_args)
                    .current_dir(&build), "cmake");

        println!("cargo:root={}", dst.display());
        return dst
    }

    fn visual_studio_generator(&self, target: &str) -> String {
        let base = match std::env::var("VisualStudioVersion") {
            Ok(version) => {
                match &version[..] {
                    "15.0" => "Visual Studio 15",
                    "14.0" => "Visual Studio 14 2015",
                    "12.0" => "Visual Studio 12 2013",
                    vers => panic!("\n\n\
                        unsupported or unknown VisualStudio version: {}\n\
                        if another version is installed consider running \
                        the appropriate vcvars script before building this \
                        crate\n\
                    ", vers),
                }
            }
            _ => {
                // Check for the presense of a specific registry key
                // that indicates visual studio is installed.
                if self.has_msbuild_version("15.0") {
                    "Visual Studio 15"
                } else if self.has_msbuild_version("14.0") {
                    "Visual Studio 14 2015"
                } else if self.has_msbuild_version("12.0") {
                    "Visual Studio 12 2013"
                } else {
                    panic!("\n\n\
                        couldn't determine visual studio generator\n\
                        if VisualStudio is installed, however, consider \
                        running the appropriate vcvars script before building \
                        this crate\n\
                    ");
                }
            }
        };

        if target.contains("i686") {
            base.to_string()
        } else if target.contains("x86_64") {
            format!("{} Win64", base)
        } else {
            panic!("unsupported msvc target: {}", target);
        }
    }

    #[cfg(not(windows))]
    fn has_msbuild_version(&self, _version: &str) -> bool {
        false
    }

    #[cfg(windows)]
    fn has_msbuild_version(&self, version: &str) -> bool {
        let key = format!("SOFTWARE\\Microsoft\\MSBuild\\ToolsVersions\\{}",
                          version);
        registry::LOCAL_MACHINE.open(key.as_ref()).is_ok()
    }

    fn defined(&self, var: &str) -> bool {
        self.defines.iter().any(|&(ref a, _)| a == var)
    }

    // If a cmake project has previously been built (e.g. CMakeCache.txt already
    // exists), then cmake will choke if the source directory for the original
    // project being built has changed. Detect this situation through the
    // `CMAKE_HOME_DIRECTORY` variable that cmake emits and if it doesn't match
    // we blow away the build directory and start from scratch (the recommended
    // solution apparently [1]).
    //
    // [1]: https://cmake.org/pipermail/cmake/2012-August/051545.html
    fn maybe_clear(&self, dir: &Path) {
        // CMake will apparently store canonicalized paths which normally
        // isn't relevant to us but we canonicalize it here to ensure
        // we're both checking the same thing.
        let path = fs::canonicalize(&self.path).unwrap_or(self.path.clone());
        let src = match path.to_str() {
            Some(src) => src,
            None => return,
        };
        let mut f = match File::open(dir.join("CMakeCache.txt")) {
            Ok(f) => f,
            Err(..) => return,
        };
        let mut u8contents = Vec::new();
        match f.read_to_end(&mut u8contents) {
            Ok(f) => f,
            Err(..) => return,
        };
        let contents = String::from_utf8_lossy(&u8contents);
        drop(f);
        for line in contents.lines() {
            if line.contains("CMAKE_HOME_DIRECTORY") && !line.contains(src) {
                println!("detected home dir change, cleaning out entire build \
                          directory");
                fs::remove_dir_all(dir).unwrap();
                break
            }
        }
    }
}

fn run(cmd: &mut Command, program: &str) {
    println!("running: {:?}", cmd);
    let status = match cmd.status() {
        Ok(status) => status,
        Err(ref e) if e.kind() == ErrorKind::NotFound => {
            fail(&format!("failed to execute command: {}\nis `{}` not installed?",
                          e, program));
        }
        Err(e) => fail(&format!("failed to execute command: {}", e)),
    };
    if !status.success() {
        fail(&format!("command did not execute successfully, got: {}", status));
    }
}

fn find_exe(path: &Path) -> PathBuf {
    env::split_paths(&env::var_os("PATH").unwrap_or(OsString::new()))
        .map(|p| p.join(path))
        .find(|p| fs::metadata(p).is_ok())
        .unwrap_or(path.to_owned())
}

fn getenv_unwrap(v: &str) -> String {
    match env::var(v) {
        Ok(s) => s,
        Err(..) => fail(&format!("environment variable `{}` not defined", v)),
    }
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
