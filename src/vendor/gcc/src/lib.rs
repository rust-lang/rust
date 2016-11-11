//! A library for build scripts to compile custom C code
//!
//! This library is intended to be used as a `build-dependencies` entry in
//! `Cargo.toml`:
//!
//! ```toml
//! [build-dependencies]
//! gcc = "0.3"
//! ```
//!
//! The purpose of this crate is to provide the utility functions necessary to
//! compile C code into a static archive which is then linked into a Rust crate.
//! The top-level `compile_library` function serves as a convenience and more
//! advanced configuration is available through the `Config` builder.
//!
//! This crate will automatically detect situations such as cross compilation or
//! other environment variables set by Cargo and will build code appropriately.
//!
//! # Examples
//!
//! Use the default configuration:
//!
//! ```no_run
//! extern crate gcc;
//!
//! fn main() {
//!     gcc::compile_library("libfoo.a", &["src/foo.c"]);
//! }
//! ```
//!
//! Use more advanced configuration:
//!
//! ```no_run
//! extern crate gcc;
//!
//! fn main() {
//!     gcc::Config::new()
//!                 .file("src/foo.c")
//!                 .define("FOO", Some("bar"))
//!                 .include("src")
//!                 .compile("libfoo.a");
//! }
//! ```

#![doc(html_root_url = "http://alexcrichton.com/gcc-rs")]
#![cfg_attr(test, deny(warnings))]
#![deny(missing_docs)]

#[cfg(feature = "parallel")]
extern crate rayon;

use std::env;
use std::ffi::{OsString, OsStr};
use std::fs;
use std::io;
use std::path::{PathBuf, Path};
use std::process::{Command, Stdio};
use std::io::{BufReader, BufRead, Write};

#[cfg(windows)]
mod registry;
pub mod windows_registry;

/// Extra configuration to pass to gcc.
pub struct Config {
    include_directories: Vec<PathBuf>,
    definitions: Vec<(String, Option<String>)>,
    objects: Vec<PathBuf>,
    flags: Vec<String>,
    files: Vec<PathBuf>,
    cpp: bool,
    cpp_link_stdlib: Option<Option<String>>,
    cpp_set_stdlib: Option<String>,
    target: Option<String>,
    host: Option<String>,
    out_dir: Option<PathBuf>,
    opt_level: Option<String>,
    debug: Option<bool>,
    env: Vec<(OsString, OsString)>,
    compiler: Option<PathBuf>,
    archiver: Option<PathBuf>,
    cargo_metadata: bool,
    pic: Option<bool>,
}

/// Configuration used to represent an invocation of a C compiler.
///
/// This can be used to figure out what compiler is in use, what the arguments
/// to it are, and what the environment variables look like for the compiler.
/// This can be used to further configure other build systems (e.g. forward
/// along CC and/or CFLAGS) or the `to_command` method can be used to run the
/// compiler itself.
pub struct Tool {
    path: PathBuf,
    args: Vec<OsString>,
    env: Vec<(OsString, OsString)>,
}

/// Compile a library from the given set of input C files.
///
/// This will simply compile all files into object files and then assemble them
/// into the output. This will read the standard environment variables to detect
/// cross compilations and such.
///
/// This function will also print all metadata on standard output for Cargo.
///
/// # Example
///
/// ```no_run
/// gcc::compile_library("libfoo.a", &["foo.c", "bar.c"]);
/// ```
pub fn compile_library(output: &str, files: &[&str]) {
    let mut c = Config::new();
    for f in files.iter() {
        c.file(*f);
    }
    c.compile(output)
}

impl Config {
    /// Construct a new instance of a blank set of configuration.
    ///
    /// This builder is finished with the `compile` function.
    pub fn new() -> Config {
        Config {
            include_directories: Vec::new(),
            definitions: Vec::new(),
            objects: Vec::new(),
            flags: Vec::new(),
            files: Vec::new(),
            cpp: false,
            cpp_link_stdlib: None,
            cpp_set_stdlib: None,
            target: None,
            host: None,
            out_dir: None,
            opt_level: None,
            debug: None,
            env: Vec::new(),
            compiler: None,
            archiver: None,
            cargo_metadata: true,
            pic: None,
        }
    }

    /// Add a directory to the `-I` or include path for headers
    pub fn include<P: AsRef<Path>>(&mut self, dir: P) -> &mut Config {
        self.include_directories.push(dir.as_ref().to_path_buf());
        self
    }

    /// Specify a `-D` variable with an optional value.
    pub fn define(&mut self, var: &str, val: Option<&str>) -> &mut Config {
        self.definitions.push((var.to_string(), val.map(|s| s.to_string())));
        self
    }

    /// Add an arbitrary object file to link in
    pub fn object<P: AsRef<Path>>(&mut self, obj: P) -> &mut Config {
        self.objects.push(obj.as_ref().to_path_buf());
        self
    }

    /// Add an arbitrary flag to the invocation of the compiler
    pub fn flag(&mut self, flag: &str) -> &mut Config {
        self.flags.push(flag.to_string());
        self
    }

    /// Add a file which will be compiled
    pub fn file<P: AsRef<Path>>(&mut self, p: P) -> &mut Config {
        self.files.push(p.as_ref().to_path_buf());
        self
    }

    /// Set C++ support.
    ///
    /// The other `cpp_*` options will only become active if this is set to
    /// `true`.
    pub fn cpp(&mut self, cpp: bool) -> &mut Config {
        self.cpp = cpp;
        self
    }

    /// Set the standard library to link against when compiling with C++
    /// support.
    ///
    /// The default value of this property depends on the current target: On
    /// OS X `Some("c++")` is used, when compiling for a Visual Studio based
    /// target `None` is used and for other targets `Some("stdc++")` is used.
    ///
    /// A value of `None` indicates that no automatic linking should happen,
    /// otherwise cargo will link against the specified library.
    ///
    /// The given library name must not contain the `lib` prefix.
    pub fn cpp_link_stdlib(&mut self, cpp_link_stdlib: Option<&str>)
                           -> &mut Config {
        self.cpp_link_stdlib = Some(cpp_link_stdlib.map(|s| s.into()));
        self
    }

    /// Force the C++ compiler to use the specified standard library.
    ///
    /// Setting this option will automatically set `cpp_link_stdlib` to the same
    /// value.
    ///
    /// The default value of this option is always `None`.
    ///
    /// This option has no effect when compiling for a Visual Studio based
    /// target.
    ///
    /// This option sets the `-stdlib` flag, which is only supported by some
    /// compilers (clang, icc) but not by others (gcc). The library will not
    /// detect which compiler is used, as such it is the responsibility of the
    /// caller to ensure that this option is only used in conjuction with a
    /// compiler which supports the `-stdlib` flag.
    ///
    /// A value of `None` indicates that no specific C++ standard library should
    /// be used, otherwise `-stdlib` is added to the compile invocation.
    ///
    /// The given library name must not contain the `lib` prefix.
    pub fn cpp_set_stdlib(&mut self, cpp_set_stdlib: Option<&str>)
                          -> &mut Config {
        self.cpp_set_stdlib = cpp_set_stdlib.map(|s| s.into());
        self.cpp_link_stdlib(cpp_set_stdlib);
        self
    }

    /// Configures the target this configuration will be compiling for.
    ///
    /// This option is automatically scraped from the `TARGET` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn target(&mut self, target: &str) -> &mut Config {
        self.target = Some(target.to_string());
        self
    }

    /// Configures the host assumed by this configuration.
    ///
    /// This option is automatically scraped from the `HOST` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn host(&mut self, host: &str) -> &mut Config {
        self.host = Some(host.to_string());
        self
    }

    /// Configures the optimization level of the generated object files.
    ///
    /// This option is automatically scraped from the `OPT_LEVEL` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn opt_level(&mut self, opt_level: u32) -> &mut Config {
        self.opt_level = Some(opt_level.to_string());
        self
    }

    /// Configures the optimization level of the generated object files.
    ///
    /// This option is automatically scraped from the `OPT_LEVEL` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn opt_level_str(&mut self, opt_level: &str) -> &mut Config {
        self.opt_level = Some(opt_level.to_string());
        self
    }

    /// Configures whether the compiler will emit debug information when
    /// generating object files.
    ///
    /// This option is automatically scraped from the `PROFILE` environment
    /// variable by build scripts (only enabled when the profile is "debug"), so
    /// it's not required to call this function.
    pub fn debug(&mut self, debug: bool) -> &mut Config {
        self.debug = Some(debug);
        self
    }

    /// Configures the output directory where all object files and static
    /// libraries will be located.
    ///
    /// This option is automatically scraped from the `OUT_DIR` environment
    /// variable by build scripts, so it's not required to call this function.
    pub fn out_dir<P: AsRef<Path>>(&mut self, out_dir: P) -> &mut Config {
        self.out_dir = Some(out_dir.as_ref().to_owned());
        self
    }

    /// Configures the compiler to be used to produce output.
    ///
    /// This option is automatically determined from the target platform or a
    /// number of environment variables, so it's not required to call this
    /// function.
    pub fn compiler<P: AsRef<Path>>(&mut self, compiler: P) -> &mut Config {
        self.compiler = Some(compiler.as_ref().to_owned());
        self
    }

    /// Configures the tool used to assemble archives.
    ///
    /// This option is automatically determined from the target platform or a
    /// number of environment variables, so it's not required to call this
    /// function.
    pub fn archiver<P: AsRef<Path>>(&mut self, archiver: P) -> &mut Config {
        self.archiver = Some(archiver.as_ref().to_owned());
        self
    }
    /// Define whether metadata should be emitted for cargo allowing it to
    /// automatically link the binary. Defaults to `true`.
    pub fn cargo_metadata(&mut self, cargo_metadata: bool) -> &mut Config {
        self.cargo_metadata = cargo_metadata;
        self
    }

    /// Configures whether the compiler will emit position independent code.
    ///
    /// This option defaults to `false` for `i686` and `windows-gnu` targets and to `true` for all
    /// other targets.
    pub fn pic(&mut self, pic: bool) -> &mut Config {
        self.pic = Some(pic);
        self
    }


    #[doc(hidden)]
    pub fn __set_env<A, B>(&mut self, a: A, b: B) -> &mut Config
        where A: AsRef<OsStr>, B: AsRef<OsStr>
    {
        self.env.push((a.as_ref().to_owned(), b.as_ref().to_owned()));
        self
    }

    /// Run the compiler, generating the file `output`
    ///
    /// The name `output` must begin with `lib` and end with `.a`
    pub fn compile(&self, output: &str) {
        assert!(output.starts_with("lib"));
        assert!(output.ends_with(".a"));
        let lib_name = &output[3..output.len() - 2];
        let dst = self.get_out_dir();

        let mut objects = Vec::new();
        let mut src_dst = Vec::new();
        for file in self.files.iter() {
            let obj = dst.join(file).with_extension("o");
            let obj = if !obj.starts_with(&dst) {
                dst.join(obj.file_name().unwrap())
            } else {
                obj
            };
            fs::create_dir_all(&obj.parent().unwrap()).unwrap();
            src_dst.push((file.to_path_buf(), obj.clone()));
            objects.push(obj);
        }
        self.compile_objects(&src_dst);
        self.assemble(lib_name, &dst.join(output), &objects);

        self.print(&format!("cargo:rustc-link-lib=static={}",
                            &output[3..output.len() - 2]));
        self.print(&format!("cargo:rustc-link-search=native={}", dst.display()));

        // Add specific C++ libraries, if enabled.
        if self.cpp {
            if let Some(stdlib) = self.get_cpp_link_stdlib() {
                self.print(&format!("cargo:rustc-link-lib={}", stdlib));
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn compile_objects(&self, objs: &[(PathBuf, PathBuf)]) {
        use self::rayon::prelude::*;

        let mut cfg = rayon::Configuration::new();
        if let Ok(amt) = env::var("NUM_JOBS") {
            if let Ok(amt) = amt.parse() {
                cfg = cfg.set_num_threads(amt);
            }
        }
        drop(rayon::initialize(cfg));

        objs.par_iter().weight_max().for_each(|&(ref src, ref dst)| {
            self.compile_object(src, dst)
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn compile_objects(&self, objs: &[(PathBuf, PathBuf)]) {
        for &(ref src, ref dst) in objs {
            self.compile_object(src, dst);
        }
    }

    fn compile_object(&self, file: &Path, dst: &Path) {
        let is_asm = file.extension().and_then(|s| s.to_str()) == Some("asm");
        let msvc = self.get_target().contains("msvc");
        let (mut cmd, name) = if msvc && is_asm {
            self.msvc_macro_assembler()
        } else {
            let compiler = self.get_compiler();
            let mut cmd = compiler.to_command();
            for &(ref a, ref b) in self.env.iter() {
                cmd.env(a, b);
            }
            (cmd, compiler.path.file_name().unwrap()
                          .to_string_lossy().into_owned())
        };
        if msvc && is_asm {
            cmd.arg("/Fo").arg(dst);
        } else if msvc {
            let mut s = OsString::from("/Fo");
            s.push(&dst);
            cmd.arg(s);
        } else {
            cmd.arg("-o").arg(&dst);
        }
        cmd.arg(if msvc {"/c"} else {"-c"});
        cmd.arg(file);

        run(&mut cmd, &name);
    }

    /// Get the compiler that's in use for this configuration.
    ///
    /// This function will return a `Tool` which represents the culmination
    /// of this configuration at a snapshot in time. The returned compiler can
    /// be inspected (e.g. the path, arguments, environment) to forward along to
    /// other tools, or the `to_command` method can be used to invoke the
    /// compiler itself.
    ///
    /// This method will take into account all configuration such as debug
    /// information, optimization level, include directories, defines, etc.
    /// Additionally, the compiler binary in use follows the standard
    /// conventions for this path, e.g. looking at the explicitly set compiler,
    /// environment variables (a number of which are inspected here), and then
    /// falling back to the default configuration.
    pub fn get_compiler(&self) -> Tool {
        let opt_level = self.get_opt_level();
        let debug = self.get_debug();
        let target = self.get_target();
        let msvc = target.contains("msvc");
        self.print(&format!("debug={} opt-level={}", debug, opt_level));

        let mut cmd = self.get_base_compiler();
        let nvcc = cmd.path.to_str()
            .map(|path| path.contains("nvcc"))
            .unwrap_or(false);

        if msvc {
            cmd.args.push("/nologo".into());
            cmd.args.push("/MD".into()); // link against msvcrt.dll for now
            match &opt_level[..] {
                "z" | "s" => cmd.args.push("/Os".into()),
                "2" => cmd.args.push("/O2".into()),
                "1" => cmd.args.push("/O1".into()),
                _ => {}
            }
            if target.contains("i686") {
                cmd.args.push("/SAFESEH".into());
            } else if target.contains("i586") {
                cmd.args.push("/SAFESEH".into());
                cmd.args.push("/ARCH:IA32".into());
            }
        } else if nvcc {
            cmd.args.push(format!("-O{}", opt_level).into());
        } else {
            cmd.args.push(format!("-O{}", opt_level).into());
            cmd.args.push("-ffunction-sections".into());
            cmd.args.push("-fdata-sections".into());
        }
        for arg in self.envflags(if self.cpp {"CXXFLAGS"} else {"CFLAGS"}) {
            cmd.args.push(arg.into());
        }

        if debug {
            cmd.args.push(if msvc {"/Z7"} else {"-g"}.into());
        }

        if target.contains("-ios") {
            self.ios_flags(&mut cmd);
        } else if !msvc {
            if target.contains("i686") || target.contains("i586") {
                cmd.args.push("-m32".into());
            } else if target.contains("x86_64") || target.contains("powerpc64") {
                cmd.args.push("-m64".into());
            }

            if !nvcc && self.pic.unwrap_or(!target.contains("i686") && !target.contains("windows-gnu")) {
                cmd.args.push("-fPIC".into());
            } else if nvcc && self.pic.unwrap_or(false) {
                cmd.args.push("-Xcompiler".into());
                cmd.args.push("\'-fPIC\'".into());
            }
            if target.contains("musl") {
                cmd.args.push("-static".into());
            }

            if target.starts_with("armv7-unknown-linux-") {
                cmd.args.push("-march=armv7-a".into());
            }
            if target.starts_with("armv7-linux-androideabi") {
                cmd.args.push("-march=armv7-a".into());
                cmd.args.push("-mfpu=vfpv3-d16".into());
            }
            if target.starts_with("arm-unknown-linux-") {
                cmd.args.push("-march=armv6".into());
                cmd.args.push("-marm".into());
            }
            if target.starts_with("i586-unknown-linux-") {
                cmd.args.push("-march=pentium".into());
            }
            if target.starts_with("i686-unknown-linux-") {
                cmd.args.push("-march=i686".into());
            }
            if target.starts_with("thumb") {
                cmd.args.push("-mthumb".into());

                if target.ends_with("eabihf") {
                    cmd.args.push("-mfloat-abi=hard".into())
                }
            }
            if target.starts_with("thumbv6m") {
                cmd.args.push("-march=armv6-m".into());
            }
            if target.starts_with("thumbv7em") {
                cmd.args.push("-march=armv7e-m".into());
            }
            if target.starts_with("thumbv7m") {
                cmd.args.push("-march=armv7-m".into());
            }
        }

        if self.cpp && !msvc {
            if let Some(ref stdlib) = self.cpp_set_stdlib {
                cmd.args.push(format!("-stdlib=lib{}", stdlib).into());
            }
        }

        for directory in self.include_directories.iter() {
            cmd.args.push(if msvc {"/I"} else {"-I"}.into());
            cmd.args.push(directory.into());
        }

        for flag in self.flags.iter() {
            cmd.args.push(flag.into());
        }

        for &(ref key, ref value) in self.definitions.iter() {
            let lead = if msvc {"/"} else {"-"};
            if let &Some(ref value) = value {
                cmd.args.push(format!("{}D{}={}", lead, key, value).into());
            } else {
                cmd.args.push(format!("{}D{}", lead, key).into());
            }
        }
        cmd
    }

    fn msvc_macro_assembler(&self) -> (Command, String) {
        let target = self.get_target();
        let tool = if target.contains("x86_64") {"ml64.exe"} else {"ml.exe"};
        let mut cmd = windows_registry::find(&target, tool).unwrap_or_else(|| {
            self.cmd(tool)
        });
        for directory in self.include_directories.iter() {
            cmd.arg("/I").arg(directory);
        }
        for &(ref key, ref value) in self.definitions.iter() {
            if let &Some(ref value) = value {
                cmd.arg(&format!("/D{}={}", key, value));
            } else {
                cmd.arg(&format!("/D{}", key));
            }
        }

        if target.contains("i686") || target.contains("i586") {
            cmd.arg("/safeseh");
        }
        for flag in self.flags.iter() {
            cmd.arg(flag);
        }

        (cmd, tool.to_string())
    }

    fn assemble(&self, lib_name: &str, dst: &Path, objects: &[PathBuf]) {
        // Delete the destination if it exists as the `ar` tool at least on Unix
        // appends to it, which we don't want.
        let _ = fs::remove_file(&dst);

        let target = self.get_target();
        if target.contains("msvc") {
            let mut cmd = match self.archiver {
                Some(ref s) => self.cmd(s),
                None => windows_registry::find(&target, "lib.exe")
                                         .unwrap_or(self.cmd("lib.exe")),
            };
            let mut out = OsString::from("/OUT:");
            out.push(dst);
            run(cmd.arg(out).arg("/nologo")
                   .args(objects)
                   .args(&self.objects), "lib.exe");

            // The Rust compiler will look for libfoo.a and foo.lib, but the
            // MSVC linker will also be passed foo.lib, so be sure that both
            // exist for now.
            let lib_dst = dst.with_file_name(format!("{}.lib", lib_name));
            let _ = fs::remove_file(&lib_dst);
            fs::hard_link(&dst, &lib_dst).or_else(|_| {
                //if hard-link fails, just copy (ignoring the number of bytes written)
                fs::copy(&dst, &lib_dst).map(|_| ())
            }).ok().expect("Copying from {:?} to {:?} failed.");;
        } else {
            let ar = self.get_ar();
            let cmd = ar.file_name().unwrap().to_string_lossy();
            run(self.cmd(&ar).arg("crs")
                                 .arg(dst)
                                 .args(objects)
                                 .args(&self.objects), &cmd);
        }
    }

    fn ios_flags(&self, cmd: &mut Tool) {
        enum ArchSpec {
            Device(&'static str),
            Simulator(&'static str),
        }

        let target = self.get_target();
        let arch = target.split('-').nth(0).unwrap();
        let arch = match arch {
            "arm" | "armv7" | "thumbv7" => ArchSpec::Device("armv7"),
            "armv7s" | "thumbv7s" => ArchSpec::Device("armv7s"),
            "arm64" | "aarch64" => ArchSpec::Device("arm64"),
            "i386" | "i686" => ArchSpec::Simulator("-m32"),
            "x86_64" => ArchSpec::Simulator("-m64"),
            _ => fail("Unknown arch for iOS target")
        };

        let sdk = match arch {
            ArchSpec::Device(arch) => {
                cmd.args.push("-arch".into());
                cmd.args.push(arch.into());
                cmd.args.push("-miphoneos-version-min=7.0".into());
                "iphoneos"
            },
            ArchSpec::Simulator(arch) => {
                cmd.args.push(arch.into());
                cmd.args.push("-mios-simulator-version-min=7.0".into());
                "iphonesimulator"
            }
        };

        self.print(&format!("Detecting iOS SDK path for {}", sdk));
        let sdk_path = self.cmd("xcrun")
            .arg("--show-sdk-path")
            .arg("--sdk")
            .arg(sdk)
            .stderr(Stdio::inherit())
            .output()
            .unwrap()
            .stdout;

        let sdk_path = String::from_utf8(sdk_path).unwrap();

        cmd.args.push("-isysroot".into());
        cmd.args.push(sdk_path.trim().into());
    }

    fn cmd<P: AsRef<OsStr>>(&self, prog: P) -> Command {
        let mut cmd = Command::new(prog);
        for &(ref a, ref b) in self.env.iter() {
            cmd.env(a, b);
        }
        return cmd
    }

    fn get_base_compiler(&self) -> Tool {
        if let Some(ref c) = self.compiler {
            return Tool::new(c.clone())
        }
        let host = self.get_host();
        let target = self.get_target();
        let (env, msvc, gnu, default) = if self.cpp {
            ("CXX", "cl.exe", "g++", "c++")
        } else {
            ("CC", "cl.exe", "gcc", "cc")
        };
        self.env_tool(env).map(|(tool, args)| {
            let mut t = Tool::new(PathBuf::from(tool));
            for arg in args {
                t.args.push(arg.into());
            }
            return t
        }).or_else(|| {
            if target.contains("emscripten") {
                if self.cpp {
                    Some(Tool::new(PathBuf::from("em++")))
                } else {
                    Some(Tool::new(PathBuf::from("emcc")))
                }
            } else {
                None
            }
        }).or_else(|| {
            windows_registry::find_tool(&target, "cl.exe")
        }).unwrap_or_else(|| {
            let compiler = if host.contains("windows") &&
                              target.contains("windows") {
                if target.contains("msvc") {
                    msvc.to_string()
                } else {
                    format!("{}.exe", gnu)
                }
            } else if target.contains("android") {
                format!("{}-{}", target, gnu)
            } else if self.get_host() != target {
                // CROSS_COMPILE is of the form: "arm-linux-gnueabi-"
                let cc_env = self.getenv("CROSS_COMPILE");
                let cross_compile = cc_env.as_ref().map(|s| s.trim_right_matches('-'));
                let prefix = cross_compile.or(match &target[..] {
                    "aarch64-unknown-linux-gnu" => Some("aarch64-linux-gnu"),
                    "arm-unknown-linux-gnueabi" => Some("arm-linux-gnueabi"),
                    "arm-unknown-linux-gnueabihf"  => Some("arm-linux-gnueabihf"),
                    "arm-unknown-linux-musleabi" => Some("arm-linux-musleabi"),
                    "arm-unknown-linux-musleabihf"  => Some("arm-linux-musleabihf"),
                    "arm-unknown-netbsdelf-eabi" => Some("arm--netbsdelf-eabi"),
                    "armv6-unknown-netbsdelf-eabihf" => Some("armv6--netbsdelf-eabihf"),
                    "armv7-unknown-linux-gnueabihf" => Some("arm-linux-gnueabihf"),
                    "armv7-unknown-linux-musleabihf" => Some("arm-linux-musleabihf"),
                    "armv7-unknown-netbsdelf-eabihf" => Some("armv7--netbsdelf-eabihf"),
                    "i686-pc-windows-gnu" => Some("i686-w64-mingw32"),
                    "i686-unknown-linux-musl" => Some("musl"),
                    "i686-unknown-netbsdelf" => Some("i486--netbsdelf"),
                    "mips-unknown-linux-gnu" => Some("mips-linux-gnu"),
                    "mipsel-unknown-linux-gnu" => Some("mipsel-linux-gnu"),
                    "mips64-unknown-linux-gnuabi64" => Some("mips64-linux-gnuabi64"),
                    "mips64el-unknown-linux-gnuabi64" => Some("mips64el-linux-gnuabi64"),
                    "powerpc-unknown-linux-gnu" => Some("powerpc-linux-gnu"),
                    "powerpc-unknown-netbsd" => Some("powerpc--netbsd"),
                    "powerpc64-unknown-linux-gnu" => Some("powerpc-linux-gnu"),
                    "powerpc64le-unknown-linux-gnu" => Some("powerpc64le-linux-gnu"),
                    "s390x-unknown-linux-gnu" => Some("s390x-linux-gnu"),
                    "thumbv6m-none-eabi" => Some("arm-none-eabi"),
                    "thumbv7em-none-eabi" => Some("arm-none-eabi"),
                    "thumbv7em-none-eabihf" => Some("arm-none-eabi"),
                    "thumbv7m-none-eabi" => Some("arm-none-eabi"),
                    "x86_64-pc-windows-gnu" => Some("x86_64-w64-mingw32"),
                    "x86_64-rumprun-netbsd" => Some("x86_64-rumprun-netbsd"),
                    "x86_64-unknown-linux-musl" => Some("musl"),
                    "x86_64-unknown-netbsd" => Some("x86_64--netbsd"),
                    _ => None,
                });
                match prefix {
                    Some(prefix) => format!("{}-{}", prefix, gnu),
                    None => default.to_string(),
                }
            } else {
                default.to_string()
            };
            Tool::new(PathBuf::from(compiler))
        })
    }

    fn get_var(&self, var_base: &str) -> Result<String, String> {
        let target = self.get_target();
        let host = self.get_host();
        let kind = if host == target {"HOST"} else {"TARGET"};
        let target_u = target.replace("-", "_");
        let res = self.getenv(&format!("{}_{}", var_base, target))
            .or_else(|| self.getenv(&format!("{}_{}", var_base, target_u)))
            .or_else(|| self.getenv(&format!("{}_{}", kind, var_base)))
            .or_else(|| self.getenv(var_base));

        match res {
            Some(res) => Ok(res),
            None => Err("could not get environment variable".to_string()),
        }
    }

    fn envflags(&self, name: &str) -> Vec<String> {
        self.get_var(name).unwrap_or(String::new())
            .split(|c: char| c.is_whitespace()).filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    fn env_tool(&self, name: &str) -> Option<(String, Vec<String>)> {
        self.get_var(name).ok().map(|tool| {
            let whitelist = ["ccache", "distcc"];
            for t in whitelist.iter() {
                if tool.starts_with(t) && tool[t.len()..].starts_with(" ") {
                    return (t.to_string(),
                            vec![tool[t.len()..].trim_left().to_string()])
                }
            }
            (tool, Vec::new())
        })
    }

    /// Returns the default C++ standard library for the current target: `libc++`
    /// for OS X and `libstdc++` for anything else.
    fn get_cpp_link_stdlib(&self) -> Option<String> {
        self.cpp_link_stdlib.clone().unwrap_or_else(|| {
            let target = self.get_target();
            if target.contains("msvc") {
                None
            } else if target.contains("darwin") {
                Some("c++".to_string())
            } else {
                Some("stdc++".to_string())
            }
        })
    }

    fn get_ar(&self) -> PathBuf {
        self.archiver.clone().or_else(|| {
            self.get_var("AR").map(PathBuf::from).ok()
        }).unwrap_or_else(|| {
            if self.get_target().contains("android") {
                PathBuf::from(format!("{}-ar", self.get_target()))
            } else if self.get_target().contains("emscripten") {
                PathBuf::from("emar")
            } else {
                PathBuf::from("ar")
            }
        })
    }

    fn get_target(&self) -> String {
        self.target.clone().unwrap_or_else(|| self.getenv_unwrap("TARGET"))
    }

    fn get_host(&self) -> String {
        self.host.clone().unwrap_or_else(|| self.getenv_unwrap("HOST"))
    }

    fn get_opt_level(&self) -> String {
        self.opt_level.as_ref().cloned().unwrap_or_else(|| {
            self.getenv_unwrap("OPT_LEVEL")
        })
    }

    fn get_debug(&self) -> bool {
        self.debug.unwrap_or_else(|| self.getenv_unwrap("PROFILE") == "debug")
    }

    fn get_out_dir(&self) -> PathBuf {
        self.out_dir.clone().unwrap_or_else(|| {
            env::var_os("OUT_DIR").map(PathBuf::from).unwrap()
        })
    }

    fn getenv(&self, v: &str) -> Option<String> {
        let r = env::var(v).ok();
        self.print(&format!("{} = {:?}", v, r));
        r
    }

    fn getenv_unwrap(&self, v: &str) -> String {
        match self.getenv(v) {
            Some(s) => s,
            None => fail(&format!("environment variable `{}` not defined", v)),
        }
    }

    fn print(&self, s: &str) {
        if self.cargo_metadata {
            println!("{}", s);
        }
    }
}

impl Tool {
    fn new(path: PathBuf) -> Tool {
        Tool {
            path: path,
            args: Vec::new(),
            env: Vec::new(),
        }
    }

    /// Converts this compiler into a `Command` that's ready to be run.
    ///
    /// This is useful for when the compiler needs to be executed and the
    /// command returned will already have the initial arguments and environment
    /// variables configured.
    pub fn to_command(&self) -> Command {
        let mut cmd = Command::new(&self.path);
        cmd.args(&self.args);
        for &(ref k, ref v) in self.env.iter() {
            cmd.env(k, v);
        }
        return cmd
    }

    /// Returns the path for this compiler.
    ///
    /// Note that this may not be a path to a file on the filesystem, e.g. "cc",
    /// but rather something which will be resolved when a process is spawned.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the default set of arguments to the compiler needed to produce
    /// executables for the target this compiler generates.
    pub fn args(&self) -> &[OsString] {
        &self.args
    }

    /// Returns the set of environment variables needed for this compiler to
    /// operate.
    ///
    /// This is typically only used for MSVC compilers currently.
    pub fn env(&self) -> &[(OsString, OsString)] {
        &self.env
    }
}

fn run(cmd: &mut Command, program: &str) {
    println!("running: {:?}", cmd);
    // Capture the standard error coming from these programs, and write it out
    // with cargo:warning= prefixes. Note that this is a bit wonky to avoid
    // requiring the output to be UTF-8, we instead just ship bytes from one
    // location to another.
    let spawn_result = match cmd.stderr(Stdio::piped()).spawn() {
        Ok(mut child) => {
            let stderr = BufReader::new(child.stderr.take().unwrap());
            for line in stderr.split(b'\n').filter_map(|l| l.ok()) {
                print!("cargo:warning=");
                std::io::stdout().write_all(&line).unwrap();
                println!("");
            }
            child.wait()
        }
        Err(e) => Err(e),
    };
    let status = match spawn_result {
        Ok(status) => status,
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
            let extra = if cfg!(windows) {
                " (see https://github.com/alexcrichton/gcc-rs#compile-time-requirements \
                   for help)"
            } else {
                ""
            };
            fail(&format!("failed to execute command: {}\nIs `{}` \
                           not installed?{}", e, program, extra));
        }
        Err(e) => fail(&format!("failed to execute command: {}", e)),
    };
    println!("{:?}", status);
    if !status.success() {
        fail(&format!("command did not execute successfully, got: {}", status));
    }
}

fn fail(s: &str) -> ! {
    println!("\n\n{}\n\n", s);
    panic!()
}
